"""Stateless PV physics: weather + system geometry -> instantaneous AC power."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd
import pvlib

from .pvgis_client import Horizon
from .timeconv import fs_az_to_pvlib

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaneConfig:
    dec: float
    az_fs: float
    kwp: float
    plane_id: str | None = None


@dataclass(frozen=True)
class SystemConfig:
    system_id: str
    lat: float
    lon: float
    elevation_m: float = 0.0
    inverter_kw: float | None = None
    system_loss: float = 0.10
    gamma_pdc: float = -0.004
    damping_morning: float = 0.0
    damping_evening: float = 0.0
    timezone: str = "UTC"
    planes: tuple[PlaneConfig, ...] = field(default_factory=tuple)

    @property
    def kwp_total(self) -> float:
        return sum(plane.kwp for plane in self.planes)


class ShadingLookup(Protocol):
    def __call__(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray: ...


@dataclass
class SimResult:
    pac: pd.Series
    pdc: pd.Series
    paco: float
    f_b: pd.Series
    f_d: pd.Series
    zenith: pd.Series
    azimuth: pd.Series
    elevation: pd.Series
    planes: dict[str, pd.DataFrame]


def _prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    frame = weather.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("weather index must be timezone aware")
    frame.index = index.tz_convert("UTC")
    frame = frame.sort_index()
    for column in ("ghi", "dni", "dhi", "temp_2m_c", "wind_10m_ms", "snowfall_cm", "snow_depth_m"):
        if column not in frame:
            frame[column] = np.nan
    return frame


def _solar_position(frame: pd.DataFrame, system: SystemConfig) -> pd.DataFrame:
    return pvlib.solarposition.get_solarposition(
        frame.index, system.lat, system.lon, altitude=system.elevation_m or 0.0
    )


def _clearsky(frame: pd.DataFrame, system: SystemConfig, position: pd.DataFrame) -> pd.DataFrame:
    location = pvlib.location.Location(
        system.lat, system.lon, "UTC", system.elevation_m or 0.0
    )
    clear = location.get_clearsky(frame.index, model="ineichen", solar_position=position)
    out = frame.copy()
    out["ghi"] = clear["ghi"].to_numpy()
    out["dni"] = clear["dni"].to_numpy()
    out["dhi"] = clear["dhi"].to_numpy()
    return out


def _snow_coverage(
    frame: pd.DataFrame, tilt: float, poa_global: pd.Series
) -> pd.Series:
    """NREL snow coverage on an hourly state, interpolated back to 15 min."""
    if frame.empty:
        return pd.Series(0.0, index=frame.index)
    hourly = pd.DataFrame(
        {
            "snowfall": frame["snowfall_cm"].resample("1h").sum(min_count=1),
            "snow_depth": frame["snow_depth_m"].resample("1h").mean(),
            "temp": frame["temp_2m_c"].resample("1h").mean(),
            "poa": poa_global.resample("1h").mean(),
        }
    )
    hourly = hourly.dropna(subset=["snowfall", "snow_depth", "temp", "poa"], how="all")
    if hourly.empty:
        return pd.Series(0.0, index=frame.index)
    hourly = hourly.copy()
    hourly["snowfall"] = hourly["snowfall"].fillna(0.0)
    hourly["snow_depth"] = hourly["snow_depth"].fillna(0.0)
    hourly["temp"] = hourly["temp"].ffill().bfill().fillna(15.0)
    hourly["poa"] = hourly["poa"].ffill().bfill().fillna(0.0)
    coverage = pvlib.snow.coverage_nrel(
        hourly["snowfall"].astype("float64"),
        hourly["poa"].astype("float64"),
        hourly["temp"].astype("float64"),
        tilt,
        snow_depth=(hourly["snow_depth"] * 100.0).astype("float64"),
        initial_coverage=0,
        threshold_snowfall=1.0,
    )
    hourly_series = pd.Series(np.asarray(coverage, dtype="float64"), index=hourly.index)
    return (
        hourly_series.reindex(hourly_series.index.union(frame.index))
        .interpolate(method="time")
        .reindex(frame.index)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )


def _damping_factors(frame: pd.DataFrame, system: SystemConfig) -> pd.Series:
    if system.damping_morning == 0.0 and system.damping_evening == 0.0:
        return pd.Series(1.0, index=frame.index)
    try:
        tz = dt.timezone.utc if system.timezone == "UTC" else __import__("zoneinfo").ZoneInfo(
            system.timezone
        )
    except Exception:
        tz = dt.timezone.utc
    local_dates = pd.DatetimeIndex(frame.index).tz_convert(tz)
    unique_days = sorted({ts.date() for ts in local_dates})
    noon_index = pd.DatetimeIndex(
        [pd.Timestamp(dt.datetime.combine(day, dt.time(12), tzinfo=tz)) for day in unique_days]
    ).tz_convert("UTC")
    try:
        events = pvlib.solarposition.sun_rise_set_transit_spa(noon_index, system.lat, system.lon)
    except Exception as exc:  # pragma: no cover - polar / numerical edge cases
        LOG.warning("damping skipped: %s", exc)
        return pd.Series(1.0, index=frame.index)

    factors = np.ones(len(frame), dtype="float64")
    event_by_date = {
        idx.date(): (row.sunrise, row.transit, row.sunset)
        for idx, row in events.iterrows()
    }
    timestamps = pd.DatetimeIndex(frame.index).tz_convert("UTC")
    for i, (stamp, local_stamp) in enumerate(zip(timestamps, local_dates)):
        event = event_by_date.get(local_stamp.date())
        if event is None:
            continue
        sunrise, transit, sunset = event
        if pd.isna(sunrise) or pd.isna(sunset) or pd.isna(transit):
            continue
        if stamp < transit:
            span = (transit - sunrise).total_seconds()
            if span <= 0:
                continue
            factor = 1.0 - system.damping_morning * (
                (transit - stamp).total_seconds() / span
            )
        else:
            span = (sunset - transit).total_seconds()
            if span <= 0:
                continue
            factor = 1.0 - system.damping_evening * (
                (stamp - transit).total_seconds() / span
            )
        factors[i] = min(max(factor, 0.0), 1.0)
    return pd.Series(factors, index=frame.index)


def simulate_system(
    weather: pd.DataFrame,
    system: SystemConfig,
    horizon: Horizon,
    shading: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    calibration_c: float = 1.0,
    clear_sky: bool = False,
) -> SimResult:
    """Run the full physics chain for one system.

    Radiation is instantaneous; energy periods are produced by the caller with
    the trapezoid rule over this output.
    """
    if not system.planes:
        raise ValueError("system has no planes")
    frame = _prepare_weather(weather)
    if frame.empty:
        empty = pd.Series(dtype="float64")
        return SimResult(
            pac=empty,
            pdc=empty,
            paco=0.0,
            f_b=empty,
            f_d=empty,
            zenith=empty,
            azimuth=empty,
            elevation=empty,
            planes={},
        )

    position = _solar_position(frame, system)
    if clear_sky:
        frame = _clearsky(frame, system, position)
    else:
        frame["temp_2m_c"] = frame["temp_2m_c"].ffill().bfill().fillna(15.0)
        frame["wind_10m_ms"] = frame["wind_10m_ms"].ffill().bfill().fillna(1.0)

    zenith = position["apparent_zenith"].astype("float64")
    elevation_angle = position["apparent_elevation"].astype("float64")
    azimuth = position["azimuth"].astype("float64")
    cos_zenith = np.cos(np.deg2rad(zenith)).clip(lower=0.0)

    horizon_el = horizon.lookup(azimuth.to_numpy())
    blocked = elevation_angle.to_numpy() <= horizon_el
    dni = frame["dni"].to_numpy(dtype="float64").copy()
    dni = np.where(np.isnan(dni), 0.0, dni)
    dni_eff = np.where(blocked, 0.0, dni)

    if shading is not None:
        dni_eff = dni_eff * np.asarray(shading(azimuth.to_numpy(), elevation_angle.to_numpy()))

    ghi_eff = frame["dhi"].to_numpy(dtype="float64") + dni_eff * cos_zenith.to_numpy()
    ghi_eff = np.clip(np.nan_to_num(ghi_eff), 0.0, None)
    dhi = np.nan_to_num(frame["dhi"].to_numpy(dtype="float64"))

    snow_depth = np.nan_to_num(frame["snow_depth_m"].to_numpy(dtype="float64"))
    albedo = pd.Series(
        np.where(snow_depth > 0.02, 0.60, 0.20), index=frame.index, dtype="float64"
    )

    dni_extra = pvlib.irradiance.get_extra_radiation(frame.index)
    airmass = pvlib.atmosphere.get_relative_airmass(zenith).fillna(38.0)
    damping = _damping_factors(frame, system)

    up = (zenith.to_numpy() < 90.0) & np.isfinite(zenith.to_numpy())
    pdc_total = np.zeros(len(frame), dtype="float64")
    f_b_numerator = np.zeros(len(frame), dtype="float64")
    planes: dict[str, pd.DataFrame] = {}
    all_poa = np.zeros(len(frame), dtype="float64")

    for ordinal, plane in enumerate(system.planes, start=1):
        plane_id = plane.plane_id or f"plane{ordinal}"
        tilt = plane.dec
        surface_azimuth = fs_az_to_pvlib(plane.az_fs)
        poa = pvlib.irradiance.get_total_irradiance(
            tilt,
            surface_azimuth,
            zenith,
            azimuth,
            dni_eff,
            ghi_eff,
            dhi,
            dni_extra=dni_extra,
            airmass=airmass,
            albedo=albedo,
            model="perez",
        )
        aoi = pvlib.irradiance.aoi(tilt, surface_azimuth, zenith, azimuth)
        iam_b = pvlib.iam.ashrae(aoi, b=0.05).fillna(0.0)
        iam_diffuse = pvlib.iam.marion_diffuse("ashrae", tilt, b=0.05)
        e_beam = poa["poa_direct"].to_numpy(dtype="float64") * iam_b.to_numpy()
        e_diff = (
            poa["poa_sky_diffuse"].to_numpy(dtype="float64") * iam_diffuse["sky"]
            + poa["poa_ground_diffuse"].to_numpy(dtype="float64") * iam_diffuse["ground"]
        )
        coverage = _snow_coverage(frame, tilt, poa["poa_global"])
        snow_factor = 1.0 - coverage.to_numpy()
        e_beam = np.nan_to_num(e_beam) * snow_factor
        e_diff = np.nan_to_num(e_diff) * snow_factor
        e_total = np.clip(e_beam + e_diff, 0.0, None)
        e_total = np.where(up, e_total, 0.0)
        e_beam = np.where(up, e_beam, 0.0)
        e_diff = np.where(up, e_diff, 0.0)

        poa_global = e_total
        cell_temp = pvlib.temperature.faiman(
            pd.Series(poa_global, index=frame.index),
            frame["temp_2m_c"],
            frame["wind_10m_ms"],
        )
        pdc = pvlib.pvsystem.pvwatts_dc(
            pd.Series(e_total, index=frame.index),
            cell_temp,
            plane.kwp * 1000.0 * calibration_c,
            system.gamma_pdc,
        )
        pdc = pdc.fillna(0.0).to_numpy(dtype="float64")
        pdc = pdc * (1.0 - system.system_loss) * damping.to_numpy()
        pdc = np.where(up, pdc, 0.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            beam_fraction = np.where(e_total > 0, e_beam / e_total, np.nan)
        pdc_total += pdc
        all_poa += poa_global
        f_b_numerator += pdc * np.nan_to_num(beam_fraction)

        planes[plane_id] = pd.DataFrame(
            {
                "poa_global": poa_global,
                "e_beam": e_beam,
                "e_diff": e_diff,
                "pdc": pdc,
                "cell_temp_c": np.asarray(cell_temp, dtype="float64"),
                "snow_coverage": coverage.to_numpy(),
                "dni_eff": dni_eff,
            },
            index=frame.index,
        )

    kwp_total = system.kwp_total
    pdc_series = pd.Series(pdc_total, index=frame.index, name="pdc")
    paco = (
        float(system.inverter_kw) * 1000.0
        if system.inverter_kw is not None
        else float(kwp_total) * 1000.0
    )
    pac = pvlib.inverter.pvwatts(pdc_series, paco / 0.96, eta_inv_nom=0.96)
    pac = pd.Series(pac, index=frame.index, dtype="float64").fillna(0.0)
    pac = pac.clip(lower=0.0, upper=paco)
    pac = pac.where(pd.Series(up, index=frame.index), 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        f_b = np.where(pdc_total > 0, f_b_numerator / pdc_total, np.nan)
    f_b_series = pd.Series(f_b, index=frame.index, name="f_b")
    f_d_series = (1.0 - f_b_series).rename("f_d")

    return SimResult(
        pac=pac.rename("pac"),
        pdc=pdc_series,
        paco=paco,
        f_b=f_b_series,
        f_d=f_d_series,
        zenith=zenith.rename("zenith"),
        azimuth=azimuth.rename("azimuth"),
        elevation=elevation_angle.rename("elevation"),
        planes=planes,
    )
