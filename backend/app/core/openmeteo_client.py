"""Open-Meteo multi-model weather client and 15-minute blender."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd
import pvlib

from .timeconv import UTC

LOG = logging.getLogger(__name__)

MINUTELY_VARS = [
    "shortwave_radiation_instant",
    "direct_normal_irradiance_instant",
    "diffuse_radiation_instant",
    "temperature_2m",
    "wind_speed_10m",
]
HOURLY_VARS = [
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "precipitation",
    "snowfall",
    "snow_depth",
    "temperature_2m",
    "wind_speed_10m",
]

MINUTELY_MAP = {
    "shortwave_radiation_instant": "ghi",
    "direct_normal_irradiance_instant": "dni",
    "diffuse_radiation_instant": "dhi",
    "temperature_2m": "temp_2m_c",
    "wind_speed_10m": "wind_10m_ms",
}
HOURLY_MAP = {
    "cloud_cover": "cloud_total",
    "cloud_cover_low": "cloud_low",
    "cloud_cover_mid": "cloud_mid",
    "cloud_cover_high": "cloud_high",
    "precipitation": "precip_mm",
    "snowfall": "snowfall_cm",
    "snow_depth": "snow_depth_m",
    "temperature_2m": "temp_2m_c",
    "wind_speed_10m": "wind_10m_ms",
}
SUM_VARS = {"precip_mm", "snowfall_cm"}

CANONICAL_COLUMNS = [
    "ghi",
    "dni",
    "dhi",
    "temp_2m_c",
    "wind_10m_ms",
    "cloud_total",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "precip_mm",
    "snowfall_cm",
    "snow_depth_m",
]


@dataclass(frozen=True)
class OpenMeteoResult:
    frame: pd.DataFrame
    elevation_m: float | None
    models: tuple[str, ...]
    raw: dict


def _utc_index(values: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _model_key(base: str, model: str, available: dict) -> str | None:
    exact = f"{base}_{model}"
    if exact in available:
        return exact
    if base in available:
        return base
    return None


def _interp_to(target: pd.DatetimeIndex, index: pd.DatetimeIndex, values: np.ndarray) -> np.ndarray:
    x = index.asi8.astype("float64")
    y = np.asarray(values, dtype="float64")
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return np.full(len(target), np.nan)
    return np.interp(target.asi8.astype("float64"), x[mask], y[mask])


def _preceding_hour_sums(
    target: pd.DatetimeIndex, hourly_index: pd.DatetimeIndex, values: np.ndarray
) -> np.ndarray:
    out = np.zeros(len(target), dtype="float64")
    positions = {ts: i for i, ts in enumerate(target)}
    for hour_start, value in zip(hourly_index, values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        for offset in range(1, 5):
            stamp = hour_start - pd.Timedelta(minutes=15 * offset)
            pos = positions.get(stamp)
            if pos is not None:
                out[pos] += float(value) / 4.0
    return out


def parse_openmeteo(
    payload: dict,
    models: list[str] | tuple[str, ...],
    weights: list[float] | tuple[float, ...],
    lat: float,
    lon: float,
) -> OpenMeteoResult:
    """Parse, join and blend a multi-model Open-Meteo forecast payload."""
    weights_arr = np.asarray(weights, dtype="float64")
    if len(weights_arr) != len(models):
        raise ValueError("models and weights must have equal length")
    m15 = payload.get("minutely_15") or {}
    hourly = payload.get("hourly") or {}
    if not m15.get("time") or not hourly.get("time"):
        raise ValueError("payload is missing minutely_15 or hourly data")

    target = _utc_index(m15["time"])
    hourly_index = _utc_index(hourly["time"])
    elevation = payload.get("elevation")

    per_model: list[pd.DataFrame] = []
    for model in models:
        frame = pd.DataFrame(index=target)
        for base, canonical in MINUTELY_MAP.items():
            key = _model_key(base, model, m15)
            if key is not None:
                frame[canonical] = pd.Series(m15[key], index=target, dtype="float64")
        for base, canonical in HOURLY_MAP.items():
            key = _model_key(base, model, hourly)
            if key is None:
                continue
            values = np.array(
                [np.nan if v is None else float(v) for v in hourly[key]], dtype="float64"
            )
            if canonical in SUM_VARS:
                joined = _preceding_hour_sums(target, hourly_index, values)
            else:
                joined = _interp_to(target, hourly_index, values)
            if canonical in ("temp_2m_c", "wind_10m_ms") and canonical in frame:
                # Prefer native minutely values; fill gaps from the hourly series.
                frame[canonical] = frame[canonical].fillna(pd.Series(joined, index=target))
            else:
                frame[canonical] = pd.Series(joined, index=target)
        for canonical in CANONICAL_COLUMNS:
            if canonical not in frame:
                frame[canonical] = np.nan
        per_model.append(frame[CANONICAL_COLUMNS])

    blended = pd.DataFrame(index=target, columns=CANONICAL_COLUMNS, dtype="float64")
    for column in CANONICAL_COLUMNS:
        values = np.vstack([frame[column].to_numpy(dtype="float64") for frame in per_model])
        valid = ~np.isnan(values)
        weighted = np.where(valid, values * weights_arr[:, None], 0.0)
        denominator = np.where(valid, weights_arr[:, None], 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(denominator > 0, weighted.sum(axis=0) / denominator, np.nan)
        blended[column] = result

    # Radiation rows with no model data are unusable; met variables are forward-filled.
    rad_cols = ["ghi", "dni", "dhi"]
    blended = blended.dropna(subset=rad_cols, how="all")
    blended[rad_cols] = blended[rad_cols].clip(lower=0)
    for column in CANONICAL_COLUMNS:
        if column not in rad_cols:
            blended[column] = blended[column].ffill()

    blended = apply_closure(blended, lat, lon, elevation)
    blended = blended.dropna(subset=rad_cols)
    blended.index.name = "time"
    return OpenMeteoResult(
        frame=blended,
        elevation_m=float(elevation) if elevation is not None else None,
        models=tuple(models),
        raw=payload,
    )


def apply_closure(
    frame: pd.DataFrame, lat: float, lon: float, elevation_m: float | None
) -> pd.DataFrame:
    """Recompute GHI from DHI + DNI*cos(zenith) after blending."""
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    try:
        position = pvlib.solarposition.get_solarposition(
            index, lat, lon, altitude=elevation_m or 0.0
        )
        cos_zenith = np.cos(np.deg2rad(position["apparent_zenith"])).clip(lower=0.0)
    except Exception:  # pragma: no cover - numerical edge cases
        cos_zenith = pd.Series(1.0, index=index)
    ghi_raw = frame["ghi"].clip(lower=0)
    dhi = frame["dhi"].clip(lower=0)
    dni = frame["dni"].clip(lower=0)
    dhi = pd.concat([dhi, ghi_raw], axis=1).min(axis=1)
    ghi = dhi + dni * cos_zenith
    out = frame.copy()
    out["dhi"] = dhi
    out["dni"] = dni
    out["ghi"] = ghi.clip(lower=0)
    return out


class OpenMeteoClient:
    """Fetching client with a small in-memory TTL cache."""

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com",
        models: list[str] | None = None,
        weights: list[float] | None = None,
        api_key: str | None = None,
        ttl_s: int = 1800,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models = models or ["icon_seamless", "ecmwf_ifs025", "gfs_seamless"]
        self.weights = weights or [0.50, 0.35, 0.15]
        self.api_key = api_key
        self.ttl_s = ttl_s
        self._client = client
        self._cache: dict[tuple[float, float], tuple[float, OpenMeteoResult]] = {}

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def build_params(self, lat: float, lon: float, forecast_days: int = 7) -> dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "minutely_15": ",".join(MINUTELY_VARS),
            "hourly": ",".join(HOURLY_VARS),
            "models": ",".join(self.models),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
            "forecast_days": forecast_days,
            "past_days": 2,
        }
        if self.api_key:
            params["apikey"] = self.api_key
        return params

    async def fetch_raw(self, lat: float, lon: float, forecast_days: int = 7) -> dict:
        http = await self._http()
        response = await http.get(
            f"{self.base_url}/v1/forecast", params=self.build_params(lat, lon, forecast_days)
        )
        response.raise_for_status()
        return response.json()

    async def get(self, lat: float, lon: float, forecast_days: int = 7) -> OpenMeteoResult:
        key = (round(lat, 3), round(lon, 3))
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < self.ttl_s:
            return cached[1]
        payload = await self.fetch_raw(lat, lon, forecast_days)
        result = parse_openmeteo(payload, self.models, self.weights, lat, lon)
        self._cache[key] = (now, result)
        return result

    def invalidate(self) -> None:
        self._cache.clear()
