import numpy as np
import pandas as pd
import pvlib
import pytest

from app.core.physics_engine import PlaneConfig, SystemConfig, simulate_system
from app.core.pvgis_client import Horizon, flat_fallback
from app.core.timeconv import instant_energy


def _system(planes, **kwargs) -> SystemConfig:
    return SystemConfig(
        system_id="test",
        lat=kwargs.pop("lat", 45.0),
        lon=kwargs.pop("lon", 0.0),
        elevation_m=100.0,
        timezone="UTC",
        planes=tuple(planes),
        **kwargs,
    )


def _weather(start, end) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    return pd.DataFrame(
        {
            "ghi": 0.0,
            "dni": 0.0,
            "dhi": 0.0,
            "temp_2m_c": 20.0,
            "wind_10m_ms": 2.0,
            "cloud_total": 0.0,
            "cloud_low": 0.0,
            "cloud_mid": 0.0,
            "cloud_high": 0.0,
            "precip_mm": 0.0,
            "snowfall_cm": 0.0,
            "snow_depth_m": 0.0,
        },
        index=idx,
    )


def _daily_yield(result, day, tz="UTC") -> float:
    energy = instant_energy(result.pac)
    local = energy.index.tz_convert(tz)
    mask = local.date == pd.Timestamp(day).date()
    return float(energy[mask].sum()) / 1000.0


def test_june_clear_sky_south():
    system = _system([PlaneConfig(dec=30, az_fs=0, kwp=1.0)])
    weather = _weather("2026-06-20 22:00", "2026-06-22 00:00")
    result = simulate_system(weather, system, flat_fallback(), clear_sky=True)
    assert not result.pac.isna().any()
    assert (result.pac >= 0).all()
    nightly = result.pac[result.zenith >= 90]
    assert nightly.max() == 0.0
    # Peak at solar transit +- 15 minutes.
    transit = pvlib.solarposition.sun_rise_set_transit_spa(
        pd.DatetimeIndex([pd.Timestamp("2026-06-21 12:00Z")]), 45.0, 0.0
    )["transit"].iloc[0]
    peak = result.pac.idxmax()
    assert abs((peak - transit).total_seconds()) <= 15 * 60
    yield_kwh = _daily_yield(result, "2026-06-21")
    assert 5.5 <= yield_kwh <= 8.0


def test_december_clear_sky_south():
    system = _system([PlaneConfig(dec=30, az_fs=0, kwp=1.0)])
    weather = _weather("2026-12-20 22:00", "2026-12-22 00:00")
    result = simulate_system(weather, system, flat_fallback(), clear_sky=True)
    yield_kwh = _daily_yield(result, "2026-12-21")
    assert 1.0 <= yield_kwh <= 3.5


def test_east_west_share_one_inverter_and_clip():
    system = _system(
        [PlaneConfig(dec=30, az_fs=-90, kwp=2.5), PlaneConfig(dec=30, az_fs=90, kwp=2.5)],
        inverter_kw=2.0,
    )
    weather = _weather("2026-06-20 22:00", "2026-06-22 00:00")
    result = simulate_system(weather, system, flat_fallback(), clear_sky=True)
    assert result.pac.max() <= 2000.0 + 1e-6
    assert result.pac.max() > 1900.0
    # DC is summed across planes before inversion; individual planes are not clipped.
    assert max(plane["pdc"].max() for plane in result.planes.values()) > 2000.0 - 100.0
    east_peak = result.planes["plane1"]["pdc"].idxmax()
    west_peak = result.planes["plane2"]["pdc"].idxmax()
    assert east_peak < west_peak


def test_east_and_west_peaks_straddle_transit():
    transit = pvlib.solarposition.sun_rise_set_transit_spa(
        pd.DatetimeIndex([pd.Timestamp("2026-06-21 12:00Z")]), 45.0, 0.0
    )["transit"].iloc[0]
    system = _system(
        [PlaneConfig(dec=30, az_fs=-90, kwp=2.5), PlaneConfig(dec=30, az_fs=90, kwp=2.5)]
    )
    result = simulate_system(
        _weather("2026-06-21 00:00", "2026-06-22 00:00"), system, flat_fallback(), clear_sky=True
    )
    assert result.planes["plane1"]["poa_global"].idxmax() < transit
    assert result.planes["plane2"]["poa_global"].idxmax() > transit


def test_terrain_horizon_blocks_morning_east_production():
    system = _system([PlaneConfig(dec=30, az_fs=-90, kwp=2.5)])
    weather = _weather("2026-06-21 00:00", "2026-06-22 00:00")
    flat = simulate_system(weather, system, flat_fallback(), clear_sky=True)
    az = np.arange(0.0, 361.0, 1.0)
    elevation = np.where((az >= 60) & (az <= 120), 30.0, 0.0)
    blocked = Horizon(azimuth=az, elevation=elevation, source="test")
    walled = simulate_system(weather, system, blocked, clear_sky=True)
    first_flat = flat.planes["plane1"]["e_beam"][flat.planes["plane1"]["e_beam"] > 5.0].index[0]
    first_walled = walled.planes["plane1"]["e_beam"][
        walled.planes["plane1"]["e_beam"] > 5.0
    ].index[0]
    assert first_walled > first_flat
    assert walled.pac.sum() < flat.pac.sum()


def test_snow_coverage_reduces_output_and_decays():
    system = _system([PlaneConfig(dec=30, az_fs=0, kwp=2.5)])
    weather = _weather("2026-06-21 00:00", "2026-06-22 00:00")
    clean = simulate_system(weather, system, flat_fallback(), clear_sky=True)

    snowy = weather.copy()
    snowy.loc[pd.Timestamp("2026-06-21 00:00Z"), "snowfall_cm"] = 20.0
    snowy.loc[:, "snow_depth_m"] = 0.20
    snowy.loc[:, "temp_2m_c"] = -8.0
    snowed = simulate_system(snowy, system, flat_fallback(), clear_sky=True)
    coverage = snowed.planes["plane1"]["snow_coverage"]
    assert coverage.max() > 0.5
    assert snowed.pac.sum() < clean.pac.sum()

    melting = snowy.copy()
    melting.loc[:, "temp_2m_c"] = 25.0
    melting.loc[:, "snow_depth_m"] = 0.0
    melted = simulate_system(melting, system, flat_fallback(), clear_sky=True)
    assert melted.planes["plane1"]["snow_coverage"].iloc[-1] < coverage.iloc[-1]


def test_perez_requires_dni_extra():
    idx = pd.date_range("2026-06-21 10:00", periods=1, freq="15min", tz="UTC")
    position = pvlib.solarposition.get_solarposition(idx, 45.0, 0.0)
    with pytest.raises(ValueError, match="dni_extra"):
        pvlib.irradiance.get_total_irradiance(
            45,
            180,
            position["apparent_zenith"],
            position["azimuth"],
            pd.Series(800.0, index=idx),
            pd.Series(900.0, index=idx),
            pd.Series(100.0, index=idx),
            model="perez",
        )


def test_calibration_factor_scales_output_linearly():
    system = _system([PlaneConfig(dec=30, az_fs=0, kwp=2.5)])
    weather = _weather("2026-06-21 08:00", "2026-06-21 16:00")
    base = simulate_system(weather, system, flat_fallback(), clear_sky=True)
    calibrated = simulate_system(weather, system, flat_fallback(), calibration_c=0.9, clear_sky=True)
    mask = base.pdc > 100
    ratio = (calibrated.pdc[mask] / base.pdc[mask]).dropna()
    assert ratio.median() == pytest.approx(0.9, rel=0.02)
