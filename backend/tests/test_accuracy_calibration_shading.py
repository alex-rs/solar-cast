import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app.core.accuracy import calibration, shading
from app.core.clearsky import simulate_clearsky
from app.core.physics_engine import PlaneConfig, SystemConfig
from app.core.pvgis_client import flat_fallback


def _system() -> SystemConfig:
    return SystemConfig(
        system_id="test",
        lat=45.0,
        lon=0.0,
        elevation_m=100.0,
        timezone="UTC",
        planes=(PlaneConfig(dec=30, az_fs=0, kwp=1.0),),
    )


def _notch(az: pd.Series, el: pd.Series) -> pd.Series:
    inside = ((az >= 100) & (az < 110) & (el >= 10) & (el < 20)).to_numpy()
    return pd.Series(np.where(inside, 0.3, 1.0), index=az.index)


def _synthetic(days: int = 30, clear_days: int = 20, true_c: float = 0.9):
    system = _system()
    index = pd.date_range("2026-03-01", periods=days * 96, freq="15min", tz="UTC")
    clear = simulate_clearsky(index, system, flat_fallback())
    k = _notch(clear.azimuth, clear.elevation)
    factor = clear.f_d + k * clear.f_b
    actual = true_c * clear.pac * factor
    actual = actual.clip(lower=0.0)

    local_days = pd.Series(index.tz_convert("UTC").date, index=index)
    all_days = sorted(local_days.unique())
    cloudy_days = set(all_days[clear_days:])
    cloudy = local_days.isin(cloudy_days)
    rng = np.random.default_rng(7)
    scale = pd.Series(
        np.where(cloudy, rng.uniform(0.2, 0.5, len(index)), 1.0), index=index
    )
    actual = actual * scale
    telemetry = pd.DataFrame(
        {
            "mean_watts": actual,
            "energy_wh": actual * 0.25,
            "curtailed": 0,
            "period_min": 15,
        },
        index=index,
    )
    return system, clear, telemetry, all_days[:clear_days]


def test_recovers_calibration_and_shading():
    system, clear, telemetry, clear_dates = _synthetic()
    result = calibration.learn(telemetry, clear, system, "UTC", previous_c=1.0, updated_at=1)
    assert result["clear_days"] == 20
    assert result["factor_c"] == pytest.approx(0.9, abs=0.03)
    assert set(result["clear_dates"]) == set(clear_dates)

    rows = shading.learn(
        telemetry,
        clear,
        result["factor_c"],
        system,
        "UTC",
        updated_at=1,
        clear_days=result["clear_days"],
        clear_dates=set(result["clear_dates"]),
    )
    assert rows
    table = {(row["az_bucket"], row["el_bucket"]): row["k_beam"] for row in rows}
    notch = [k for (az, el), k in table.items() if 100 <= az < 110 and 10 <= el < 16]
    assert notch
    for value in notch:
        assert value < 0.5
    unaffected = [k for (az, el), k in table.items() if 160 <= az < 200 and 30 <= el < 50]
    assert unaffected
    for value in unaffected:
        assert value == pytest.approx(1.0, abs=0.05)


def test_shading_recovers_injected_notch_on_dense_samples():
    """20 clear days of dense samples at the notch angles recover K within 0.1."""
    system = _system()
    days = pd.date_range("2026-03-01", periods=20, freq="D")
    stamps = []
    azimuth, elevation = [], []
    for day in days:
        for quarter in range(4):
            stamps.append(day + pd.Timedelta(hours=6, minutes=15 * quarter))
            azimuth.append(105.0)
            elevation.append(15.0)
        for quarter in range(4):
            stamps.append(day + pd.Timedelta(hours=12, minutes=15 * quarter))
            azimuth.append(180.0)
            elevation.append(40.0)
    index = pd.DatetimeIndex(stamps).tz_localize("UTC")
    azimuth = pd.Series(azimuth, index=index)
    elevation = pd.Series(elevation, index=index)
    pac = pd.Series(800.0, index=index)
    f_b = pd.Series(0.8, index=index)
    f_d = pd.Series(0.2, index=index)
    k = pd.Series(np.where(azimuth == 105.0, 0.3, 1.0), index=index)
    actual = pac * (f_d + k * f_b)
    telemetry = pd.DataFrame(
        {"mean_watts": actual, "energy_wh": actual * 0.25, "curtailed": 0, "period_min": 15},
        index=index,
    )
    from app.core.physics_engine import SimResult

    clear = SimResult(
        pac=pac,
        pdc=pac.copy(),
        paco=1000.0,
        f_b=f_b,
        f_d=f_d,
        zenith=90.0 - elevation,
        azimuth=azimuth,
        elevation=elevation,
        planes={},
    )
    rows = shading.learn(
        telemetry, clear, 1.0, system, "UTC", 1, clear_days=20,
        clear_dates=set(days.date),
    )
    table = {(row["az_bucket"], row["el_bucket"]): row["k_beam"] for row in rows}
    assert table[(105, 14)] == pytest.approx(0.3, abs=0.1)
    assert table[(180, 40)] == pytest.approx(1.0, abs=0.05)


def test_idempotent_learning():
    system, clear, telemetry, clear_dates = _synthetic()
    first = calibration.learn(telemetry, clear, system, "UTC", 1.0, 1)
    second = calibration.learn(telemetry, clear, system, "UTC", 1.0, 1)
    assert first["factor_c"] == second["factor_c"]
    rows_a = shading.learn(
        telemetry, clear, first["factor_c"], system, "UTC", 1, first["clear_days"], set(first["clear_dates"])
    )
    rows_b = shading.learn(
        telemetry, clear, second["factor_c"], system, "UTC", 1, second["clear_days"], set(second["clear_dates"])
    )
    assert rows_a == rows_b


def test_pass_through_below_minimums():
    system, clear, telemetry, _ = _synthetic(days=30, clear_days=2)
    result = calibration.learn(telemetry, clear, system, "UTC", previous_c=1.0, updated_at=1)
    assert result["factor_c"] == 1.0
    assert result["clear_days"] < 3
    rows = shading.learn(
        telemetry, clear, 1.0, system, "UTC", 1, result["clear_days"], set(result["clear_dates"])
    )
    assert rows == []


def test_four_clear_days_learn_c_but_not_k():
    system, clear, telemetry, _ = _synthetic(days=10, clear_days=4)
    result = calibration.learn(telemetry, clear, system, "UTC", previous_c=1.0, updated_at=1)
    assert result["clear_days"] == 4
    assert result["factor_c"] == pytest.approx(0.9, abs=0.03)
    rows = shading.learn(
        telemetry, clear, result["factor_c"], system, "UTC", 1, result["clear_days"], set(result["clear_dates"])
    )
    assert rows == []


def test_shading_lookup_bilinear_and_missing_is_one():
    lookup = shading.lookup_from_rows(
        [{"az_bucket": 105, "el_bucket": 15, "k_beam": 0.4, "sample_count": 10}]
    )
    value = lookup(np.array([107.5]), np.array([16.0]))
    assert value[0] == pytest.approx(0.4, abs=0.02)
    far = lookup(np.array([0.0]), np.array([60.0]))
    assert far[0] == 1.0
