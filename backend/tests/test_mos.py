import asyncio
import datetime as dt

import numpy as np
import pandas as pd
import pvlib
import pytest

from app.core.accuracy import mos
from app.core.clearsky import simulate_clearsky
from app.core.database import Database
from app.core.physics_engine import PlaneConfig, SystemConfig
from app.core.pvgis_client import flat_fallback


def _build_dataset(tmp_path, pattern: str):
    db_path = tmp_path / f"{pattern}.db"
    db = Database(db_path)
    asyncio.run(db.initialize())
    system_row = {
        "system_id": "home",
        "name": "Home",
        "lat": 45.0,
        "lon": 0.0,
        "elevation_m": 100.0,
        "timezone": "UTC",
        "inverter_kw": 5.0,
        "system_loss": 0.10,
        "gamma_pdc": -0.004,
        "damping_morning": 0.0,
        "damping_evening": 0.0,
        "planes": [],
    }
    asyncio.run(
        db.upsert_system(
            {k: v for k, v in system_row.items() if k != "planes"},
            [{"dec": 30, "az_fs": 0, "kwp": 5.0}],
        )
    )
    stored = asyncio.run(db.get_system("home"))
    system = SystemConfig(
        system_id="home",
        lat=45.0,
        lon=0.0,
        elevation_m=100.0,
        inverter_kw=5.0,
        timezone="UTC",
        planes=(PlaneConfig(dec=30, az_fs=0, kwp=5.0),),
    )
    horizon = flat_fallback()
    rng = np.random.default_rng(42)
    start = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    for day in range(40):
        day_start = start + dt.timedelta(days=day)
        fetched_at = int((day_start + dt.timedelta(hours=4)).timestamp())
        index = pd.date_range(day_start + dt.timedelta(hours=8), periods=32, freq="15min", tz="UTC")
        clear = simulate_clearsky(index, system, horizon)
        location = pvlib.location.Location(45.0, 0.0, "UTC", 100.0)
        sky = location.get_clearsky(index, model="ineichen")
        cloud_mid = pd.Series(
            [80.0 if (hour % 2 == 0 and day % 2 == 0) else 10.0 for hour in range(32)],
            index=index,
        )
        weather = pd.DataFrame(
            {
                "ghi": sky["ghi"].to_numpy(),
                "dni": sky["dni"].to_numpy(),
                "dhi": sky["dhi"].to_numpy(),
                "temp_2m_c": 20.0,
                "wind_10m_ms": 2.0,
                "cloud_total": cloud_mid,
                "cloud_low": 5.0,
                "cloud_mid": cloud_mid,
                "cloud_high": 5.0,
                "precip_mm": 0.0,
                "snowfall_cm": 0.0,
                "snow_depth_m": 0.0,
            },
            index=index,
        )
        asyncio.run(db.store_weather(45.0, 0.0, fetched_at, weather))
        physics = clear.pac
        actual = physics.copy()
        if pattern == "bias":
            actual = physics + np.where(cloud_mid > 60, 0.08 * physics, 0.0)
        else:
            actual = physics + rng.normal(0.0, 100.0, len(index))
        actual = pd.Series(np.clip(actual, 0, 5000), index=index)
        telemetry = [
            {
                "period_start": int(ts.timestamp()),
                "period_min": 15,
                "energy_wh": float(value * 0.25),
                "mean_watts": float(value),
                "curtailed": 0,
                "origin": "push",
            }
            for ts, value in actual.items()
        ]
        asyncio.run(db.upsert_telemetry("home", telemetry))
        runs = [
            {
                "system_id": "home",
                "fetched_at": fetched_at,
                "period_start": int(ts.timestamp()),
                "period_min": 15,
                "lead_hours": (ts.timestamp() - fetched_at) / 3600.0,
                "source": "physics",
                "energy_wh": float(value * 0.25),
                "mean_watts": float(value),
            }
            for ts, value in physics.items()
        ]
        asyncio.run(db.insert_runs(runs))
    return db, stored, system, horizon


def test_gate_rejects_noise_and_accepts_bias(tmp_path):
    noise_db, noise_row, noise_system, noise_horizon = _build_dataset(tmp_path, "noise")
    assert asyncio.run(asyncio.to_thread(
        mos.train, noise_db, noise_row, noise_system, noise_horizon, tmp_path / "models"
    )) is None
    assert asyncio.run(noise_db.active_model("home")) is None

    bias_db, bias_row, bias_system, bias_horizon = _build_dataset(tmp_path, "bias")
    record = asyncio.run(asyncio.to_thread(
        mos.train, bias_db, bias_row, bias_system, bias_horizon, tmp_path / "models"
    ))
    assert record is not None
    assert record["holdout_nmae_hybrid"] < 0.98 * record["holdout_nmae_physics"]
    active = asyncio.run(bias_db.active_model("home"))
    assert active is not None
    assert active["artifact_path"] == record["artifact_path"]


def test_below_minimum_days_is_pass_through(tmp_path):
    db, row, system, horizon = _build_dataset(tmp_path, "noise")
    cutoff = int(dt.datetime(2026, 6, 11, tzinfo=dt.timezone.utc).timestamp())
    asyncio.run(
        db.execute(
            "DELETE FROM actual_telemetry WHERE period_start >= ?", (cutoff,)
        )
    )
    assert asyncio.run(asyncio.to_thread(
        mos.train, db, row, system, horizon, tmp_path / "models"
    )) is None
