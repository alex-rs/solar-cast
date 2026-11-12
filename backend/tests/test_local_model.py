"""Local fallback model: features, training, gating and serving."""

from __future__ import annotations

import pandas as pd
import pytest

import numpy as np

from app.core.accuracy import local_model
from app.core.clearsky import simulate_clearsky
from app.core.pipeline import system_config
from app.core.pvgis_client import flat_fallback
from app.core.database import Database

SYSTEM = {
    "system_id": "home",
    "name": "Home",
    "lat": 44.8125,
    "lon": 20.4612,
    "elevation_m": 100.0,
    "timezone": "Europe/Belgrade",
    "inverter_kw": 5.0,
    "inverters": [{"inverter_id": "main", "ac_kw": 5.0}],
    "system_loss": 0.10,
    "gamma_pdc": -0.004,
    "damping_morning": 0.0,
    "damping_evening": 0.0,
    "planes": [{"dec": 30.0, "az_fs": 0.0, "kwp": 5.0, "inverter_id": "main"}],
}


def telemetry_rows(days: int = 45, ratio: float = 0.7, seed: int = 0) -> list[dict]:
    system = system_config(SYSTEM)
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    index = pd.date_range(
        end - pd.Timedelta(days=days), end, freq="15min", inclusive="left"
    )
    clear = simulate_clearsky(index, system, flat_fallback())
    rng = np.random.default_rng(seed)
    # Autocorrelated cloudiness: without persistence in the weather, no model can
    # beat a climatological clear-sky index, and the gate rightly refuses to ship one.
    noise = rng.standard_normal(len(index))
    state = np.zeros(len(index))
    for step in range(1, len(index)):
        state[step] = 0.985 * state[step - 1] + 0.10 * noise[step]
    k = pd.Series((ratio + state).clip(0.0, 1.2), index=index)
    watts = (clear.pac * k).clip(lower=0.0)
    return [
        {
            "period_start": int(stamp.timestamp()),
            "period_min": 15,
            "energy_wh": float(value / 4.0),
            "mean_watts": float(value),
            "curtailed": False,
            "origin": "test",
        }
        for stamp, value in watts.items()
    ]


async def _database(tmp_path) -> Database:
    db = Database(tmp_path / "local.db")
    await db.initialize()
    await db.upsert_system(dict(SYSTEM), [dict(SYSTEM["planes"][0])])
    return db


def _clear_for(index: pd.DatetimeIndex):
    return simulate_clearsky(index, system_config(SYSTEM), flat_fallback())


def test_telemetry_series_drops_curtailed_rows():
    rows = telemetry_rows(days=2)
    for row in rows[:5]:
        row["curtailed"] = True
    series = local_model.telemetry_series(rows)
    assert len(series) == len(rows) - 5
    assert series.index.tz is not None
    assert (series >= 0).all()


def test_clear_sky_index_is_dimensionless_and_clipped():
    actual = pd.Series([0.5, 1.0, 2.0, 5.0])
    clear = pd.Series([0.5, 0.5, 0.5, 0.5])
    k = local_model.clear_sky_index(actual, clear)
    assert k.tolist() == pytest.approx([1.0, 1.5, 1.5, 1.5])
    # Below the daylight floor the index is undefined, not zero.
    assert local_model.clear_sky_index(pd.Series([0.0]), pd.Series([0.0])).isna().all()


def test_climatology_is_per_slot_with_fallback():
    index = pd.date_range("2024-06-01 10:00", periods=8, freq="15min", tz="UTC")
    k = pd.Series([0.8] * 4 + [0.4] * 4, index=index)
    table, fallback = local_model.climatology(k)
    series = local_model.climatology_series(
        pd.date_range("2024-06-01 10:00", periods=8, freq="15min", tz="UTC"), table, fallback
    )
    assert series.iloc[0] == pytest.approx(0.8)
    assert series.iloc[4] == pytest.approx(0.4)
    # A slot with no history falls back to the scalar median.
    other = local_model.climatology_series(
        pd.DatetimeIndex(["2024-01-15 23:45"], tz="UTC"), table, fallback
    )
    assert other.iloc[0] == pytest.approx(0.6)


def test_anchor_features_ignore_the_future():
    index = pd.date_range("2024-06-01 00:00", periods=40, freq="15min", tz="UTC")
    anchor = index[32]
    clear = _clear_for(index)
    model = local_model.LocalForecast(
        models={},
        p_max=5000.0,
        clim_table=pd.Series(dtype="float64"),
        clim_default=0.6,
        trained_at=0,
        holdout_skill={},
    )
    history = (clear.pac * 0.7).clip(lower=0.0)
    base = local_model.anchor_features(history, clear, model, anchor, now=anchor)
    assert base is not None

    mutated = clear
    mutated.pac = clear.pac.copy()
    mutated.pac.loc[mutated.pac.index > anchor] = 0.0
    changed = local_model.anchor_features(history, mutated, model, anchor, now=anchor)
    pd.testing.assert_frame_equal(base, changed)


async def test_train_registers_a_model_and_predicts(tmp_path):
    db = await _database(tmp_path)
    await db.upsert_telemetry("home", telemetry_rows(days=30))
    system = system_config(SYSTEM)
    record = await local_model.train(
        db, SYSTEM, system, flat_fallback(), tmp_path / "models"
    )
    assert record is not None
    assert record["model_type"] == "local_lgbm"
    assert record["holdout_nmae_hybrid"] < record["holdout_nmae_physics"]

    loaded = await local_model.load(db, "home")
    assert loaded is not None
    # The gate publishes only horizons that beat the clear-sky climatology curve.
    assert 60 in loaded.models
    assert loaded.horizon_min == max(loaded.models)
    assert all(h % 15 == 0 for h in loaded.models)

    index = pd.date_range(
        pd.Timestamp.now(tz="UTC").floor("15min"),
        periods=96,
        freq="15min",
    )
    clear = _clear_for(index)
    anchor = index[0]
    history = (clear.pac * 0.7).clip(lower=0.0).iloc[:5]
    features = local_model.anchor_features(history, clear, loaded, anchor, now=anchor)
    assert features is not None
    head = local_model.predict_power(
        loaded, clear, features, anchor, float(history.loc[anchor]), clear.paco
    )
    assert len(head) == len(index)
    assert (head >= 0).all()
    assert (head <= clear.paco + 1e-6).all()
    full = local_model.with_tail(loaded, clear, head)
    assert len(full) == len(clear.pac.index)
    assert (full >= 0).all()


async def test_train_skips_when_history_is_too_short(tmp_path):
    db = await _database(tmp_path)
    await db.upsert_telemetry("home", telemetry_rows(days=3))
    record = await local_model.train(
        db, SYSTEM, system_config(SYSTEM), flat_fallback(), tmp_path / "models"
    )
    assert record is None
    assert await db.active_model("home", local_model.MODEL_TYPE) is None


async def test_gate_rejects_a_model_that_cannot_beat_climatology(tmp_path, monkeypatch):
    db = await _database(tmp_path)
    await db.upsert_telemetry("home", telemetry_rows(days=30))
    monkeypatch.setattr(local_model, "GATE_RATIO", -1.0)
    record = await local_model.train(
        db, SYSTEM, system_config(SYSTEM), flat_fallback(), tmp_path / "models"
    )
    assert record is None
    assert await db.active_model("home", local_model.MODEL_TYPE) is None


async def test_models_are_scoped_by_type(tmp_path):
    """A local model must never be returned as the MOS model, or vice versa."""
    db = await _database(tmp_path)
    await db.upsert_telemetry("home", telemetry_rows(days=30))
    await local_model.train(db, SYSTEM, system_config(SYSTEM), flat_fallback(), tmp_path / "models")
    await db.register_model(
        {
            "system_id": "home",
            "model_type": "mos_lgbm",
            "trained_at": 1,
            "artifact_path": str(tmp_path / "nope.joblib"),
            "train_rows": 1,
            "holdout_rows": 1,
            "holdout_nmae_physics": 1.0,
            "holdout_nmae_hybrid": 0.9,
            "active": True,
        }
    )
    assert (await db.active_model("home", "mos_lgbm"))["model_type"] == "mos_lgbm"
    assert (await db.active_model("home", local_model.MODEL_TYPE))["model_type"] == "local_lgbm"
