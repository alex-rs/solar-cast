import time

import numpy as np
import pandas as pd
import pytest

from app.core import weights
from app.core.clearsky import clearsky_frame, clearsky_ghi, simulate_clearsky
from app.core.database import Database
from app.core.pipeline import system_config
from app.core.pvgis_client import flat_fallback
from app.core.timeconv import instant_to_period_mean

MODELS = ["alpha", "beta", "gamma"]


def test_fit_weights_recovers_mix():
    rng = np.random.default_rng(11)
    true = np.array([0.6, 0.3, 0.1])
    x = rng.uniform(0.2, 1.2, size=(500, 3))
    y = x @ true
    result = weights.fit_weights(x, y, np.array([0.5, 0.3, 0.2]))
    assert result is not None
    fitted, learned_mae, default_mae = result
    assert np.allclose(fitted, true, atol=0.03)
    assert learned_mae < default_mae


async def _seed(tmp_path, *, days=12, samples_per_day=20, mix=None):
    mix = mix or np.array([0.6, 0.3, 0.1])
    db = Database(tmp_path / "weights.db")
    await db.initialize()
    stored = {
        "system_id": "home",
        "name": "Home",
        "lat": 45.0,
        "lon": 0.0,
        "elevation_m": 100.0,
        "timezone": "UTC",
        "inverter_kw": None,
        "system_loss": 0.1,
        "gamma_pdc": -0.004,
        "damping_morning": 0.0,
        "damping_evening": 0.0,
    }
    await db.upsert_system(
        stored,
        [{"dec": 30, "az_fs": 0, "kwp": 5.0, "inverter_id": "main"}],
        [{"inverter_id": "main", "ac_kw": 1.0}],
    )
    row = await db.get_system("home")
    system = system_config(row)
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    index = pd.DatetimeIndex(
        [
            (now - pd.Timedelta(days=20) + pd.Timedelta(days=day)).replace(
                hour=8, minute=0, second=0, microsecond=0
            )
            + pd.Timedelta(minutes=15 * slot)
            for day in range(days)
            for slot in range(samples_per_day)
        ]
    )
    clear = simulate_clearsky(index, system, flat_fallback())
    clear_mean = instant_to_period_mean(clear.pac)
    ghi = clearsky_ghi(index, system)
    rng = np.random.default_rng(3)
    k_model = pd.DataFrame(
        {
            model: rng.uniform(0.3, 0.9, len(index))
            for model in MODELS
        },
        index=index,
    )
    model_ghi = k_model.mul(ghi, axis=0)
    actual = clear_mean * pd.Series(k_model.to_numpy() @ mix, index=index)
    actual = actual[np.isfinite(actual)]
    items = [
        {
            "period_start": int(stamp.timestamp()),
            "period_min": 15,
            "energy_wh": float(watts * 0.25),
            "mean_watts": float(watts),
            "curtailed": False,
            "origin": "push",
        }
        for stamp, watts in actual.items()
    ]
    await db.upsert_telemetry("home", items)
    await db.store_weather(45.0, 0.0, int(time.time()), clearsky_frame(index), model_ghi)
    return db, row


@pytest.mark.asyncio
async def test_learn_for_location_writes_normalized_weights(tmp_path):
    db, row = await _seed(tmp_path)
    learned = await weights.learn_for_location(
        db, row, flat_fallback(), MODELS, [0.5, 0.3, 0.2], int(time.time())
    )
    assert learned is not None
    assert abs(sum(learned.values()) - 1.0) < 1e-9
    for model, value in zip(MODELS, [0.6, 0.3, 0.1]):
        assert learned[model] == pytest.approx(value, abs=0.03)
    stored = await db.get_weights(45.0, 0.0)
    assert stored == pytest.approx(learned)
    assert await db.weights_updated_at(45.0, 0.0) is not None


@pytest.mark.asyncio
async def test_learn_for_location_requires_minimum_samples(tmp_path):
    db, row = await _seed(tmp_path, days=4, samples_per_day=20)
    learned = await weights.learn_for_location(
        db, row, flat_fallback(), MODELS, [0.5, 0.3, 0.2], int(time.time())
    )
    assert learned is None
    assert await db.get_weights(45.0, 0.0) is None


@pytest.mark.asyncio
async def test_learn_for_location_gate_leaves_weights_untouched(tmp_path, monkeypatch):
    db, row = await _seed(tmp_path)
    monkeypatch.setattr(
        weights,
        "fit_weights",
        lambda x, y, default: (np.array([0.6, 0.3, 0.1]), 0.2, 0.1),
    )
    learned = await weights.learn_for_location(
        db, row, flat_fallback(), MODELS, [0.5, 0.3, 0.2], int(time.time())
    )
    assert learned is None
    assert await db.get_weights(45.0, 0.0) is None
