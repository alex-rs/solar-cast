"""Fallback tiers end to end: remote -> stale snapshot -> local model -> climatology."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from app.core.accuracy import local_model
from app.core.pipeline import system_config
from tests.conftest import system_body, synthetic_weather
from tests.test_local_model import telemetry_rows

pytestmark = pytest.mark.asyncio

LAYER_KEYS = {"calibration", "shading", "mos", "nowcast"}


async def _create_system(api, **overrides):
    response = await api.put("/api/v1/systems/home", json=system_body(**overrides))
    assert response.status_code == 200, response.text
    return response.json()


def _kill_weather(api, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("weather endpoint unreachable")

    monkeypatch.setattr(api.app.state.pipeline.weather, "get", boom)


async def _train_local(api, days: int = 45) -> dict:
    db = api.app.state.db
    await db.upsert_telemetry("home", telemetry_rows(days=days))
    system_row = await db.get_system("home")
    pipeline = api.app.state.pipeline
    record = await local_model.train(
        db,
        system_row,
        system_config(system_row),
        await pipeline.offline_horizon(system_row),
        pipeline.settings.models_dir,
    )
    assert record is not None
    return record


async def test_remote_path_keeps_provenance_remote(api):
    await _create_system(api)
    payload = (await api.get("/api/v1/systems/home/forecast")).json()
    assert payload["provenance"] == "remote"
    assert payload["degraded"] is False
    assert payload["reason"] is None
    assert payload["local_horizon_minutes"] == 0
    assert all(period["confidence"] is None for period in payload["periods"])
    # The published layer set is unchanged, so the compat route stays byte-identical.
    assert set(payload["layers_active"]) == LAYER_KEYS


async def test_weather_failure_without_model_returns_climatology(api, monkeypatch):
    await _create_system(api)
    _kill_weather(api, monkeypatch)

    response = await api.get("/api/v1/systems/home/forecast")
    assert response.status_code == 200
    payload = response.json()
    assert payload["provenance"] == "clearsky_climatology"
    assert payload["degraded"] is True
    assert payload["reason"] == "no_telemetry"
    assert payload["local_horizon_minutes"] == 0
    assert len(payload["periods"]) > 0
    assert all(period["confidence"] == "low" for period in payload["periods"])
    assert set(payload["layers_active"]) == LAYER_KEYS


async def test_weather_failure_with_model_uses_local_layer(api, monkeypatch):
    await _create_system(api)
    record = await _train_local(api)
    _kill_weather(api, monkeypatch)

    payload = (await api.get("/api/v1/systems/home/forecast")).json()
    assert payload["provenance"] == "local_model"
    assert payload["degraded"] is True
    assert payload["reason"] is None

    model = await local_model.load(api.app.state.db, "home")
    assert model is not None
    assert payload["local_horizon_minutes"] == model.horizon_min >= 60
    assert payload["learning"]["local"] is True
    assert payload["learning"]["local_skill"] is not None
    assert payload["learning"]["local_trained_at"] == record["trained_at"]

    # Confidence follows the published horizon exactly, whatever time the test runs:
    # after sunset every remaining period is beyond it.
    cutoff = pd.Timestamp(payload["generated_at"]) + pd.Timedelta(
        minutes=payload["local_horizon_minutes"]
    )
    for period in payload["periods"]:
        start = pd.Timestamp(period["start"]).tz_convert("UTC")
        assert period["confidence"] == ("low" if start > cutoff else None)
    assert any(period["confidence"] == "low" for period in payload["periods"])
    assert any(period["mean_watts"] > 0 for period in payload["periods"])


async def test_stale_snapshot_is_flagged_degraded(api, monkeypatch):
    await _create_system(api)
    db = api.app.state.db
    system = await db.get_system("home")
    await db.store_weather(
        system["lat"], system["lon"], int(time.time()) - 2 * 3600, synthetic_weather()
    )
    _kill_weather(api, monkeypatch)

    payload = (await api.get("/api/v1/systems/home/forecast")).json()
    assert payload["provenance"] == "remote_stale"
    assert payload["degraded"] is True
    assert payload["reason"] == "remote_stale"
    assert len(payload["periods"]) > 0


async def test_global_flag_disables_the_fallback(api, monkeypatch):
    await _create_system(api)
    api.app.state.settings.enable_local_forecast = False
    _kill_weather(api, monkeypatch)
    with pytest.raises(RuntimeError):
        await api.get("/api/v1/systems/home/forecast")


async def test_per_system_toggle_disables_the_fallback(api, monkeypatch):
    created = await _create_system(api, local_fallback=False)
    assert created["local_fallback"] is False
    _kill_weather(api, monkeypatch)
    with pytest.raises(RuntimeError):
        await api.get("/api/v1/systems/home/forecast")


async def test_stale_telemetry_falls_back_to_climatology(api, monkeypatch):
    """A trained model is useless without recent telemetry; say so instead of pretending."""
    await _create_system(api)
    await _train_local(api)
    db = api.app.state.db
    now = int(time.time())
    await db.delete_telemetry("home", now - 4 * 3600, now + 3600)
    # One row inside the a six-hour window but older than the anchor limit.
    stale = now - 4 * 3600
    await db.upsert_telemetry(
        "home",
        [
            {
                "period_start": stale,
                "period_min": 15,
                "energy_wh": 125.0,
                "mean_watts": 500.0,
                "curtailed": False,
                "origin": "push",
            }
        ],
    )

    _kill_weather(api, monkeypatch)
    payload = (await api.get("/api/v1/systems/home/forecast")).json()
    assert payload["provenance"] == "clearsky_climatology"
    assert payload["reason"] == "no_recent_telemetry"
    assert payload["local_horizon_minutes"] == 0
    assert payload["learning"]["local"] is False
    assert payload["learning"]["local_skill"] is None


async def test_benchmark_snapshots_ignore_the_local_layer(api, monkeypatch):
    await _create_system(api)
    await _train_local(api)
    _kill_weather(api, monkeypatch)
    written = await api.app.state.pipeline.snapshot_layers()
    assert written == 0
    rows = await api.app.state.db.query("SELECT COUNT(*) AS n FROM forecast_runs")
    assert rows[0]["n"] == 0


async def test_shadow_snapshot_writes_local_rows(api):
    await _create_system(api)
    await _train_local(api)
    written = await api.app.state.pipeline.snapshot_local()
    assert written > 0
    rows = await api.app.state.db.query(
        "SELECT COUNT(*) AS n FROM forecast_runs WHERE source='local'"
    )
    assert rows[0]["n"] == written
