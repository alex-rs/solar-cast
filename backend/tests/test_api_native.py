import datetime as dt
import time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tests.conftest import system_body


@pytest.mark.asyncio
async def test_put_get_list_delete(api):
    response = await api.put("/api/v1/systems/home", json=system_body())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["system_id"] == "home"
    assert payload["timezone"] == "Europe/Belgrade"
    assert payload["planes"][0]["plane_id"]
    assert len(payload["planes"]) == 1

    listed = await api.get("/api/v1/systems")
    assert [s["system_id"] for s in listed.json()] == ["home"]
    assert (await api.get("/api/v1/systems/home")).status_code == 200
    assert (await api.delete("/api/v1/systems/home")).status_code == 204
    assert (await api.get("/api/v1/systems/home")).status_code == 404


@pytest.mark.asyncio
async def test_validation(api):
    bad = system_body(planes=[{"dec": 30, "az": 0, "kwp": 1.0}] * 5)
    assert (await api.put("/api/v1/systems/x", json=bad)).status_code == 422
    bad_az = system_body(planes=[{"dec": 30, "az": 200, "kwp": 1.0}])
    assert (await api.put("/api/v1/systems/x", json=bad_az)).status_code == 422
    assert (await api.put("/api/v1/systems/_bad", json=system_body())).status_code == 422


@pytest.mark.asyncio
async def test_native_forecast_shape_and_periods(api):
    await api.put("/api/v1/systems/home", json=system_body())
    response = await api.get(
        "/api/v1/systems/home/forecast", params={"resolution": 60, "days": 2, "layer": "final"}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["resolution_minutes"] == 60
    assert payload["layer"] == "final"
    assert payload["kwp_total"] == pytest.approx(5.0)
    assert payload["ac_limit_w"] == pytest.approx(5000.0)
    assert payload["periods"]
    for period in payload["periods"]:
        start = dt.datetime.fromisoformat(period["start"])
        end = dt.datetime.fromisoformat(period["end"])
        assert start.utcoffset() is not None
        assert start.minute == 0
        assert (end - start) == dt.timedelta(hours=1)
        assert period["wh"] >= 0
        assert period["mean_watts"] == pytest.approx(period["wh"])
    # Daily sums follow local dates and equal the selected period sums.
    local = pd.Series(
        [p["wh"] for p in payload["periods"]],
        index=pd.DatetimeIndex([p["start"] for p in payload["periods"]]),
    )
    by_date = local.groupby(local.index.date).sum()
    today_local = dt.datetime.now(ZoneInfo("Europe/Belgrade")).date()
    first_date = pd.Timestamp(payload["periods"][0]["start"]).date()
    for entry in payload["daily"]:
        day = pd.Timestamp(entry["date"]).date()
        # Daily totals are whole local days; the returned periods start at "now"
        # and stop at the requested horizon, so they never exceed the day total.
        assert entry["wh"] >= by_date.get(day, 0.0) - 1e-6
        if day == today_local:
            assert entry["wh"] == pytest.approx(payload["today"]["wh"])
        elif day == first_date:
            assert entry["wh"] == pytest.approx(by_date[day])

    today = payload["today"]
    assert today["date"] == today_local.isoformat()
    assert today["elapsed_wh"] + today["remaining_wh"] == pytest.approx(today["wh"])
    assert today["remaining_wh"] <= today["wh"] + 1e-6


@pytest.mark.asyncio
async def test_multi_inverter_system_round_trip(api):
    body = system_body(
        planes=[
            {"dec": 30, "az": -90, "kwp": 2.5, "inverter": "east"},
            {"dec": 30, "az": 90, "kwp": 2.5, "inverter": "west"},
        ],
        inverters=[
            {"id": "east", "ac_kw": 2.0},
            {"id": "west", "ac_kw": 2.0},
        ],
    )
    body.pop("inverter_kw")
    response = await api.put("/api/v1/systems/home", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["inverters"] == [
        {"id": "east", "ac_kw": 2.0},
        {"id": "west", "ac_kw": 2.0},
    ]
    assert [plane["inverter"] for plane in payload["planes"]] == ["east", "west"]
    forecast = await api.get("/api/v1/systems/home/forecast?days=1")
    assert forecast.status_code == 200, forecast.text
    assert forecast.json()["ac_limit_w"] == 4000.0

    conflict = system_body(inverters=[{"id": "main", "ac_kw": 5.0}])
    assert (await api.put("/api/v1/systems/home", json=conflict)).status_code == 422


@pytest.mark.asyncio
async def test_forecast_uses_stored_weights(api, weather_stub):
    await api.put("/api/v1/systems/home", json=system_body())
    db = api.app.state.db
    await db.put_weights(
        44.8125,
        20.4612,
        [
            {"model": "icon_seamless", "weight": 0.6, "samples": 250},
            {"model": "ecmwf_ifs025", "weight": 0.3, "samples": 250},
            {"model": "gfs_seamless", "weight": 0.1, "samples": 250},
        ],
        int(time.time()),
    )
    response = await api.get("/api/v1/systems/home/forecast")
    assert response.status_code == 200, response.text
    assert weather_stub.last_weights == [0.6, 0.3, 0.1]


@pytest.mark.asyncio
async def test_forecast_unknown_system(api):
    response = await api.get("/api/v1/systems/nope/forecast")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_geometry_change_resets_learning(api):
    await api.put("/api/v1/systems/home", json=system_body())
    app = api.app
    db = app.state.db
    await db.put_calibration("home", 0.9, 5, 100, 1)
    await db.replace_shading("home", [{"az_bucket": 90, "el_bucket": 10, "k_beam": 0.8, "sample_count": 4, "day_count": 2, "updated_at": 1}])
    await db.register_model(
        {
            "system_id": "home",
            "model_type": "mos_lgbm",
            "trained_at": 1,
            "artifact_path": "/tmp/none.joblib",
            "train_rows": 10,
            "holdout_rows": 2,
            "holdout_nmae_physics": 1.0,
            "holdout_nmae_hybrid": 0.9,
            "active": True,
        }
    )
    await api.put(
        "/api/v1/systems/home",
        json=system_body(planes=[{"dec": 35, "az": 10, "kwp": 4.0}]),
    )
    assert await db.get_calibration("home") is None
    assert await db.get_shading("home") == []
    assert await db.active_model("home") is None


@pytest.mark.asyncio
async def test_auth_required_when_token_set(api):
    api.app.state.settings.api_token = "secret"
    assert (await api.get("/api/v1/systems")).status_code == 401
    ok = await api.get("/api/v1/systems", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    api.app.state.settings.api_token = None


@pytest.mark.asyncio
async def test_forecast_learning_block(api):
    await api.put("/api/v1/systems/home", json=system_body())
    response = await api.get("/api/v1/systems/home/forecast", params={"days": 1})
    assert response.status_code == 200, response.text
    learning = response.json()["learning"]
    assert learning["calibration_c"] == pytest.approx(1.0)
    assert learning["calibration_days"] == 0
    assert learning["mos"] is False
    assert learning["shading"] is False
    assert learning["nowcast"] is False
    assert learning["nowcast_buckets"] == 0
    assert learning["telemetry_last_seen"] is None
    assert learning["telemetry_age_minutes"] is None

    now = dt.datetime.now(dt.timezone.utc)
    start = (now - dt.timedelta(minutes=15)).replace(minute=0, second=0, microsecond=0)
    start = start.replace(minute=start.minute - start.minute % 15)
    posted = await api.post(
        "/api/v1/systems/home/telemetry",
        json={
            "items": [
                {
                    "period_start": start.isoformat(),
                    "period_min": 15,
                    "energy_wh": 120.0,
                }
            ]
        },
    )
    assert posted.status_code == 200, posted.text
    assert posted.json()["accepted"] == 1

    refreshed = (await api.get("/api/v1/systems/home/forecast", params={"days": 1})).json()
    learning = refreshed["learning"]
    assert learning["telemetry_last_seen"] is not None
    assert learning["telemetry_age_minutes"] is not None
    assert 0 <= learning["telemetry_age_minutes"] <= 60
    assert learning["nowcast_buckets"] in (0, 1)


@pytest.mark.asyncio
async def test_nowcast_lookback_covers_sixty_minute_buckets(api, monkeypatch):
    await api.put("/api/v1/systems/home", json=system_body())
    now = dt.datetime.now(dt.timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=1)
    posted = await api.post(
        "/api/v1/systems/home/telemetry",
        json={
            "items": [
                {
                    "period_start": start.isoformat(),
                    "period_min": 60,
                    "energy_wh": 2400.0,
                }
            ]
        },
    )
    assert posted.status_code == 200, posted.text

    db = api.app.state.db
    seen: dict = {}
    real = db.telemetry

    async def spy(system_id, start_ts=None, end_ts=None):
        seen["span"] = end_ts - start_ts
        seen["rows"] = await real(system_id, start_ts, end_ts)
        return seen["rows"]

    monkeypatch.setattr(db, "telemetry", spy)
    response = await api.get("/api/v1/systems/home/forecast", params={"days": 1})
    assert response.status_code == 200, response.text
    # A 60-min bucket ended within 45 min starts up to 105 min ago; the lookback
    # must cover that, otherwise 60-min telemetry can never feed the nowcast.
    assert seen["span"] >= (45 + 60) * 60
    assert seen["rows"] is not None


@pytest.mark.asyncio
async def test_telemetry_age_never_negative(api):
    await api.put("/api/v1/systems/home", json=system_body())
    now = dt.datetime.now(dt.timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0)
    # A 60-minute bucket starting this hour has not ended yet.
    posted = await api.post(
        "/api/v1/systems/home/telemetry",
        json={
            "items": [
                {
                    "period_start": start.isoformat(),
                    "period_min": 60,
                    "energy_wh": 100.0,
                }
            ]
        },
    )
    assert posted.status_code == 200, posted.text
    learning = (await api.get("/api/v1/systems/home/forecast", params={"days": 1})).json()["learning"]
    assert learning["telemetry_age_minutes"] == 0.0


@pytest.mark.asyncio
async def test_identical_planes_on_different_inverters(api):
    body = system_body(
        inverter_kw=None,
        inverters=[{"id": "deye_a", "ac_kw": 1.6}, {"id": "deye_b", "ac_kw": 1.6}],
        planes=[
            {"dec": 17, "az": 62, "kwp": 0.87, "inverter": "deye_a"},
            {"dec": 17, "az": 62, "kwp": 0.87, "inverter": "deye_b"},
        ],
    )
    response = await api.put("/api/v1/systems/home", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    plane_ids = [plane["plane_id"] for plane in payload["planes"]]
    assert len(plane_ids) == len(set(plane_ids)) == 2
    assert [plane["inverter"] for plane in payload["planes"]] == ["deye_a", "deye_b"]

    forecast = await api.get("/api/v1/systems/home/forecast", params={"days": 1})
    assert forecast.status_code == 200, forecast.text
    assert forecast.json()["kwp_total"] == pytest.approx(1.74)
    assert forecast.json()["ac_limit_w"] == pytest.approx(3200.0)


@pytest.mark.asyncio
async def test_duplicate_planes_same_inverter_rejected(api):
    body = system_body(
        inverter_kw=None,
        inverters=[{"id": "deye_a", "ac_kw": 1.6}],
        planes=[
            {"dec": 17, "az": 62, "kwp": 0.87, "inverter": "deye_a"},
            {"dec": 17, "az": 62, "kwp": 0.87, "inverter": "deye_a"},
        ],
    )
    response = await api.put("/api/v1/systems/home", json=body)
    assert response.status_code == 422, response.text
    assert "duplicate plane" in response.text
    assert (await api.get("/api/v1/systems/home")).status_code == 404


@pytest.mark.asyncio
async def test_duplicate_planes_without_inverters_rejected(api):
    body = system_body(
        planes=[
            {"dec": 17, "az": 62, "kwp": 0.87},
            {"dec": 17, "az": 62, "kwp": 0.87},
        ]
    )
    response = await api.put("/api/v1/systems/home", json=body)
    assert response.status_code == 422, response.text
    assert "duplicate plane" in response.text


@pytest.mark.asyncio
async def test_database_rejects_duplicate_plane_ids(api):
    await api.put("/api/v1/systems/home", json=system_body())
    db = api.app.state.db
    system = {
        "system_id": "static",
        "name": "Static",
        "lat": 44.8125,
        "lon": 20.4612,
        "timezone": "UTC",
        "system_loss": 0.1,
    }
    planes = [
        {"dec": 17, "az_fs": 62, "kwp": 0.87},
        {"dec": 17, "az_fs": 62, "kwp": 0.87},
    ]
    with pytest.raises(ValueError, match="duplicate planes"):
        await db.upsert_system(system, planes)
    assert await db.get_system("static") is None
