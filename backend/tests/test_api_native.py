import datetime as dt
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
    for entry in payload["daily"]:
        assert entry["wh"] == pytest.approx(by_date[pd.Timestamp(entry["date"]).date()])


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
