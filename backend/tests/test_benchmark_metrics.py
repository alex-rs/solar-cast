import datetime as dt

import numpy as np
import pytest

from app.core.benchmark import assign_bucket, benchmark, metrics


def test_metrics_hand_computed():
    result = metrics(np.array([10.0, -10.0]), np.array([100.0, 100.0]), kwp_total=5.0)
    assert result["nMAE_pct"] == pytest.approx(10.0)
    assert result["nRMSE_pct"] == pytest.approx(10.0)
    assert result["nMAE_kwp_pct"] == pytest.approx(100.0 * 10.0 / 5000.0)
    assert result["MBE_wh"] == pytest.approx(0.0)
    assert result["n_samples"] == 2


def test_metrics_null_on_empty():
    result = metrics(np.array([]), np.array([]), kwp_total=5.0)
    assert result["n_samples"] == 0
    assert result["nMAE_pct"] is None


def test_lead_buckets():
    assert assign_bucket(0) == "intraday"
    assert assign_bucket(11.99) == "intraday"
    assert assign_bucket(12) == "day_ahead"
    assert assign_bucket(35.99) == "day_ahead"
    assert assign_bucket(36) == "d2_3"
    assert assign_bucket(83.99) == "d2_3"
    assert assign_bucket(84) == "d4_7"
    assert assign_bucket(167.99) == "d4_7"
    assert assign_bucket(168) is None


@pytest.mark.asyncio
async def test_benchmark_intersection_and_metrics(api):
    await api.put("/api/v1/systems/home", json={"name": "Home", "lat": 44.8125, "lon": 20.4612, "timezone": "Europe/Belgrade", "planes": [{"dec": 30, "az": 0, "kwp": 5.0}]})
    app = api.app
    db = app.state.db
    start = dt.datetime(2026, 6, 21, 10, 0, tzinfo=dt.timezone.utc)
    fetched = int(start.timestamp()) - 20 * 3600
    telemetry = []
    runs = {"physics_base": [], "hybrid": [], "forecast_solar": [], "local": []}
    for hour in range(6):
        hour_start = start + dt.timedelta(hours=hour)
        for quarter in range(4):
            ts = hour_start + dt.timedelta(minutes=15 * quarter)
            telemetry.append(
                {
                    "period_start": int(ts.timestamp()),
                    "period_min": 15,
                    "energy_wh": 100.0,
                    "mean_watts": 400.0,
                    "curtailed": 0,
                    "origin": "push",
                }
            )
            for source, quarter_wh in (
                ("physics_base", 35.0),
                ("hybrid", 100.0),
                ("forecast_solar", 75.0),
                ("local", 90.0),
            ):
                runs[source].append(
                    {
                        "system_id": "home",
                        "fetched_at": fetched,
                        "period_start": int(ts.timestamp()),
                        "period_min": 15,
                        "lead_hours": 20.0,
                        "source": source,
                        "energy_wh": quarter_wh,
                        "mean_watts": quarter_wh * 4,
                    }
                )
    await db.upsert_telemetry("home", telemetry)
    for source, rows in runs.items():
        await db.insert_runs(rows)

    system = await db.get_system("home")
    result = await benchmark(db, system, days=14, now=dt.datetime(2026, 6, 22, tzinfo=dt.timezone.utc))
    bucket = result["buckets"]["day_ahead"]
    assert bucket["n_samples"] == 6
    assert bucket["sources"]["hybrid"]["nMAE_pct"] == pytest.approx(0.0)
    assert bucket["sources"]["physics_base"]["nMAE_pct"] == pytest.approx(65.0)
    assert bucket["sources"]["forecast_solar"]["nMAE_pct"] == pytest.approx(25.0)
    # The local shadow layer is scored like any other source.
    assert bucket["sources"]["local"]["nMAE_pct"] == pytest.approx(10.0)
    assert bucket["sources"]["hybrid"]["skill_vs_physics_base"] == pytest.approx(1.0)
    assert bucket["sources"]["physics_base"]["MBE_wh"] == pytest.approx(-260.0)
    assert result["buckets"]["d2_3"]["n_samples"] == 0
    assert result["buckets"]["d2_3"]["sources"] == {}


@pytest.mark.asyncio
async def test_benchmark_null_without_telemetry(api):
    await api.put("/api/v1/systems/home", json={"name": "Home", "lat": 44.8125, "lon": 20.4612, "timezone": "Europe/Belgrade", "planes": [{"dec": 30, "az": 0, "kwp": 5.0}]})
    db = api.app.state.db
    system = await db.get_system("home")
    result = await benchmark(db, system, days=14, now=dt.datetime(2026, 6, 22, tzinfo=dt.timezone.utc))
    assert result["buckets"]["day_ahead"]["n_samples"] == 0
    assert result["buckets"]["day_ahead"]["sources"] == {}


@pytest.mark.asyncio
async def test_telemetry_validation_and_override(api):
    await api.put("/api/v1/systems/home", json={"name": "Home", "lat": 44.8125, "lon": 20.4612, "timezone": "Europe/Belgrade", "planes": [{"dec": 30, "az": 0, "kwp": 5.0}]})
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    aligned = (now - dt.timedelta(minutes=15)).replace(minute=(now.minute // 15) * 15)
    unaligned = aligned + dt.timedelta(minutes=7)
    payload = {
        "origin": "backfill",
        "items": [
            {"period_start": aligned.isoformat(), "period_min": 15, "energy_wh": 100.0},
            {"period_start": unaligned.isoformat(), "period_min": 15, "energy_wh": 50.0},
            {"period_start": (now + dt.timedelta(hours=1)).isoformat(), "period_min": 15, "energy_wh": 50.0},
            {"period_start": (aligned - dt.timedelta(minutes=15)).isoformat(), "period_min": 15, "energy_wh": -1.0},
        ],
    }
    response = await api.post("/api/v1/systems/home/telemetry", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["accepted"] == 1
    assert [item["index"] for item in body["rejected"]] == [1, 2, 3]

    override = {
        "origin": "push",
        "items": [
            {"period_start": aligned.isoformat(), "period_min": 15, "energy_wh": 200.0, "curtailed": True}
        ],
    }
    assert (await api.post("/api/v1/systems/home/telemetry", json=override)).json()["accepted"] == 1
    rows = await api.app.state.db.telemetry("home")
    assert len(rows) == 1
    assert rows[0]["energy_wh"] == pytest.approx(200.0)
    assert rows[0]["origin"] == "push"
    assert rows[0]["mean_watts"] == pytest.approx(800.0)

    # Backfill must not override an existing push row.
    downdate = {
        "origin": "backfill",
        "items": [
            {"period_start": aligned.isoformat(), "period_min": 15, "energy_wh": 50.0}
        ],
    }
    await api.post("/api/v1/systems/home/telemetry", json=downdate)
    rows = await api.app.state.db.telemetry("home")
    assert rows[0]["energy_wh"] == pytest.approx(200.0)
    assert rows[0]["origin"] == "push"
