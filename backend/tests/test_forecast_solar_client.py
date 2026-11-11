import json
from pathlib import Path

import httpx
import pytest
import respx

from app.core.forecast_solar_client import (
    ForecastSolarClient,
    RateLimited,
    fetch_system_shadow,
    parse_estimate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "forecast_solar_estimate.json"


def test_parse_estimate_converts_period_ends_to_utc_starts():
    payload = json.loads(FIXTURE.read_text())
    periods = parse_estimate(payload)
    assert sum(periods.values()) == pytest.approx(
        sum(payload["result"]["watt_hours_period"].values())
    )
    # 07:00 local (Europe/Belgrade, UTC+1) is a period end; its UTC start is 05:00.
    start = int(
        __import__("datetime").datetime(
            2026, 11, 11, 5, 0, tzinfo=__import__("datetime").timezone.utc
        ).timestamp()
    )
    assert periods[start] == pytest.approx(16.0)


@respx.mock
@pytest.mark.asyncio
async def test_429_disables_until_next_slot_without_retry_storm():
    clock = [1000.0]
    route = respx.get("https://api.forecast.solar/estimate/1/2/30/0/1").mock(
        return_value=httpx.Response(
            429,
            json={"message": {"ratelimit": {"period": 3600}}},
        )
    )
    client = ForecastSolarClient(now=lambda: clock[0])
    with pytest.raises(RateLimited):
        await client.estimate_plane(1, 2, 30, 0, 1)
    with pytest.raises(RateLimited):
        await client.estimate_plane(1, 2, 30, 0, 1)
    assert route.call_count == 1
    clock[0] += 3601
    respx.get("https://api.forecast.solar/estimate/1/2/30/0/1").mock(
        return_value=httpx.Response(200, json={"result": {}, "message": {}})
    )
    response = await client.estimate_plane(1, 2, 30, 0, 1)
    assert response.status_code == 200


@respx.mock
@pytest.mark.asyncio
async def test_three_calls_per_hour_budget():
    clock = [5000.0]
    respx.get("https://api.forecast.solar/estimate/1/2/30/0/1").mock(
        return_value=httpx.Response(200, json={"result": {}, "message": {}})
    )
    client = ForecastSolarClient(now=lambda: clock[0])
    for _ in range(3):
        await client.estimate_plane(1, 2, 30, 0, 1)
    with pytest.raises(RateLimited):
        await client.estimate_plane(1, 2, 30, 0, 1)
    clock[0] += 3601
    assert (await client.estimate_plane(1, 2, 30, 0, 1)).status_code == 200


@respx.mock
@pytest.mark.asyncio
async def test_fetch_shadow_sums_planes_and_caps_energy(tmp_path):
    import datetime as dt

    from app.core.database import Database

    payload = json.loads(FIXTURE.read_text())
    respx.get(url__startswith="https://api.forecast.solar/estimate/").mock(
        return_value=httpx.Response(200, json=payload)
    )
    db = Database(tmp_path / "shadow.db")
    await db.initialize()
    await db.upsert_system(
        {
            "system_id": "home",
            "name": "Home",
            "lat": 44.8125,
            "lon": 20.4612,
            "timezone": "Europe/Belgrade",
            "inverter_kw": 1.0,
        },
        [{"dec": 30, "az_fs": 0, "kwp": 5.0}],
    )
    system = {
        "system_id": "home",
        "lat": 44.8125,
        "lon": 20.4612,
        "inverter_kw": 1.0,
        "planes": [
            {"plane_id": "a", "dec": 30, "az_fs": 0, "kwp": 2.5},
            {"plane_id": "b", "dec": 30, "az_fs": 0, "kwp": 2.5},
        ],
    }
    client = ForecastSolarClient()
    fetched_at = int(dt.datetime(2026, 11, 11, 6, 0, tzinfo=dt.timezone.utc).timestamp())
    count = await fetch_system_shadow(db, system, client, inverter_kw=1.0, fetched_at=fetched_at)
    assert count > 0
    rows = await db.query(
        "SELECT * FROM forecast_runs WHERE system_id='home' AND source='forecast_solar'"
    )
    # Two planes were summed, then capped at 1 kW * 1 h.
    assert max(row["energy_wh"] for row in rows) <= 1000.0
    raw = await db.query("SELECT * FROM forecast_solar_raw WHERE system_id='home'")
    assert len(raw) == 2
