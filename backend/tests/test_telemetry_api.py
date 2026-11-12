import datetime as dt

import pytest

from tests.conftest import system_body


def _period_start(minutes_ago: int) -> dt.datetime:
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    return (now - dt.timedelta(minutes=minutes_ago)).replace(minute=0)


def _item(start: dt.datetime, energy_wh: float = 500.0) -> dict:
    return {
        "period_start": start.isoformat(),
        "period_min": 15,
        "energy_wh": energy_wh,
    }


async def _seed(api, start: dt.datetime, origin: str, energy_wh: float = 500.0) -> dict:
    response = await api.post(
        "/api/v1/systems/home/telemetry",
        json={"items": [_item(start, energy_wh)], "origin": origin},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_manual_origin_survives_push_and_backfill(api):
    await api.put("/api/v1/systems/home", json=system_body())
    start = _period_start(60)

    assert (await _seed(api, start, "manual", 900.0))["accepted"] == 1
    # The pusher must not overwrite a user correction for the same bucket.
    assert (await _seed(api, start, "push", 100.0))["accepted"] == 0
    assert (await _seed(api, start, "backfill", 100.0))["accepted"] == 0

    # Deleting the bucket proves the manual row was the one stored.
    deleted = await api.delete(
        "/api/v1/systems/home/telemetry",
        params={"start": start.isoformat(), "end": (start + dt.timedelta(minutes=15)).isoformat()},
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] == 1


@pytest.mark.asyncio
async def test_push_outranks_backfill_and_equal_rank_replaces(api):
    await api.put("/api/v1/systems/home", json=system_body())
    start = _period_start(90)

    assert (await _seed(api, start, "push", 200.0))["accepted"] == 1
    assert (await _seed(api, start, "backfill", 800.0))["accepted"] == 0
    # Equal rank replaces, keeping the push idempotent.
    assert (await _seed(api, start, "push", 250.0))["accepted"] == 1


@pytest.mark.asyncio
async def test_delete_only_removes_requested_range(api):
    await api.put("/api/v1/systems/home", json=system_body())
    first = _period_start(180)
    second = first + dt.timedelta(minutes=15)
    third = first + dt.timedelta(minutes=30)
    for start in (first, second, third):
        await _seed(api, start, "push")

    response = await api.delete(
        "/api/v1/systems/home/telemetry",
        params={"start": second.isoformat(), "end": third.isoformat()},
    )
    assert response.status_code == 200, response.text
    assert response.json()["deleted"] == 1

    # The remaining neighbours are still deletable, the removed bucket is not.
    remaining = await api.delete(
        "/api/v1/systems/home/telemetry",
        params={"start": first.isoformat(), "end": (third + dt.timedelta(minutes=15)).isoformat()},
    )
    assert remaining.json()["deleted"] == 2


@pytest.mark.asyncio
async def test_delete_validation(api):
    await api.put("/api/v1/systems/home", json=system_body())
    start = _period_start(60)
    end = start + dt.timedelta(minutes=15)

    assert (
        await api.delete(
            "/api/v1/systems/home/telemetry",
            params={"start": start.isoformat(), "end": start.isoformat()},
        )
    ).status_code == 422
    assert (
        await api.delete(
            "/api/v1/systems/home/telemetry",
            params={"start": start.replace(tzinfo=None).isoformat(), "end": end.isoformat()},
        )
    ).status_code == 422
    assert (
        await api.delete("/api/v1/systems/missing/telemetry", params={"start": start.isoformat(), "end": end.isoformat()})
    ).status_code == 404
    assert (
        await api.delete("/api/v1/systems/home/telemetry", params={"start": start.isoformat()})
    ).status_code == 422
    assert (
        await api.post(
            "/api/v1/systems/home/telemetry",
            json={"items": [_item(start)], "origin": "sensor"},
        )
    ).status_code == 422


@pytest.mark.asyncio
async def test_delete_respects_sub_second_start(api):
    await api.put("/api/v1/systems/home", json=system_body())
    start = _period_start(120)
    await _seed(api, start, "push")
    # The documented range starts after 00:00:30, so the :00:00 row survives.
    response = await api.delete(
        "/api/v1/systems/home/telemetry",
        params={
            "start": (start + dt.timedelta(seconds=30)).isoformat(),
            "end": (start + dt.timedelta(minutes=15)).isoformat(),
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["deleted"] == 0
