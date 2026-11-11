import datetime as dt
import re
from zoneinfo import ZoneInfo

import pytest

KEY_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


@pytest.mark.asyncio
async def test_compat_allows_one_to_four_planes(api):
    one = await api.get("/estimate/44.8125/20.4612/30/0/2.5")
    assert one.status_code == 200, one.text
    four = await api.get("/estimate/44.8125/20.4612/30/0/2.5/30/90/2.5/30/-90/2.5/30/180/1")
    assert four.status_code == 200, four.text
    five = await api.get("/estimate/44.8125/20.4612/30/0/1/30/0/1/30/0/1/30/0/1/30/0/1")
    assert five.status_code == 422
    assert (await api.get("/estimate/44.8125/20.4612/30/0")).status_code == 422
    assert (await api.get("/estimate/44.8125/20.4612/abc/0/1")).status_code == 422
    assert (await api.get("/estimate/44.8125/20.4612/30/0/-1")).status_code == 422


@pytest.mark.asyncio
async def test_compat_apikey_route_is_ignored(api):
    response = await api.get("/demo-key/estimate/44.8125/20.4612/30/0/2.5")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_compat_matches_golden_structure(api, fs_fixture):
    response = await api.get(
        "/estimate/44.8125/20.4612/30/0/5",
        params={"inverter": 5.0, "damping": "0.1,0.1"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = payload["result"]
    assert set(result) == {"watts", "watt_hours_period", "watt_hours", "watt_hours_day"}
    for key in result["watts"]:
        assert KEY_RE.match(key)
    assert list(result["watts"]) == list(result["watt_hours_period"]) == list(result["watt_hours"])

    fixture_keys = list(fs_fixture["result"]["watts"])
    fixture_days = sorted({key[:10] for key in fixture_keys})
    # The fixture is a snapshot; only its day structure is stable.
    result_days = sorted({key[:10] for key in result["watts"]})
    assert len(result_days) == len(fixture_days) == 2
    assert (
        dt.date.fromisoformat(result_days[1]) - dt.date.fromisoformat(result_days[0])
    ).days == 1
    for day in result_days:
        day_keys = [key for key in result["watts"] if key[:10] == day]
        assert day_keys[0] < day_keys[1]
    # Cumulative watt_hours matches period sums and the daily totals per local day.
    for day in result_days:
        keys = [key for key in result["watt_hours"] if key[:10] == day]
        total = sum(result["watt_hours_period"][key] for key in keys)
        assert result["watt_hours"][keys[-1]] == pytest.approx(total)
        assert result["watt_hours_day"][day] == pytest.approx(total)

    message = payload["message"]
    assert message["code"] == 0
    assert message["type"] == "success"
    assert set(message["info"]) == {
        "latitude",
        "longitude",
        "distance",
        "place",
        "timezone",
        "time",
        "time_utc",
        "solarcast",
    }
    assert set(message["ratelimit"]) == {"zone", "period", "limit", "remaining"}
    assert message["info"]["timezone"] == "Europe/Belgrade"


@pytest.mark.asyncio
async def test_compat_anchors_bracket_hourly_keys(api):
    response = await api.get("/estimate/44.8125/20.4612/30/0/5")
    payload = response.json()
    for day in {key[:10] for key in payload["result"]["watts"]}:
        keys = [key for key in payload["result"]["watts"] if key[:10] == day]
        first = dt.datetime.strptime(keys[0], "%Y-%m-%d %H:%M:%S")
        last = dt.datetime.strptime(keys[-1], "%Y-%m-%d %H:%M:%S")
        hourly = [
            dt.datetime.strptime(key, "%Y-%m-%d %H:%M:%S")
            for key in keys
            if key.endswith(":00:00")
        ]
        # Sunrise/sunset anchors have non-midnight minutes and bracket the hourly keys.
        assert first <= min(hourly)
        assert last >= max(hourly)
        assert first < min(hourly)
        assert last >= max(hourly)


@pytest.mark.asyncio
async def test_compat_reports_matched_system(api):
    from tests.conftest import system_body

    await api.put("/api/v1/systems/home", json=system_body())
    matched = (
        await api.get("/estimate/44.8125/20.4612/30/0/5", params={"inverter": 5.0})
    ).json()
    info = matched["message"]["info"]["solarcast"]
    assert info["matched_system"] == "home"
    assert set(info["layers_active"]) == {"calibration", "shading", "mos", "nowcast"}
    assert info["layers_active"]["calibration"] is False

    unmatched = (
        await api.get("/estimate/44.8125/20.4612/30/0/4", params={"inverter": 5.0})
    ).json()
    info = unmatched["message"]["info"]["solarcast"]
    assert info["matched_system"] is None
    assert info["layers_active"] == {}
