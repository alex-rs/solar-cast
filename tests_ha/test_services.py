import datetime as dt
from types import SimpleNamespace

import pytest

from custom_components.solarcast import _async_register_services
from custom_components.solarcast.const import (
    DOMAIN,
    SERVICE_CLEAR_HORIZON,
    SERVICE_DELETE_TELEMETRY,
    SERVICE_SET_TELEMETRY,
    SERVICE_UPLOAD_HORIZON,
)
from tests_ha.fixtures import make_entry

pytestmark = pytest.mark.asyncio


class FakeApi:
    def __init__(self) -> None:
        self.uploaded: list[tuple[str, str]] = []
        self.cleared: list[str] = []
        self.manual: list[tuple[str, dict]] = []
        self.deleted: list[tuple[str, str, str]] = []

    async def put_horizon(self, system_id: str, csv_text: str) -> None:
        self.uploaded.append((system_id, csv_text))

    async def delete_horizon(self, system_id: str) -> None:
        self.cleared.append(system_id)

    async def set_telemetry(self, system_id: str, item: dict) -> dict:
        self.manual.append((system_id, item))
        return {"accepted": 1, "rejected": []}

    async def delete_telemetry(self, system_id: str, start: str, end: str) -> dict:
        self.deleted.append((system_id, start, end))
        return {"deleted": 2, "start": start, "end": end}


async def test_horizon_services_forward_to_api(hass):
    api = FakeApi()
    entry = make_entry()
    entry.add_to_hass(hass)
    entry.runtime_data = SimpleNamespace(
        api=api, coordinator=SimpleNamespace(system_id="home"), telemetry=None
    )
    _async_register_services(hass)
    assert hass.services.has_service(DOMAIN, SERVICE_UPLOAD_HORIZON)
    assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_HORIZON)

    csv_text = "azimuth_fs,elevation\n-90,10\n0,5\n90,10\n"
    await hass.services.async_call(
        DOMAIN,
        SERVICE_UPLOAD_HORIZON,
        {"config_entry_id": entry.entry_id, "horizon": csv_text},
        blocking=True,
    )
    assert api.uploaded == [("home", csv_text)]

    await hass.services.async_call(
        DOMAIN,
        SERVICE_CLEAR_HORIZON,
        {"config_entry_id": entry.entry_id},
        blocking=True,
    )
    assert api.cleared == ["home"]


async def test_telemetry_correction_services(hass):
    api = FakeApi()
    entry = make_entry()
    entry.add_to_hass(hass)
    entry.runtime_data = SimpleNamespace(
        api=api, coordinator=SimpleNamespace(system_id="home"), telemetry=None
    )
    _async_register_services(hass)
    assert hass.services.has_service(DOMAIN, SERVICE_SET_TELEMETRY)
    assert hass.services.has_service(DOMAIN, SERVICE_DELETE_TELEMETRY)

    start = dt.datetime(2026, 11, 10, 10, 0, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(hours=1)
    await hass.services.async_call(
        DOMAIN,
        SERVICE_SET_TELEMETRY,
        {
            "config_entry_id": entry.entry_id,
            "period_start": start,
            "energy_wh": 1234.5,
            "curtailed": True,
        },
        blocking=True,
    )
    assert api.manual == [
        (
            "home",
            {
                "period_start": start.isoformat(),
                "period_min": 15,
                "energy_wh": 1234.5,
                "curtailed": True,
            },
        )
    ]

    await hass.services.async_call(
        DOMAIN,
        SERVICE_DELETE_TELEMETRY,
        {"config_entry_id": entry.entry_id, "start": start, "end": end},
        blocking=True,
    )
    assert api.deleted == [("home", start.isoformat(), end.isoformat())]


async def test_service_inputs_are_localized_and_coerced(hass):
    api = FakeApi()
    entry = make_entry()
    entry.add_to_hass(hass)
    entry.runtime_data = SimpleNamespace(
        api=api, coordinator=SimpleNamespace(system_id="home"), telemetry=None
    )
    _async_register_services(hass)

    # Offset-less strings (UI/automation form) and the select selector's string
    # values must reach the backend as an aware timestamp and an int.
    await hass.services.async_call(
        DOMAIN,
        SERVICE_SET_TELEMETRY,
        {
            "config_entry_id": entry.entry_id,
            "period_start": "2026-11-10 10:00:00",
            "energy_wh": 10,
            "period_min": "15",
        },
        blocking=True,
    )
    item = api.manual[-1][1]
    parsed = dt.datetime.fromisoformat(item["period_start"])
    assert parsed.utcoffset() is not None
    assert parsed.replace(tzinfo=None) == dt.datetime(2026, 11, 10, 10, 0)
    assert item["period_min"] == 15

    await hass.services.async_call(
        DOMAIN,
        SERVICE_DELETE_TELEMETRY,
        {
            "config_entry_id": entry.entry_id,
            "start": "2026-11-09 00:00:00",
            "end": "2026-11-10 00:00:00",
        },
        blocking=True,
    )
    _, start_iso, end_iso = api.deleted[-1]
    assert dt.datetime.fromisoformat(start_iso).utcoffset() is not None
    assert dt.datetime.fromisoformat(end_iso).utcoffset() is not None
