from types import SimpleNamespace

import pytest

from custom_components.solarcast import _async_register_services
from custom_components.solarcast.const import (
    DOMAIN,
    SERVICE_CLEAR_HORIZON,
    SERVICE_UPLOAD_HORIZON,
)
from tests_ha.fixtures import make_entry

pytestmark = pytest.mark.asyncio


class FakeApi:
    def __init__(self) -> None:
        self.uploaded: list[tuple[str, str]] = []
        self.cleared: list[str] = []

    async def put_horizon(self, system_id: str, csv_text: str) -> None:
        self.uploaded.append((system_id, csv_text))

    async def delete_horizon(self, system_id: str) -> None:
        self.cleared.append(system_id)


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
