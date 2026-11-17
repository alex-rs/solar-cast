import datetime as dt

import pytest

from custom_components.solarcast.coordinator import SolarCastCoordinator
from tests_ha.fixtures import TZ, make_entry

pytestmark = pytest.mark.asyncio


class FakeApi:
    async def forecast(self, system_id: str, resolution: int = 60, days: int = 7, layer: str = "final"):
        now = dt.datetime.now(dt.UTC)
        first_day = now.astimezone(TZ).date()
        return {
            "timezone": "Europe/Belgrade",
            "layer": layer,
            "generated_at": now.isoformat(),
            "periods": [],
            "daily": [
                {
                    "date": (first_day + dt.timedelta(days=offset)).isoformat(),
                    "wh": 1000.0 * (offset + 1),
                }
                for offset in range(days)
            ],
        }

    async def benchmark(self, system_id: str, days: int = 14) -> dict:
        return {}


async def test_update_parses_daily_totals(hass):
    coordinator = SolarCastCoordinator(hass, make_entry(), FakeApi(), "home")
    data = await coordinator._async_update_data()

    first_day = dt.datetime.now(TZ).date()
    assert [(entry.date, entry.wh) for entry in data.daily] == [
        (first_day + dt.timedelta(days=offset), 1000.0 * (offset + 1))
        for offset in range(7)
    ]
