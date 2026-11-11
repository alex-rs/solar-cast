import datetime as dt

import pytest

from custom_components.solarcast.energy import async_get_solar_forecast
from tests_ha.fixtures import make_data, make_entry, runtime_with


@pytest.mark.asyncio
async def test_energy_dashboard_shape(hass):
    entry = make_entry()
    entry.add_to_hass(hass)
    data = make_data()
    entry.runtime_data = runtime_with(data)
    result = await async_get_solar_forecast(hass, entry.entry_id)
    assert set(result) == {"wh_hours"}
    wh_hours = result["wh_hours"]
    assert len(wh_hours) == len(data.hourly)
    for period in data.hourly:
        assert wh_hours[period.start.isoformat()] == pytest.approx(period.wh)
        assert dt.datetime.fromisoformat(period.start.isoformat()).utcoffset() is not None
    assert await async_get_solar_forecast(hass, "missing") is None
