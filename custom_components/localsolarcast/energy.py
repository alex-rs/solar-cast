"""Home Assistant Energy dashboard solar forecast provider."""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant

LOG = logging.getLogger(__name__)


async def async_get_solar_forecast(
    hass: HomeAssistant, config_entry_id: str
) -> dict[str, dict[str, float | int]] | None:
    entry = hass.config_entries.async_get_entry(config_entry_id)
    if entry is None:
        return None
    runtime = getattr(entry, "runtime_data", None)
    if runtime is None or runtime.coordinator.data is None:
        return None
    data = runtime.coordinator.data
    return {
        "wh_hours": {
            period.start.isoformat(): period.wh for period in data.hourly
        }
    }
