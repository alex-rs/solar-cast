"""Diagnostics support."""

from __future__ import annotations

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .api import LocalSolarCastError
from .const import CONF_TOKEN

TO_REDACT = {CONF_TOKEN}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict:
    runtime = getattr(entry, "runtime_data", None)
    health = None
    error = None
    if runtime is not None:
        try:
            health = await runtime.api.health()
        except LocalSolarCastError as exc:
            error = str(exc)
    data = runtime.coordinator.data if runtime is not None else None
    return {
        "entry": async_redact_data(dict(entry.data), TO_REDACT),
        "options": async_redact_data(dict(entry.options), TO_REDACT),
        "health": health,
        "health_error": error,
        "forecast": (
            {
                "generated_at": data.generated_at.isoformat(),
                "timezone": data.timezone,
                "layer": data.layer,
                "layers_active": data.layers_active,
                "hourly_periods": len(data.hourly),
                "intraday_periods": len(data.intraday),
                "accuracy_nmae_14d": data.accuracy_nmae_14d,
            }
            if data is not None
            else None
        ),
    }
