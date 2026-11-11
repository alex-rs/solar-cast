"""SolarCast Home Assistant integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import SolarCastApi, SolarCastError
from .backfill import async_backfill, async_maybe_backfill
from .const import (
    CONF_AC_KW,
    CONF_ENERGY_ENTITY,
    CONF_INVERTER,
    CONF_INVERTER_ID,
    CONF_INVERTER_KW,
    CONF_INVERTERS,
    CONF_PLANES,
    CONF_SYSTEM_ID,
    CONF_TOKEN,
    CONF_URL,
    DOMAIN,
    PLATFORMS,
    SERVICE_BACKFILL,
    SERVICE_CLEAR_HORIZON,
    SERVICE_DELETE_TELEMETRY,
    SERVICE_GET_FORECAST,
    SERVICE_RETRAIN,
    SERVICE_SET_TELEMETRY,
    SERVICE_UPLOAD_HORIZON,
)
from .coordinator import SolarCastCoordinator
from .telemetry import TelemetryPusher

LOG = logging.getLogger(__name__)

CONFIG_ENTRY_ID = "config_entry_id"


@dataclass
class RuntimeData:
    api: SolarCastApi
    coordinator: SolarCastCoordinator
    telemetry: TelemetryPusher


def _system_payload(entry: ConfigEntry) -> dict:
    options = entry.options
    return {
        "name": options.get("name", entry.title),
        "lat": float(options["lat"]),
        "lon": float(options["lon"]),
        "system_loss": float(options.get("system_loss", 0.10)),
        "damping_morning": float(options.get("damping_morning", 0.0)),
        "damping_evening": float(options.get("damping_evening", 0.0)),
        "planes": [
            {
                "dec": float(p["dec"]),
                "az": float(p["az"]),
                "kwp": float(p["kwp"]),
                "inverter": p.get(CONF_INVERTER),
            }
            for p in options.get(CONF_PLANES, [])
        ],
        "inverters": [
            {
                "id": inverter[CONF_INVERTER_ID],
                "ac_kw": float(inverter[CONF_AC_KW]),
            }
            for inverter in options.get(CONF_INVERTERS, [])
        ],
    }


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    if entry.version < 2:
        options = dict(entry.options)
        kw = options.pop(CONF_INVERTER_KW, None)
        if kw and not options.get(CONF_INVERTERS):
            options[CONF_INVERTERS] = [
                {CONF_INVERTER_ID: "main", CONF_AC_KW: float(kw)}
            ]
            options[CONF_PLANES] = [
                {**plane, CONF_INVERTER: "main"}
                for plane in options.get(CONF_PLANES, [])
            ]
        hass.config_entries.async_update_entry(entry, options=options, version=2)
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    api = SolarCastApi(
        async_get_clientsession(hass),
        entry.data[CONF_URL],
        entry.data.get(CONF_TOKEN),
    )
    system_id = entry.data[CONF_SYSTEM_ID]
    try:
        await api.put_system(system_id, _system_payload(entry))
    except SolarCastError as exc:
        from homeassistant.exceptions import ConfigEntryNotReady

        raise ConfigEntryNotReady(f"cannot register system: {exc}") from exc

    coordinator = SolarCastCoordinator(hass, entry, api, system_id)
    await coordinator.async_config_entry_first_refresh()
    telemetry = TelemetryPusher(hass, entry, api, coordinator)
    entry.runtime_data = RuntimeData(api=api, coordinator=coordinator, telemetry=telemetry)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    await telemetry.async_start()
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    energy_entity = entry.options.get(CONF_ENERGY_ENTITY)
    if energy_entity:
        hass.async_create_task(
            async_maybe_backfill(hass, api, entry, energy_entity, 60)
        )

    _async_register_services(hass)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    runtime = getattr(entry, "runtime_data", None)
    if runtime is not None:
        await runtime.telemetry.async_stop()
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if not hass.config_entries.async_entries(DOMAIN):
        for service in (
            SERVICE_GET_FORECAST,
            SERVICE_BACKFILL,
            SERVICE_RETRAIN,
            SERVICE_UPLOAD_HORIZON,
            SERVICE_CLEAR_HORIZON,
            SERVICE_SET_TELEMETRY,
            SERVICE_DELETE_TELEMETRY,
        ):
            if hass.services.has_service(DOMAIN, service):
                hass.services.async_remove(DOMAIN, service)
    return unloaded


def _get_runtime(hass: HomeAssistant, entry_id: str) -> RuntimeData | None:
    entry = hass.config_entries.async_get_entry(entry_id)
    return getattr(entry, "runtime_data", None) if entry else None


def _async_register_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, SERVICE_GET_FORECAST):
        return

    async def handle_get_forecast(call: ServiceCall) -> dict:
        runtime = _get_runtime(hass, call.data[CONFIG_ENTRY_ID])
        if runtime is None:
            raise ValueError("unknown config entry")
        payload = await runtime.api.forecast(
            runtime.coordinator.system_id,
            int(call.data.get("resolution", 60)),
            7,
            call.data.get("layer", "final"),
        )
        return {
            "system_id": payload["system_id"],
            "timezone": payload["timezone"],
            "layer": payload["layer"],
            "periods": payload["periods"],
        }

    async def handle_backfill(call: ServiceCall) -> None:
        entry = hass.config_entries.async_get_entry(call.data[CONFIG_ENTRY_ID])
        runtime = _get_runtime(hass, call.data[CONFIG_ENTRY_ID])
        if entry is None or runtime is None:
            raise ValueError("unknown config entry")
        entity_id = call.data.get("entity_id") or entry.options.get(CONF_ENERGY_ENTITY)
        if not entity_id:
            raise ValueError("no PV energy entity configured")
        await async_backfill(
            hass, runtime.api, runtime.coordinator.system_id, entity_id, call.data.get("days", 60)
        )

    async def handle_retrain(call: ServiceCall) -> None:
        runtime = _get_runtime(hass, call.data[CONFIG_ENTRY_ID])
        if runtime is None:
            raise ValueError("unknown config entry")
        await runtime.api.retrain(runtime.coordinator.system_id)

    async def handle_upload_horizon(call: ServiceCall) -> None:
        runtime = _get_runtime(hass, call.data[CONFIG_ENTRY_ID])
        if runtime is None:
            raise ValueError("unknown config entry")
        await runtime.api.put_horizon(
            runtime.coordinator.system_id, call.data["horizon"]
        )

    async def handle_clear_horizon(call: ServiceCall) -> None:
        runtime = _get_runtime(hass, call.data[CONFIG_ENTRY_ID])
        if runtime is None:
            raise ValueError("unknown config entry")
        await runtime.api.delete_horizon(runtime.coordinator.system_id)

    async def handle_set_telemetry(call: ServiceCall) -> None:
        runtime = _get_runtime(hass, call.data[CONFIG_ENTRY_ID])
        if runtime is None:
            raise ValueError("unknown config entry")
        await runtime.api.set_telemetry(
            runtime.coordinator.system_id,
            {
                "period_start": call.data["period_start"].isoformat(),
                "period_min": int(call.data.get("period_min", 15)),
                "energy_wh": float(call.data["energy_wh"]),
                "curtailed": bool(call.data.get("curtailed", False)),
            },
        )

    async def handle_delete_telemetry(call: ServiceCall) -> None:
        runtime = _get_runtime(hass, call.data[CONFIG_ENTRY_ID])
        if runtime is None:
            raise ValueError("unknown config entry")
        await runtime.api.delete_telemetry(
            runtime.coordinator.system_id,
            call.data["start"].isoformat(),
            call.data["end"].isoformat(),
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_FORECAST,
        handle_get_forecast,
        schema=vol.Schema(
            {
                vol.Required(CONFIG_ENTRY_ID): cv.string,
                vol.Optional("resolution", default=60): vol.In([15, 60]),
                vol.Optional("layer", default="final"): vol.In(
                    ["final", "hybrid", "physics", "physics_base"]
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_BACKFILL,
        handle_backfill,
        schema=vol.Schema(
            {
                vol.Required(CONFIG_ENTRY_ID): cv.string,
                vol.Optional("days", default=60): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=60)
                ),
                vol.Optional("entity_id"): cv.entity_id,
            }
        ),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RETRAIN,
        handle_retrain,
        schema=vol.Schema({vol.Required(CONFIG_ENTRY_ID): cv.string}),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_UPLOAD_HORIZON,
        handle_upload_horizon,
        schema=vol.Schema(
            {
                vol.Required(CONFIG_ENTRY_ID): cv.string,
                vol.Required("horizon"): cv.string,
            }
        ),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_HORIZON,
        handle_clear_horizon,
        schema=vol.Schema({vol.Required(CONFIG_ENTRY_ID): cv.string}),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SET_TELEMETRY,
        handle_set_telemetry,
        schema=vol.Schema(
            {
                vol.Required(CONFIG_ENTRY_ID): cv.string,
                vol.Required("period_start"): cv.datetime,
                vol.Required("energy_wh"): vol.All(
                    vol.Coerce(float), vol.Range(min=0)
                ),
                vol.Optional("period_min", default=15): vol.In([15, 60]),
                vol.Optional("curtailed", default=False): cv.boolean,
            }
        ),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DELETE_TELEMETRY,
        handle_delete_telemetry,
        schema=vol.Schema(
            {
                vol.Required(CONFIG_ENTRY_ID): cv.string,
                vol.Required("start"): cv.datetime,
                vol.Required("end"): cv.datetime,
            }
        ),
    )
