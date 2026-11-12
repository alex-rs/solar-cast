"""Binary sensors exposing whether the learning loop has fresh data."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, TELEMETRY_STALE_MINUTES
from .coordinator import SolarCastCoordinator


async def async_setup_entry(
    hass, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: SolarCastCoordinator = entry.runtime_data.coordinator
    async_add_entities([SolarCastTelemetryFresh(entry, coordinator)])


class SolarCastTelemetryFresh(CoordinatorEntity[SolarCastCoordinator], BinarySensorEntity):
    """On while the newest telemetry period is recent enough for the nowcast."""

    _attr_has_entity_name = True
    _attr_name = "Telemetry fresh"
    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, entry: ConfigEntry, coordinator: SolarCastCoordinator) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_telemetry_fresh"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="SolarCast",
            model="PV forecast",
        )

    @property
    def is_on(self) -> bool | None:
        data = self.coordinator.data
        if data is None or data.telemetry_age_minutes is None:
            return None
        return data.telemetry_age_minutes <= TELEMETRY_STALE_MINUTES

    @property
    def extra_state_attributes(self):
        data = self.coordinator.data
        if data is None:
            return None
        learning = data.learning or {}
        return {
            "system_id": self.coordinator.system_id,
            "telemetry_age_minutes": data.telemetry_age_minutes,
            "nowcast": learning.get("nowcast"),
            "nowcast_buckets": learning.get("nowcast_buckets"),
            "calibration_c": learning.get("calibration_c"),
            "provenance": data.provenance,
            "degraded": data.degraded,
            "local_horizon_minutes": data.local_horizon_minutes,
        }
