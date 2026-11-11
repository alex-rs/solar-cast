"""Forecast sensors mirroring the Forecast.Solar integration names."""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import ForecastData, ForecastPeriod, LocalSolarCastCoordinator

LOG = logging.getLogger(__name__)


def _now(tz) -> dt.datetime:
    return dt.datetime.now(tz)


def _today(data: ForecastData, offset: int = 0) -> dt.date:
    return (_now(data.tz) + dt.timedelta(days=offset)).date()


def _periods_for_day(periods: list[ForecastPeriod], data: ForecastData, offset: int) -> list[ForecastPeriod]:
    target = _today(data, offset)
    return [p for p in periods if p.start.astimezone(data.tz).date() == target]


def _energy_kwh(periods: list[ForecastPeriod]) -> float:
    return round(sum(p.wh for p in periods) / 1000.0, 4)


def energy_today(data: ForecastData) -> float:
    return _energy_kwh(_periods_for_day(data.hourly, data, 0))


def energy_tomorrow(data: ForecastData) -> float:
    return _energy_kwh(_periods_for_day(data.hourly, data, 1))


def energy_today_remaining(data: ForecastData) -> float:
    now = _now(data.tz)
    total = 0.0
    for period in _periods_for_day(data.intraday, data, 0):
        if period.end <= now:
            continue
        if period.start <= now:
            fraction = (period.end - now).total_seconds() / (period.end - period.start).total_seconds()
            total += period.wh * max(0.0, fraction)
        else:
            total += period.wh
    return round(total / 1000.0, 4)


def _power_now(data: ForecastData) -> float:
    if not data.intraday:
        return 0.0
    now = _now(data.tz)
    stamps = np.array([p.start.timestamp() for p in data.intraday])
    values = np.array([p.mean_watts for p in data.intraday])
    result = float(np.interp(now.timestamp(), stamps, values))
    return round(max(result, 0.0), 1)


def _next_hour_period(data: ForecastData) -> ForecastPeriod | None:
    now = _now(data.tz)
    target = now + dt.timedelta(hours=1)
    for period in data.hourly:
        if period.start <= target < period.end:
            return period
    return None


def power_next_hour(data: ForecastData) -> float:
    period = _next_hour_period(data)
    return round(period.mean_watts, 1) if period else 0.0


def energy_next_hour(data: ForecastData) -> float:
    period = _next_hour_period(data)
    return round(period.wh / 1000.0, 4) if period else 0.0


def energy_current_hour(data: ForecastData) -> float:
    now = _now(data.tz)
    for period in data.hourly:
        if period.start <= now < period.end:
            return round(period.wh / 1000.0, 4)
    return 0.0


def _mean_power_window(data: ForecastData, hours: int) -> float:
    now = _now(data.tz)
    end = now + dt.timedelta(hours=hours)
    periods = [p for p in data.hourly if p.start >= now and p.start < end]
    if not periods:
        return 0.0
    return round(float(np.mean([p.mean_watts for p in periods])), 1)


def _peak(data: ForecastData, offset: int) -> dt.datetime | None:
    periods = _periods_for_day(data.intraday, data, offset)
    if not periods:
        return None
    return max(periods, key=lambda p: p.wh).start


SENSORS: dict[str, dict] = {
    "energy_production_today": {
        "name": "Energy production today",
        "unit": "kWh",
        "device_class": SensorDeviceClass.ENERGY,
        "state_class": SensorStateClass.TOTAL,
        "value": energy_today,
    },
    "energy_production_today_remaining": {
        "name": "Energy production today remaining",
        "unit": "kWh",
        "device_class": SensorDeviceClass.ENERGY,
        "state_class": SensorStateClass.TOTAL,
        "value": energy_today_remaining,
    },
    "energy_production_tomorrow": {
        "name": "Energy production tomorrow",
        "unit": "kWh",
        "device_class": SensorDeviceClass.ENERGY,
        "state_class": SensorStateClass.TOTAL,
        "value": energy_tomorrow,
    },
    "power_production_now": {
        "name": "Power production now",
        "unit": "W",
        "device_class": SensorDeviceClass.POWER,
        "state_class": SensorStateClass.MEASUREMENT,
        "value": _power_now,
    },
    "power_production_next_hour": {
        "name": "Power production next hour",
        "unit": "W",
        "device_class": SensorDeviceClass.POWER,
        "state_class": SensorStateClass.MEASUREMENT,
        "value": power_next_hour,
    },
    "power_production_next_12hours": {
        "name": "Power production next 12 hours",
        "unit": "W",
        "device_class": SensorDeviceClass.POWER,
        "state_class": SensorStateClass.MEASUREMENT,
        "value": lambda data: _mean_power_window(data, 12),
    },
    "power_production_next_24hours": {
        "name": "Power production next 24 hours",
        "unit": "W",
        "device_class": SensorDeviceClass.POWER,
        "state_class": SensorStateClass.MEASUREMENT,
        "value": lambda data: _mean_power_window(data, 24),
    },
    "energy_current_hour": {
        "name": "Energy current hour",
        "unit": "kWh",
        "device_class": SensorDeviceClass.ENERGY,
        "state_class": SensorStateClass.TOTAL,
        "value": energy_current_hour,
    },
    "energy_next_hour": {
        "name": "Energy next hour",
        "unit": "kWh",
        "device_class": SensorDeviceClass.ENERGY,
        "state_class": SensorStateClass.TOTAL,
        "value": energy_next_hour,
    },
    "power_highest_peak_time_today": {
        "name": "Power highest peak time today",
        "device_class": SensorDeviceClass.TIMESTAMP,
        "value": lambda data: _peak(data, 0),
    },
    "power_highest_peak_time_tomorrow": {
        "name": "Power highest peak time tomorrow",
        "device_class": SensorDeviceClass.TIMESTAMP,
        "value": lambda data: _peak(data, 1),
    },
    "forecast_accuracy_nmae_14d": {
        "name": "Forecast accuracy nMAE 14d",
        "unit": "%",
        "state_class": SensorStateClass.MEASUREMENT,
        "value": lambda data: data.accuracy_nmae_14d,
        "diagnostic": True,
    },
    "active_layers": {
        "name": "Active layers",
        "value": lambda data: sum(1 for active in data.layers_active.values() if active),
        "diagnostic": True,
        "attributes": lambda data: {"layers": sorted(k for k, v in data.layers_active.items() if v)},
    },
}


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    runtime = entry.runtime_data
    entities = [
        LocalSolarCastSensor(entry, runtime.coordinator, key, spec)
        for key, spec in SENSORS.items()
    ]
    async_add_entities(entities)


class LocalSolarCastSensor(CoordinatorEntity[LocalSolarCastCoordinator], SensorEntity):
    _attr_has_entity_name = True

    def __init__(
        self,
        entry: ConfigEntry,
        coordinator: LocalSolarCastCoordinator,
        key: str,
        spec: dict,
    ) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._key = key
        self._spec = spec
        self._attr_name = spec["name"]
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        if spec.get("unit"):
            self._attr_native_unit_of_measurement = spec["unit"]
        if spec.get("device_class"):
            self._attr_device_class = spec["device_class"]
        if spec.get("state_class"):
            self._attr_state_class = spec["state_class"]
        if spec.get("diagnostic"):
            self._attr_entity_category = EntityCategory.DIAGNOSTIC
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="LocalSolarCast",
            model="PV forecast",
        )

    @property
    def native_value(self):
        if self.coordinator.data is None:
            return None
        try:
            value = self._spec["value"](self.coordinator.data)
        except Exception:  # pragma: no cover - defensive around recorder state
            LOG.exception("sensor %s failed", self._key)
            return None
        if isinstance(value, float):
            return round(value, 3)
        return value

    @property
    def extra_state_attributes(self):
        data = self.coordinator.data
        if data is None:
            return None
        attrs = {
            "system_id": self.coordinator.system_id,
            "layer": data.layer,
            "generated_at": data.generated_at.isoformat(),
        }
        if spec_attrs := self._spec.get("attributes"):
            attrs.update(spec_attrs(data))
        return attrs
