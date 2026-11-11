"""15-minute telemetry push from Home Assistant entities."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.storage import Store

from .api import LocalSolarCastApi, LocalSolarCastError
from .const import (
    CONF_BATTERY_SOC_ENTITY,
    CONF_CURTAILMENT_ENTITY,
    CONF_ENERGY_ENTITY,
    CONF_EXPORT_LIMITED,
    CONF_INVERTER_TEMP_ENTITY,
    CONF_SOC_THRESHOLD,
    DOMAIN,
)

LOG = logging.getLogger(__name__)

_UNIT_FACTORS = {"Wh": 1.0, "kWh": 1000.0, "MWh": 1_000_000.0}
STORAGE_VERSION = 1
BATCH_SIZE = 500


class TelemetryPusher:
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api: LocalSolarCastApi,
        coordinator,
    ) -> None:
        self.hass = hass
        self.entry = entry
        self.api = api
        self.coordinator = coordinator
        self.store = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}.telemetry")
        self.last_value: float | None = None
        self.last_period_start: dt.datetime | None = None
        self.queue: list[dict[str, Any]] = []
        self._unsub = None

    @property
    def options(self) -> dict[str, Any]:
        return self.entry.options

    async def async_start(self) -> None:
        await self._async_load()
        if not self.options.get(CONF_ENERGY_ENTITY):
            return
        self._unsub = async_track_time_change(
            self.hass, self._async_tick, minute=[0, 15, 30, 45], second=30
        )

    async def async_stop(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    async def _async_load(self) -> None:
        stored = await self.store.async_load() or {}
        self.last_value = stored.get("last_value")
        last_start = stored.get("last_period_start")
        self.last_period_start = (
            dt.datetime.fromisoformat(last_start) if last_start else None
        )
        self.queue = list(stored.get("queue") or [])

    async def _async_save(self) -> None:
        await self.store.async_save(
            {
                "last_value": self.last_value,
                "last_period_start": (
                    self.last_period_start.isoformat() if self.last_period_start else None
                ),
                "queue": self.queue[-5000:],
            }
        )

    def _energy_value(self) -> float | None:
        state = self.hass.states.get(self.options.get(CONF_ENERGY_ENTITY, ""))
        if state is None or state.state in ("unknown", "unavailable", ""):
            return None
        unit = state.attributes.get("unit_of_measurement", "Wh")
        factor = _UNIT_FACTORS.get(unit)
        if factor is None:
            LOG.warning("unsupported energy unit %s", unit)
            return None
        try:
            return float(state.state) * factor
        except (TypeError, ValueError):
            return None

    async def _is_curtailed(
        self, period_start: dt.datetime, period_end: dt.datetime
    ) -> bool:
        entity = self.options.get(CONF_CURTAILMENT_ENTITY)
        if entity:
            try:
                changes = await self.hass.async_add_executor_job(
                    state_changes_during_period,
                    self.hass,
                    period_start,
                    period_end,
                    entity,
                )
                states = changes.get(entity, []) if changes else []
                if any(item.state == "on" for item in states):
                    return True
            except Exception as exc:  # recorder may be unavailable
                LOG.debug("curtailment history failed: %s", exc)
        if self.options.get(CONF_EXPORT_LIMITED):
            soc_entity = self.options.get(CONF_BATTERY_SOC_ENTITY)
            state = self.hass.states.get(soc_entity) if soc_entity else None
            if state is not None and state.state not in ("unknown", "unavailable"):
                try:
                    threshold = float(self.options.get(CONF_SOC_THRESHOLD, 98))
                    if float(state.state) >= threshold:
                        return True
                except (TypeError, ValueError):
                    pass
        return False

    def _inverter_temp(self) -> float | None:
        entity = self.options.get(CONF_INVERTER_TEMP_ENTITY)
        state = self.hass.states.get(entity) if entity else None
        if state is None or state.state in ("unknown", "unavailable"):
            return None
        try:
            return float(state.state)
        except (TypeError, ValueError):
            return None

    async def _async_tick(self, now: dt.datetime) -> None:
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt.timezone.utc)
        now = now.astimezone(dt.timezone.utc)
        period_start = (now - dt.timedelta(minutes=15)).replace(second=0, microsecond=0)
        value = self._energy_value()
        if value is None:
            return
        if self.last_value is None:
            self.last_value = value
            self.last_period_start = period_start
            await self._async_save()
            return
        if (
            self.last_period_start
            and period_start - self.last_period_start > dt.timedelta(minutes=20)
        ):
            # A gap: do not invent periods, resume with a fresh baseline.
            self.last_value = value
            self.last_period_start = period_start
            await self._async_save()
            await self._async_flush()
            return

        delta = value - self.last_value
        if delta < 0:
            ac_limit = getattr(self.coordinator.data, "ac_limit_w", 0.0) or 0.0
            plausible_cap = 1.1 * ac_limit * 0.25 if ac_limit else None
            if plausible_cap is None or value <= plausible_cap:
                delta = value
            else:
                LOG.warning("discarding implausible meter reset value %s", value)
                self.last_value = value
                self.last_period_start = period_start
                await self._async_save()
                return
        if delta < 0:
            return

        curtailed = await self._is_curtailed(period_start, now)
        self.queue.append(
            {
                "period_start": period_start.isoformat(),
                "period_min": 15,
                "energy_wh": round(delta, 3),
                "curtailed": curtailed,
                "inverter_temp_c": self._inverter_temp(),
            }
        )
        self.last_value = value
        self.last_period_start = period_start
        await self._async_save()
        await self._async_flush()

    async def _async_flush(self) -> None:
        while self.queue:
            batch = self.queue[:BATCH_SIZE]
            try:
                await self.api.telemetry(
                    self.entry.data["system_id"], batch, origin="push"
                )
            except LocalSolarCastError as exc:
                LOG.warning("telemetry push failed, will retry: %s", exc)
                await self._async_save()
                return
            self.queue = self.queue[BATCH_SIZE:]
            await self._async_save()
