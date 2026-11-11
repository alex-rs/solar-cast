import datetime as dt
from types import SimpleNamespace

import pytest

from custom_components.localsolarcast.api import LocalSolarCastError
from custom_components.localsolarcast.const import (
    CONF_BATTERY_SOC_ENTITY,
    CONF_CURTAILMENT_ENTITY,
    CONF_ENERGY_ENTITY,
    CONF_EXPORT_LIMITED,
    CONF_SOC_THRESHOLD,
)
import custom_components.localsolarcast.telemetry as telemetry_module
from custom_components.localsolarcast.telemetry import TelemetryPusher
from tests_ha.fixtures import runtime_with


class FakeApi:
    def __init__(self) -> None:
        self.calls = []
        self.fail = False

    async def telemetry(self, system_id, items, origin="push"):
        if self.fail:
            raise LocalSolarCastError("down")
        self.calls.append((system_id, list(items), origin))
        return {"accepted": len(items), "rejected": []}


def _pusher(hass, options=None):
    from tests_ha.fixtures import make_entry

    entry = make_entry()
    entry.add_to_hass(hass)
    if options:
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, **options}
        )
        entry = hass.config_entries.async_get_entry(entry.entry_id)
    api = FakeApi()
    pusher = TelemetryPusher(hass, entry, api, runtime_with().coordinator)
    return pusher, api, entry


def _tick_time(minutes: int) -> dt.datetime:
    base = dt.datetime.now(dt.timezone.utc).replace(second=30, microsecond=0)
    return base + dt.timedelta(minutes=minutes)


@pytest.mark.asyncio
async def test_normal_delta_and_baseline(hass):
    pusher, api, entry = _pusher(hass)
    hass.states.async_set("sensor.pv_energy", "1000", {"unit_of_measurement": "Wh"})
    now = _tick_time(0)
    await pusher._async_tick(now)
    assert api.calls == []
    assert pusher.last_value == 1000

    hass.states.async_set("sensor.pv_energy", "1200", {"unit_of_measurement": "Wh"})
    await pusher._async_tick(now + dt.timedelta(minutes=15))
    assert len(api.calls) == 1
    system_id, items, origin = api.calls[0]
    assert system_id == "home"
    assert origin == "push"
    assert items[0]["energy_wh"] == pytest.approx(200.0)
    assert items[0]["period_min"] == 15
    assert items[0]["period_start"] == now.replace(second=0).isoformat()


@pytest.mark.asyncio
async def test_kwh_conversion(hass):
    pusher, api, _ = _pusher(hass)
    hass.states.async_set("sensor.pv_energy", "1.0", {"unit_of_measurement": "kWh"})
    await pusher._async_tick(_tick_time(0))
    hass.states.async_set("sensor.pv_energy", "1.2", {"unit_of_measurement": "kWh"})
    await pusher._async_tick(_tick_time(15))
    assert api.calls[0][1][0]["energy_wh"] == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_unavailable_and_gap_are_skipped(hass):
    pusher, api, _ = _pusher(hass)
    pusher.last_value = 1000.0
    pusher.last_period_start = _tick_time(0)
    hass.states.async_set("sensor.pv_energy", "unavailable")
    await pusher._async_tick(_tick_time(15))
    assert api.calls == []
    assert pusher.last_value == 1000.0

    pusher.last_period_start = _tick_time(0) - dt.timedelta(hours=1)
    hass.states.async_set("sensor.pv_energy", "1500", {"unit_of_measurement": "Wh"})
    await pusher._async_tick(_tick_time(15))
    assert api.calls == []
    assert pusher.last_value == 1500.0


@pytest.mark.asyncio
async def test_meter_reset_handling(hass):
    pusher, api, _ = _pusher(hass)
    pusher.last_value = 1000.0
    pusher.last_period_start = _tick_time(0)
    hass.states.async_set("sensor.pv_energy", "50", {"unit_of_measurement": "Wh"})
    await pusher._async_tick(_tick_time(15))
    assert api.calls[0][1][0]["energy_wh"] == pytest.approx(50.0)

    pusher.last_value = 200000.0
    pusher.last_period_start = _tick_time(0)
    hass.states.async_set("sensor.pv_energy", "100000", {"unit_of_measurement": "Wh"})
    await pusher._async_tick(_tick_time(30))
    assert len(api.calls) == 1


@pytest.mark.asyncio
async def test_curtailment_binary_sensor(hass, monkeypatch):
    pusher, api, entry = _pusher(
        hass, {CONF_CURTAILMENT_ENTITY: "binary_sensor.curtail"}
    )
    monkeypatch.setattr(
        telemetry_module,
        "state_changes_during_period",
        lambda *args, **kwargs: {"binary_sensor.curtail": [SimpleNamespace(state="on")]},
    )
    pusher.last_value = 1000.0
    pusher.last_period_start = _tick_time(0)
    hass.states.async_set("sensor.pv_energy", "1100", {"unit_of_measurement": "Wh"})
    await pusher._async_tick(_tick_time(15))
    assert api.calls[0][1][0]["curtailed"] is True


@pytest.mark.asyncio
async def test_soc_export_limit_rule(hass):
    pusher, api, entry = _pusher(
        hass,
        {
            CONF_BATTERY_SOC_ENTITY: "sensor.soc",
            CONF_SOC_THRESHOLD: 98,
            CONF_EXPORT_LIMITED: True,
        },
    )
    hass.states.async_set("sensor.soc", "99")
    pusher.last_value = 1000.0
    pusher.last_period_start = _tick_time(0)
    hass.states.async_set("sensor.pv_energy", "1100", {"unit_of_measurement": "Wh"})
    await pusher._async_tick(_tick_time(15))
    assert api.calls[0][1][0]["curtailed"] is True


@pytest.mark.asyncio
async def test_queue_survives_backend_failure(hass):
    pusher, api, _ = _pusher(hass)
    pusher.last_value = 1000.0
    pusher.last_period_start = _tick_time(0)
    api.fail = True
    hass.states.async_set("sensor.pv_energy", "1100", {"unit_of_measurement": "Wh"})
    await pusher._async_tick(_tick_time(15))
    assert len(pusher.queue) == 1
    assert api.calls == []
    stored = await pusher.store.async_load()
    assert len(stored["queue"]) == 1

    api.fail = False
    await pusher._async_flush()
    assert pusher.queue == []
    assert len(api.calls) == 1
