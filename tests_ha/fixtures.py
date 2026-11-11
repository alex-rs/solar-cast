import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.solarcast.const import (
    CONF_AC_KW,
    CONF_ENERGY_ENTITY,
    CONF_INVERTER,
    CONF_INVERTER_ID,
    CONF_INVERTERS,
    CONF_LAT,
    CONF_LON,
    CONF_NAME,
    CONF_PLANES,
    CONF_SYSTEM_ID,
    CONF_TOKEN,
    CONF_URL,
    DOMAIN,
)
from custom_components.solarcast.coordinator import ForecastData, ForecastPeriod

TZ = ZoneInfo("Europe/Belgrade")


def make_data() -> ForecastData:
    now = dt.datetime.now(TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    hourly = []
    for hour in range(72):
        start = midnight + dt.timedelta(hours=hour)
        hourly.append(
            ForecastPeriod(
                start=start,
                end=start + dt.timedelta(hours=1),
                wh=100.0,
                mean_watts=100.0,
            )
        )
    quarter = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    intraday = []
    for slot in range(-4, 8):
        start = quarter + dt.timedelta(minutes=15 * slot)
        intraday.append(
            ForecastPeriod(
                start=start,
                end=start + dt.timedelta(minutes=15),
                wh=125.0,
                mean_watts=500.0,
            )
        )
    return ForecastData(
        hourly=hourly,
        intraday=intraday,
        timezone="Europe/Belgrade",
        layer="final",
        layers_active={
            "calibration": True,
            "shading": True,
            "mos": False,
            "nowcast": True,
        },
        generated_at=now,
        ac_limit_w=5000.0,
        accuracy_nmae_14d=12.5,
        accuracy_detail={},
    )


def make_entry() -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="Home",
        version=2,
        data={
            CONF_URL: "http://backend:8000",
            CONF_TOKEN: None,
            CONF_SYSTEM_ID: "home",
        },
        options={
            CONF_NAME: "Home",
            CONF_LAT: 44.8125,
            CONF_LON: 20.4612,
            CONF_INVERTERS: [
                {CONF_INVERTER_ID: "main", CONF_AC_KW: 5.0},
            ],
            "system_loss": 0.1,
            "damping_morning": 0.0,
            "damping_evening": 0.0,
            CONF_PLANES: [
                {
                    "dec": 30,
                    "az": 0,
                    "kwp": 5.0,
                    CONF_INVERTER: "main",
                }
            ],
            CONF_ENERGY_ENTITY: "sensor.pv_energy",
            "soc_threshold": 98,
            "export_limited": False,
        },
        entry_id="test_entry",
    )


def runtime_with(data=None) -> SimpleNamespace:
    data = data or make_data()
    coordinator = SimpleNamespace(data=data, system_id="home")
    return SimpleNamespace(api=None, coordinator=coordinator, telemetry=None)
