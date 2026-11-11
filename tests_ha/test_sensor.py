import datetime as dt

import pytest

from custom_components.solarcast.coordinator import ForecastData, ForecastPeriod
from custom_components.solarcast.sensor import (
    SENSORS,
    _mean_power_window,
    _peak,
    _power_now,
    energy_current_hour,
    energy_next_hour,
    energy_today,
    energy_today_remaining,
    energy_tomorrow,
    power_next_hour,
)
from tests_ha.fixtures import make_data


def test_energy_sensors_from_fixture():
    data = make_data()
    assert energy_today(data) == pytest.approx(2.4)
    assert energy_tomorrow(data) == pytest.approx(2.4)
    remaining = energy_today_remaining(data)
    assert 0 < remaining <= 2.4
    assert energy_current_hour(data) == pytest.approx(0.1)
    assert energy_next_hour(data) == pytest.approx(0.1)


def test_power_sensors():
    data = make_data()
    assert _power_now(data) == pytest.approx(500.0)
    assert power_next_hour(data) == pytest.approx(100.0)
    assert _mean_power_window(data, 12) == pytest.approx(100.0)
    peak = _peak(data, 0)
    assert peak is not None
    assert peak.astimezone(data.tz).date() == dt.datetime.now(data.tz).date()


def test_diagnostic_sensors():
    data = make_data()
    assert SENSORS["forecast_accuracy_nmae_14d"]["value"](data) == pytest.approx(12.5)
    assert SENSORS["active_layers"]["value"](data) == 3
    attrs = SENSORS["active_layers"]["attributes"](data)
    assert attrs["layers"] == ["calibration", "nowcast", "shading"]


def test_telemetry_age_sensor():
    data = make_data()
    assert SENSORS["telemetry_age_minutes"]["value"](data) == pytest.approx(10.0)
    assert SENSORS["telemetry_age_minutes"]["diagnostic"] is True
