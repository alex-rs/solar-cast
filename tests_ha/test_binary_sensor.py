import dataclasses

import pytest

from custom_components.solarcast.binary_sensor import SolarCastTelemetryFresh
from custom_components.solarcast.const import TELEMETRY_STALE_MINUTES
from tests_ha.fixtures import make_data, make_entry


def _entity(age):
    data = dataclasses.replace(make_data(), telemetry_age_minutes=age)
    coordinator = type("Coordinator", (), {"data": data, "system_id": "home"})()
    return SolarCastTelemetryFresh(make_entry(), coordinator)


def test_fresh_when_recent():
    entity = _entity(TELEMETRY_STALE_MINUTES - 1)
    assert entity.is_on is True
    attrs = entity.extra_state_attributes
    assert attrs["nowcast"] is True
    assert attrs["nowcast_buckets"] == 3
    assert attrs["calibration_c"] == pytest.approx(0.93)


def test_stale_when_old():
    assert _entity(TELEMETRY_STALE_MINUTES + 1).is_on is False


def test_unknown_without_telemetry():
    entity = _entity(None)
    assert entity.is_on is None
