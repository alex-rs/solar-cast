"""Whole-day totals must not decay as the day advances."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.core.clearsky import simulate_clearsky
from app.core.pipeline import ForecastPipeline, system_config
from app.core.pvgis_client import flat_fallback
from tests.test_local_model import SYSTEM

TZ = ZoneInfo("Europe/Belgrade")
DAY = dt.date(2026, 6, 21)


def _sim():
    start = pd.Timestamp(dt.datetime.combine(DAY, dt.time(0), tzinfo=TZ))
    index = pd.date_range(start, periods=96, freq="15min")
    return simulate_clearsky(index, system_config(SYSTEM), flat_fallback())


def _now(hour: int) -> dt.datetime:
    return dt.datetime.combine(DAY, dt.time(hour), tzinfo=TZ).astimezone(dt.timezone.utc)


def _block(hour: int, measured: float | None = None) -> dict:
    pipeline = ForecastPipeline.__new__(ForecastPipeline)  # _today_block holds no state
    return pipeline._today_block(_sim(), _now(hour), "Europe/Belgrade", measured)


def test_today_total_is_the_whole_day_at_noon():
    block = _block(12)
    assert block["date"] == DAY.isoformat()
    assert block["wh"] > 0
    assert block["elapsed_wh"] > 0
    assert block["remaining_wh"] > 0
    assert block["elapsed_wh"] + block["remaining_wh"] == pytest.approx(block["wh"])
    assert block["source"] == "forecast"


def test_today_total_does_not_decay_towards_evening():
    """The reported bug: "today" fell 1.47 -> 0.62 -> 0.19 -> 0.0 through the day."""
    noon = _block(12)["wh"]
    evening = _block(21)["wh"]
    assert evening == pytest.approx(noon)
    assert _block(21)["remaining_wh"] == pytest.approx(0.0, abs=1e-6)
    assert _block(6)["remaining_wh"] > noon / 2


def test_measured_energy_replaces_the_elapsed_estimate():
    measured = 1234.0
    block = _block(15, measured=measured)
    assert block["measured_wh"] == pytest.approx(measured)
    assert block["elapsed_wh"] == pytest.approx(measured)
    assert block["wh"] == pytest.approx(measured + block["remaining_wh"])
    assert block["source"] == "measured+forecast"
    # Without telemetry the same day keeps the simulated elapsed portion.
    assert _block(15)["wh"] == pytest.approx(_block(12)["wh"])


def test_zero_measured_is_respected():
    """No production yet (e.g. telemetry exists but the array is snow-covered)."""
    block = _block(8, measured=0.0)
    assert block["measured_wh"] == pytest.approx(0.0)
    assert block["wh"] == pytest.approx(block["remaining_wh"])
