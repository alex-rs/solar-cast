import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app.core.accuracy import nowcast


def _series(values, start="2026-06-21 12:00"):
    index = pd.date_range(start, periods=len(values), freq="15min", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_no_reference_returns_hybrid():
    hybrid = _series([500.0] * 12)
    cs = _series([600.0] * 12)
    result, active = nowcast.apply(
        dt.datetime(2026, 6, 21, 12, 30, tzinfo=dt.timezone.utc),
        hybrid,
        cs,
        None,
        None,
        5000.0,
        5.0,
    )
    assert not active
    assert result.equals(hybrid)


def test_reference_selection_rules():
    now = dt.datetime(2026, 6, 21, 12, 30, tzinfo=dt.timezone.utc)
    row = {"period_start": int((now - dt.timedelta(minutes=30)).timestamp()), "period_min": 15, "mean_watts": 400.0}
    start, watts = nowcast.reference_from_telemetry(row, now, zenith_at_ref=40.0)
    assert watts == 400.0
    assert nowcast.reference_from_telemetry(row, now, 85.0) is None
    stale = {"period_start": int((now - dt.timedelta(hours=2)).timestamp()), "period_min": 15, "mean_watts": 400.0}
    assert nowcast.reference_from_telemetry(stale, now, 40.0) is None


def test_correction_decays_and_hits_hybrid_after_horizon():
    index = pd.date_range("2026-06-21 12:00", periods=17, freq="15min", tz="UTC")
    hybrid = pd.Series(500.0, index=index)
    cs = pd.Series(600.0, index=index)
    reference = index[0]
    result, active = nowcast.apply(
        dt.datetime(2026, 6, 21, 11, 0, tzinfo=dt.timezone.utc),
        hybrid,
        cs,
        reference.to_pydatetime(),
        300.0,
        5000.0,
        5.0,
    )
    assert active
    # The reference instant itself is unchanged.
    assert result.loc[reference] == pytest.approx(500.0)
    # At one hour the negative correction is already smaller than at 15 minutes.
    dev_15 = abs(result.iloc[1] - 500.0)
    dev_60 = abs(result.iloc[4] - 500.0)
    assert dev_60 < dev_15
    # Beyond the 3-hour horizon the hybrid forecast is returned exactly.
    assert result.iloc[-1] == pytest.approx(500.0)
    assert abs(result.iloc[-5] - 500.0) / 500.0 < 0.06


def test_night_has_no_phantom_watts():
    index = pd.date_range("2026-06-21 19:00", periods=5, freq="15min", tz="UTC")
    hybrid = pd.Series(100.0, index=index)
    cs = pd.Series([800.0, 200.0, 0.0, 0.0, 0.0], index=index)
    result, _ = nowcast.apply(
        dt.datetime(2026, 6, 21, 20, 0, tzinfo=dt.timezone.utc),
        hybrid,
        cs,
        index[0].to_pydatetime(),
        100.0,
        5000.0,
        5.0,
    )
    assert result.iloc[2:].eq(0.0).all()


def test_never_exceeds_paco():
    index = pd.date_range("2026-06-21 12:00", periods=5, freq="15min", tz="UTC")
    hybrid = pd.Series(4900.0, index=index)
    cs = pd.Series(300.0, index=index)
    result, _ = nowcast.apply(
        dt.datetime(2026, 6, 21, 11, 0, tzinfo=dt.timezone.utc),
        hybrid,
        cs,
        index[0].to_pydatetime(),
        290.0,
        5000.0,
        5.0,
    )
    assert result.max() <= 5000.0
