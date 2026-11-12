import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app.core.accuracy import nowcast


def _series(values, start="2026-06-21 11:30"):
    index = pd.date_range(start, periods=len(values), freq="15min", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def _row(start: pd.Timestamp, watts: float, period_min: int = 15) -> dict:
    return {
        "period_start": int(start.timestamp()),
        "period_min": period_min,
        "mean_watts": watts,
    }


def test_no_reference_returns_hybrid():
    hybrid = _series([500.0] * 12)
    cs = _series([600.0] * 12)
    result, active = nowcast.apply(hybrid, cs, None, None, 5000.0)
    assert not active
    assert result.equals(hybrid.rename("final"))


def test_delta_selection_rules():
    now = dt.datetime(2026, 6, 21, 12, 30, tzinfo=dt.timezone.utc)
    hybrid = _series([500.0] * 12)
    cs = _series([600.0] * 12)
    zenith = _series([40.0] * 12)

    fresh = [_row(pd.Timestamp("2026-06-21 12:00", tz="UTC"), 400.0)]
    anchor, delta_k, buckets = nowcast.deltas_from_telemetry(
        fresh, now, cs, hybrid, zenith, floor_w=100.0
    )
    assert buckets == 1
    assert anchor == pd.Timestamp("2026-06-21 12:00", tz="UTC").to_pydatetime()
    assert delta_k == pytest.approx(400.0 / 600.0 - 500.0 / 600.0)

    # Sun too low for a meaningful clear-sky index.
    high_zenith = _series([85.0] * 12)
    assert nowcast.deltas_from_telemetry(fresh, now, cs, hybrid, high_zenith, 100.0) is None

    # Clear-sky below the plant floor (night, heavy shading).
    assert nowcast.deltas_from_telemetry(fresh, now, cs, hybrid, zenith, floor_w=1000.0) is None

    # Stale bucket: ended more than the recent window before now.
    stale = [_row(pd.Timestamp("2026-06-21 11:00", tz="UTC"), 400.0)]
    assert nowcast.deltas_from_telemetry(stale, now, cs, hybrid, zenith, 100.0) is None

    # A bucket that has not ended yet must not be used.
    future = [_row(pd.Timestamp("2026-06-21 12:30", tz="UTC"), 400.0)]
    assert nowcast.deltas_from_telemetry(future, now, cs, hybrid, zenith, 100.0) is None

    # Period start must exist in the simulation index.
    outside = [_row(pd.Timestamp("2026-06-21 12:07", tz="UTC"), 400.0)]
    assert nowcast.deltas_from_telemetry(outside, now, cs, hybrid, zenith, 100.0) is None


def test_median_resists_a_single_cloudy_bucket():
    now = dt.datetime(2026, 6, 21, 12, 30, tzinfo=dt.timezone.utc)
    hybrid = _series([500.0] * 12)
    cs = _series([600.0] * 12)
    zenith = _series([40.0] * 12)
    rows = [
        _row(pd.Timestamp("2026-06-21 11:45", tz="UTC"), 540.0),
        _row(pd.Timestamp("2026-06-21 12:00", tz="UTC"), 540.0),
        _row(pd.Timestamp("2026-06-21 12:15", tz="UTC"), 300.0),
    ]
    anchor, delta_k, buckets = nowcast.deltas_from_telemetry(
        rows, now, cs, hybrid, zenith, floor_w=100.0
    )
    assert buckets == 3
    assert anchor == pd.Timestamp("2026-06-21 12:15", tz="UTC").to_pydatetime()
    # Two clean buckets at +0.0667 and one cloudy at -0.3333: median wins.
    assert delta_k == pytest.approx(540.0 / 600.0 - 500.0 / 600.0)

    result, active = nowcast.apply(hybrid, cs, anchor, delta_k, 5000.0)
    assert active
    # The correction is upward, unlike the raw cloudy-bucket reading.
    assert result.loc[pd.Timestamp("2026-06-21 12:30", tz="UTC")] > 500.0


def test_window_uses_at_most_four_buckets():
    now = dt.datetime(2026, 6, 21, 12, 30, tzinfo=dt.timezone.utc)
    hybrid = _series([500.0] * 14)
    cs = _series([600.0] * 14)
    zenith = _series([40.0] * 14)
    # Four buckets fit in the 45-minute window; the 11:00 bucket is too old.
    rows = [
        _row(pd.Timestamp("2026-06-21 11:00", tz="UTC"), 500.0),
        _row(pd.Timestamp("2026-06-21 11:30", tz="UTC"), 500.0),
        _row(pd.Timestamp("2026-06-21 11:45", tz="UTC"), 500.0),
        _row(pd.Timestamp("2026-06-21 12:00", tz="UTC"), 500.0),
        _row(pd.Timestamp("2026-06-21 12:15", tz="UTC"), 500.0),
    ]
    anchor, _, buckets = nowcast.deltas_from_telemetry(
        rows, now, cs, hybrid, zenith, floor_w=100.0
    )
    assert buckets == 4
    assert anchor == pd.Timestamp("2026-06-21 12:15", tz="UTC").to_pydatetime()


def test_correction_decays_and_hits_hybrid_after_horizon():
    index = pd.date_range("2026-06-21 12:00", periods=17, freq="15min", tz="UTC")
    hybrid = pd.Series(500.0, index=index)
    cs = pd.Series(600.0, index=index)
    anchor = index[0].to_pydatetime()
    delta_k = 300.0 / 600.0 - 500.0 / 600.0
    result, active = nowcast.apply(hybrid, cs, anchor, delta_k, 5000.0)
    assert active
    # The reference instant itself is unchanged.
    assert result.loc[index[0]] == pytest.approx(500.0)
    # At one hour the negative correction is already smaller than at 15 minutes.
    dev_15 = abs(result.iloc[1] - 500.0)
    dev_60 = abs(result.iloc[4] - 500.0)
    assert dev_60 < dev_15
    # Beyond the 3-hour horizon the hybrid forecast is returned exactly.
    assert result.iloc[-1] == pytest.approx(500.0)
    assert abs(result.iloc[-5] - 500.0) / 500.0 < 0.06


def test_single_bucket_matches_legacy_formula():
    index = pd.date_range("2026-06-21 12:00", periods=9, freq="15min", tz="UTC")
    hybrid = pd.Series([480.0, 500.0, 520.0, 540.0, 560.0, 540.0, 520.0, 500.0, 480.0], index=index)
    cs = pd.Series([600.0] * 9, index=index)
    anchor = index[0].to_pydatetime()
    measured = 420.0
    delta_k = measured / 600.0 - hybrid.loc[index[0]] / 600.0
    result, active = nowcast.apply(hybrid, cs, anchor, delta_k, 5000.0)
    assert active

    expected = hybrid.copy()
    minutes = np.arange(len(index)) * 15.0
    window = minutes > 0
    k = hybrid.to_numpy() / cs.to_numpy()
    k[window] = k[window] + delta_k * np.exp(-minutes[window] / nowcast.TAU_MIN)
    expected.iloc[window] = np.clip(k[window] * cs.to_numpy()[window], 0.0, 5000.0)
    assert np.allclose(result.to_numpy(), expected.to_numpy())


def test_night_has_no_phantom_watts():
    index = pd.date_range("2026-06-21 19:00", periods=5, freq="15min", tz="UTC")
    hybrid = pd.Series(100.0, index=index)
    cs = pd.Series([800.0, 200.0, 0.0, 0.0, 0.0], index=index)
    result, _ = nowcast.apply(
        hybrid,
        cs,
        index[0].to_pydatetime(),
        100.0 / 800.0 - 100.0 / 800.0,
        5000.0,
    )
    assert result.iloc[2:].eq(0.0).all()


def test_never_exceeds_paco():
    index = pd.date_range("2026-06-21 12:00", periods=5, freq="15min", tz="UTC")
    hybrid = pd.Series(4900.0, index=index)
    cs = pd.Series(300.0, index=index)
    delta_k = 290.0 / 300.0 - 4900.0 / 300.0
    result, _ = nowcast.apply(hybrid, cs, index[0].to_pydatetime(), delta_k, 5000.0)
    assert result.max() <= 5000.0


def test_sixty_minute_bucket_is_eligible():
    now = dt.datetime(2026, 6, 21, 12, 30, tzinfo=dt.timezone.utc)
    index = pd.date_range("2026-06-21 10:30", periods=17, freq="15min", tz="UTC")
    hybrid = pd.Series(500.0, index=index)
    cs = pd.Series(600.0, index=index)
    zenith = pd.Series(40.0, index=index)
    # 11:00-12:00 bucket, still inside the 45-minute eligibility window.
    rows = [_row(pd.Timestamp("2026-06-21 11:00", tz="UTC"), 480.0, period_min=60)]
    anchor, delta_k, buckets = nowcast.deltas_from_telemetry(
        rows, now, cs, hybrid, zenith, floor_w=100.0
    )
    assert buckets == 1
    assert anchor == pd.Timestamp("2026-06-21 11:00", tz="UTC").to_pydatetime()
    assert delta_k == pytest.approx(480.0 / 600.0 - 500.0 / 600.0)
