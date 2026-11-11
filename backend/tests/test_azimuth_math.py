import datetime as dt
import hashlib

import numpy as np
import pandas as pd
import pytest

from app.core.timeconv import (
    aggregate_periods,
    dt_to_ts,
    fs_az_to_pvlib,
    instant_energy,
    instant_to_period_mean,
    is_aligned,
    local_daily_energy,
    period_start,
    pvlib_az_to_fs,
    ts_to_dt,
)


def test_fs_to_pvlib_full_circle():
    table = {
        0.0: 180.0,   # South
        -90.0: 90.0,  # East
        90.0: 270.0,  # West
        180.0: 0.0,   # North
        -180.0: 0.0,
        -45.0: 135.0,
    }
    for fs, expected in table.items():
        assert fs_az_to_pvlib(fs) == pytest.approx(expected)


def test_pvlib_to_fs_roundtrip():
    for az_fs in (-180, -135, -90, -45, 0, 45, 90, 135, 179.9):
        assert pvlib_az_to_fs(fs_az_to_pvlib(az_fs)) == pytest.approx(az_fs, abs=1e-9)


def test_period_alignment_and_energy():
    stamp = dt.datetime(2026, 11, 11, 12, 7, 33, tzinfo=dt.timezone.utc)
    assert period_start(stamp) == dt.datetime(2026, 11, 11, 12, 0, tzinfo=dt.timezone.utc)
    assert period_start(stamp, 60) == dt.datetime(2026, 11, 11, 12, 0, tzinfo=dt.timezone.utc)
    assert is_aligned(period_start(stamp))
    assert not is_aligned(stamp)

    idx = pd.date_range("2026-06-21 10:00", periods=3, freq="15min", tz="UTC")
    power = pd.Series([0.0, 400.0, 800.0], index=idx)
    energy = instant_energy(power)
    assert list(energy) == pytest.approx([50.0, 150.0])
    assert np.allclose(instant_to_period_mean(power), [200.0, 600.0])
    hourly = aggregate_periods(energy, 15, 60)
    assert hourly.sum() == pytest.approx(200.0)


def test_naive_datetimes_rejected():
    with pytest.raises(ValueError):
        dt_to_ts(dt.datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        period_start(dt.datetime(2026, 1, 1))


def test_ts_roundtrip_is_utc():
    stamp = dt.datetime(2026, 3, 29, 1, 30, tzinfo=dt.timezone.utc)
    assert ts_to_dt(dt_to_ts(stamp)) == stamp


def test_local_daily_dst_grouping():
    # Europe/Belgrade DST starts 2026-03-29; days still get four 15-min cells per hour.
    idx = pd.date_range("2026-03-28 22:00", "2026-03-29 03:00", freq="15min", tz="UTC")
    energy = pd.Series(1.0, index=idx)
    daily = local_daily_energy(energy, "Europe/Belgrade")
    assert daily.loc[dt.date(2026, 3, 28)] == pytest.approx(4.0)
    assert daily.loc[dt.date(2026, 3, 29)] == pytest.approx(17.0)
    assert daily.sum() == pytest.approx(len(energy))


def test_plane_id_formula():
    plane_id = hashlib.sha1(f"{'home'}|{30.0:.1f}|{-90.0:.1f}|{2.5:.3f}".encode()).hexdigest()[:12]
    assert len(plane_id) == 12
    assert plane_id == hashlib.sha1(b"home|30.0|-90.0|2.500").hexdigest()[:12]
