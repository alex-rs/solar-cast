"""Time, period and azimuth conventions shared across the backend."""

from __future__ import annotations

import calendar
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

UTC = dt.timezone.utc
PERIOD_MIN = 15


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def utc_now_ts() -> int:
    return int(utc_now().timestamp())


def ts_to_dt(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(int(ts), UTC)


def dt_to_ts(value: dt.datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("naive datetime is not allowed")
    return int(value.timestamp())


def ensure_utc(value: dt.datetime | pd.Timestamp | str) -> dt.datetime:
    """Parse an ISO-ish value and return an aware UTC datetime."""
    if isinstance(value, str):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return value.tz_convert("UTC").to_pydatetime()
    if value.tzinfo is None:
        raise ValueError("naive datetime is not allowed")
    return value.astimezone(UTC)


def period_start(value: dt.datetime, period_min: int = PERIOD_MIN) -> dt.datetime:
    """Floor to the start of the period containing ``value`` (UTC)."""
    value = ensure_utc(value)
    minute = (value.minute // period_min) * period_min
    return value.replace(minute=minute, second=0, microsecond=0)


def is_aligned(value: dt.datetime, period_min: int = PERIOD_MIN) -> bool:
    value = ensure_utc(value)
    return value.second == 0 and value.microsecond == 0 and value.minute % period_min == 0


def fs_az_to_pvlib(az: float) -> float:
    """Forecast.Solar azimuth (0=S, -90=E, +90=W, +/-180=N) to pvlib (0=N, 90=E)."""
    return float((az + 180.0) % 360.0)


def pvlib_az_to_fs(az: float) -> float:
    """pvlib azimuth to Forecast.Solar azimuth, in [-180, 180)."""
    return (float(az) % 360.0) - 180.0


def local_tz(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def local_midnight(day: dt.date, tz_name: str) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(0, 0), tzinfo=ZoneInfo(tz_name)).astimezone(UTC)


def local_date(ts: int, tz_name: str) -> dt.date:
    return ts_to_dt(ts).astimezone(ZoneInfo(tz_name)).date()


def instant_energy(power_w: pd.Series, period_min: int = PERIOD_MIN) -> pd.Series:
    """Trapezoid energy (Wh) per period, labeled by period start.

    ``power_w`` holds instantaneous power at the period instants.  The final
    instant has no following instant, so it yields no period.
    """
    if len(power_w) < 2:
        return pd.Series(dtype="float64")
    hours = period_min / 60.0
    p = power_w.to_numpy(dtype="float64")
    energy = 0.5 * (p[:-1] + p[1:]) * hours
    return pd.Series(energy, index=power_w.index[:-1], name="energy_wh")


def instant_to_period_mean(power_w: pd.Series, period_min: int = PERIOD_MIN) -> pd.Series:
    energy = instant_energy(power_w, period_min)
    return (energy / (period_min / 60.0)).rename("mean_watts")


def aggregate_periods(
    energy_wh: pd.Series, period_min: int = PERIOD_MIN, target_min: int = 60
) -> pd.Series:
    """Sum 15-min period energy into hourly periods without the trapezoid rule."""
    if target_min % period_min:
        raise ValueError("target period must be a multiple of the source period")
    if len(energy_wh) == 0:
        return energy_wh
    freq = f"{target_min}min"
    return energy_wh.resample(freq, label="left", closed="left").sum(min_count=1)


def hourly_from_instants(power_w: pd.Series) -> pd.Series:
    """Hourly energy: sum of four 15-min trapezoid energies."""
    return aggregate_periods(instant_energy(power_w), 15, 60)


def local_daily_energy(energy_wh: pd.Series, tz_name: str) -> pd.Series:
    """Sum period energy by local calendar date (DST-safe)."""
    if len(energy_wh) == 0:
        return energy_wh
    idx = pd.DatetimeIndex(energy_wh.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local_idx = idx.tz_convert(tz_name)
    return energy_wh.groupby(local_idx.date).sum()


def iso(dt_value: dt.datetime) -> str:
    return ensure_utc(dt_value).isoformat()


def month_days(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def hourly_index(start: dt.datetime, end: dt.datetime) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="h", inclusive="left", tz="UTC")
