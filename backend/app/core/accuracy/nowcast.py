"""Clear-sky-index persistence nowcast for the next three hours."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

TAU_MIN = 90.0
HORIZON_H = 3.0
RECENT_WINDOW_MIN = 45.0
MAX_BUCKETS = 4
MAX_ZENITH_DEG = 80.0
K_LIMIT = 1.5


def deltas_from_telemetry(
    rows: list[dict],
    now: dt.datetime,
    clearsky_w: pd.Series,
    hybrid_w: pd.Series,
    zenith: pd.Series,
    floor_w: float,
) -> tuple[dt.datetime, float, int] | None:
    """Measured clear-sky-index bias over recent telemetry buckets.

    Returns ``(anchor_start, median_delta_k, buckets)`` or ``None`` when no
    bucket ended within the last ``RECENT_WINDOW_MIN`` minutes at a usable sun
    angle and clear-sky level. The median over up to ``MAX_BUCKETS`` buckets
    keeps a single passing cloud from flattening the whole correction window.
    """
    if hybrid_w.empty:
        return None
    index = pd.DatetimeIndex(hybrid_w.index)
    candidates: list[tuple[pd.Timestamp, float]] = []
    for row in rows:
        start = dt.datetime.fromtimestamp(int(row["period_start"]), dt.timezone.utc)
        period_min = int(row["period_min"])
        end = start + dt.timedelta(minutes=period_min)
        if end > now or (now - end) > dt.timedelta(minutes=RECENT_WINDOW_MIN):
            continue
        stamp = pd.Timestamp(start)
        if stamp not in index:
            continue
        try:
            sun_zenith = float(zenith.loc[stamp])
        except KeyError:
            continue
        if not np.isfinite(sun_zenith) or sun_zenith >= MAX_ZENITH_DEG:
            continue
        reference_cs = float(clearsky_w.loc[stamp])
        if not np.isfinite(reference_cs) or reference_cs < floor_w:
            continue
        k_meas = float(np.clip(float(row["mean_watts"]) / reference_cs, 0.0, K_LIMIT))
        k_fc = float(np.clip(float(hybrid_w.loc[stamp]) / reference_cs, 0.0, K_LIMIT))
        candidates.append((stamp, k_meas - k_fc))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    recent = candidates[-MAX_BUCKETS:]
    anchor = recent[-1][0].to_pydatetime()
    delta_k = float(np.median([delta for _, delta in recent]))
    return anchor, delta_k, len(recent)


def apply(
    hybrid_w: pd.Series,
    clearsky_w: pd.Series,
    anchor_start: dt.datetime | None,
    delta_k: float | None,
    paco: float,
) -> tuple[pd.Series, bool]:
    """Apply the decaying clear-sky-index correction.

    Returns the adjusted series and whether the nowcast was active.
    """
    if anchor_start is None or delta_k is None or clearsky_w.empty:
        return hybrid_w.rename("final"), False
    index = pd.DatetimeIndex(hybrid_w.index)
    if pd.Timestamp(anchor_start) not in index:
        return hybrid_w.rename("final"), False

    out = hybrid_w.to_numpy(dtype="float64").copy()
    delta_min = (index - pd.Timestamp(anchor_start)).total_seconds() / 60.0
    window = (delta_min > 0) & (delta_min <= HORIZON_H * 60.0)
    if not window.any():
        return hybrid_w.rename("final"), False
    cs = clearsky_w.to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        k_fc = np.clip(hybrid_w.to_numpy(dtype="float64") / cs, 0.0, K_LIMIT)
        k_final = k_fc + float(delta_k) * np.exp(-delta_min / TAU_MIN)
        out[window] = np.clip(k_final[window] * cs[window], 0.0, paco)
    out = np.where(cs > 0, out, 0.0)
    out = np.clip(out, 0.0, paco)
    return pd.Series(out, index=index, name="final"), True
