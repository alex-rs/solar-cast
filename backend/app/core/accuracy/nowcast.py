"""Clear-sky-index persistence nowcast for the next three hours."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

TAU_MIN = 90.0
HORIZON_H = 3.0


def reference_from_telemetry(
    row: dict | None, now: dt.datetime, zenith_at_ref: float | None
) -> tuple[dt.datetime, float] | None:
    """Pick a telemetry reference period that ended within the last 30 minutes."""
    if row is None or zenith_at_ref is None or zenith_at_ref >= 80.0:
        return None
    start = dt.datetime.fromtimestamp(int(row["period_start"]), dt.timezone.utc)
    period_min = int(row["period_min"])
    end = start + dt.timedelta(minutes=period_min)
    if end > now or (now - end) > dt.timedelta(minutes=30):
        return None
    return start, float(row["mean_watts"])


def apply(
    now: dt.datetime,
    hybrid_w: pd.Series,
    clearsky_w: pd.Series,
    reference_start: dt.datetime | None,
    reference_watts: float | None,
    paco: float,
    kwp_total: float,
) -> tuple[pd.Series, bool]:
    """Apply the decaying clear-sky-index correction.

    Returns the adjusted series and whether the nowcast was active.
    """
    if reference_start is None or reference_watts is None or clearsky_w.empty:
        return hybrid_w.rename("final"), False
    index = pd.DatetimeIndex(hybrid_w.index)
    if reference_start not in index:
        return hybrid_w.rename("final"), False
    reference_cs = float(clearsky_w.loc[reference_start])
    if not np.isfinite(reference_cs) or reference_cs < 0.1 * kwp_total * 1000.0:
        return hybrid_w.rename("final"), False

    k_meas = float(np.clip(reference_watts / reference_cs, 0.0, 1.5))
    k_fc_ref = float(np.clip(hybrid_w.loc[reference_start] / reference_cs, 0.0, 1.5))
    out = hybrid_w.to_numpy(dtype="float64").copy()
    delta_min = (index - pd.Timestamp(reference_start)).total_seconds() / 60.0
    window = (delta_min > 0) & (delta_min <= HORIZON_H * 60.0)
    if not window.any():
        return hybrid_w.rename("final"), False
    cs = clearsky_w.to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        k_fc = np.clip(hybrid_w.to_numpy(dtype="float64") / cs, 0.0, 1.5)
        k_final = k_fc + (k_meas - k_fc_ref) * np.exp(-delta_min / TAU_MIN)
        out[window] = np.clip(k_final[window] * cs[window], 0.0, paco)
    out = np.where(cs > 0, out, 0.0)
    out = np.clip(out, 0.0, paco)
    return pd.Series(out, index=index, name="final"), True


def reference_zenith(zenith: pd.Series, reference_start: dt.datetime) -> float | None:
    try:
        return float(zenith.loc[reference_start])
    except KeyError:
        return None
