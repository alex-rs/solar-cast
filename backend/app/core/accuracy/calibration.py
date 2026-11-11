"""Calibration factor C learned from measured clear days.

The learner compares telemetry against a retrospective clear-sky run, never
against stored forecasts, so repeated runs are idempotent.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging

import numpy as np
import pandas as pd

from ..database import Database
from ..physics_engine import SimResult, SystemConfig
from ..pvgis_client import Horizon
from ..timeconv import instant_to_period_mean, local_daily_energy

LOG = logging.getLogger(__name__)

MIN_CLEAR_DAYS = 3
C_BOUNDS = (0.6, 1.2)


async def get_factor(db: Database, system_id: str) -> float:
    row = await db.get_calibration(system_id)
    return float(row["factor_c"]) if row else 1.0


def learn(
    telemetry: pd.DataFrame,
    clear: SimResult,
    system: SystemConfig,
    timezone: str,
    previous_c: float = 1.0,
    updated_at: int | None = None,
) -> dict:
    """Compute C from 15-min telemetry and a retrospective clear-sky run.

    Returns ``{"factor_c", "clear_days", "samples", "updated_at"}``.
    """
    if telemetry.empty:
        return {
            "factor_c": previous_c,
            "clear_days": 0,
            "samples": 0,
            "updated_at": updated_at or int(dt.datetime.now(dt.timezone.utc).timestamp()),
        }

    tel = telemetry.copy()
    if "period_start" in tel.columns:
        index = pd.to_datetime(tel.pop("period_start"), unit="s", utc=True)
        tel.index = index
    tel = tel.sort_index()
    clear_p = instant_to_period_mean(clear.pac).reindex(tel.index)
    actual = tel["mean_watts"].astype("float64")
    tz = timezone

    local = tel.index.tz_convert(tz)
    tel_dates = pd.Series(local.date, index=tel.index)
    elevation = clear.elevation.reindex(tel.index)
    daylight = elevation > 3.0
    paco = clear.paco

    valid = (
        (tel["curtailed"].fillna(0).astype(int) == 0)
        & clear_p.notna()
        & (clear_p > 0)
        & daylight
        & (actual <= 1.05 * paco)
        & np.isfinite(actual)
    )
    if "snow_flag" in tel:
        valid &= ~tel["snow_flag"].astype(bool)

    samples: list[float] = []
    clear_days = 0
    clear_dates: list[object] = []
    clear_fb = clear.f_b.reindex(tel.index)
    for day, positions in tel.groupby(tel_dates).groups.items():
        idx = pd.DatetimeIndex(positions)
        daylight_idx = idx[elevation.reindex(idx).fillna(0.0) > 3.0]
        if len(daylight_idx) < 4:
            continue
        day_valid = valid.reindex(daylight_idx).fillna(False)
        if day_valid.mean() < 0.8:
            continue
        high = day_valid & (elevation.reindex(daylight_idx) > 20.0)
        if high.sum() < 4:
            continue
        ratio = (actual.reindex(daylight_idx)[high] / clear_p.reindex(daylight_idx)[high]).dropna()
        if len(ratio) < 4:
            continue
        median = float(ratio.median())
        cv = float(ratio.std() / median) if median > 0 else np.inf
        if median < 0.6 or cv >= 0.15:
            continue
        clear_days += 1
        clear_dates.append(day)
        day_mask = (
            (elevation.reindex(daylight_idx) > 25.0)
            & (clear_fb.reindex(daylight_idx) > 0.7)
            & day_valid
        )
        day_samples = ratio.reindex(daylight_idx)[day_mask.reindex(daylight_idx).fillna(False)]
        samples.extend(day_samples.dropna().tolist())

    stamp = updated_at or int(dt.datetime.now(dt.timezone.utc).timestamp())
    if clear_days < MIN_CLEAR_DAYS or not samples:
        return {
            "factor_c": previous_c,
            "clear_days": clear_days,
            "samples": len(samples),
            "updated_at": stamp,
            "clear_dates": clear_dates,
        }
    factor = float(np.clip(np.median(samples), *C_BOUNDS))
    LOG.info("calibration C=%.4f from %d clear days", factor, clear_days)
    return {
        "factor_c": factor,
        "clear_days": clear_days,
        "samples": len(samples),
        "updated_at": stamp,
        "clear_dates": clear_dates,
    }
