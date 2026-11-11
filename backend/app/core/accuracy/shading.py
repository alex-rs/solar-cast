"""Beam shading factor K(azimuth, elevation)."""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from ..database import Database
from ..physics_engine import SimResult, SystemConfig
from ..timeconv import instant_to_period_mean

LOG = logging.getLogger(__name__)

AZ_STEP = 5
EL_STEP = 2
AZ_COUNT = 360 // AZ_STEP
EL_COUNT = 90 // EL_STEP
MIN_CLEAR_DAYS = 5
MIN_SAMPLES = 3
MIN_DAYS_PER_BUCKET = 2
SHRINK = 3.0


class ShadingLookup:
    def __init__(self, grid: np.ndarray, counts: np.ndarray | None = None) -> None:
        self.grid = grid
        self.counts = counts

    def __call__(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        az = np.asarray(azimuth, dtype="float64") % 360.0
        el = np.clip(np.asarray(elevation, dtype="float64"), 0.0, 89.999)
        af = az / AZ_STEP - 0.5
        ef = el / EL_STEP - 0.5
        a0 = np.floor(af).astype(int) % AZ_COUNT
        a1 = (a0 + 1) % AZ_COUNT
        e0 = np.clip(np.floor(ef).astype(int), 0, EL_COUNT - 1)
        e1 = np.clip(e0 + 1, 0, EL_COUNT - 1)
        wa = af - np.floor(af)
        we = ef - np.floor(ef)

        total = np.zeros_like(az, dtype="float64")
        weight = np.zeros_like(az, dtype="float64")
        for ai, aw in ((a0, 1 - wa), (a1, wa)):
            for ei, ew in ((e0, 1 - we), (e1, we)):
                value = self.grid[ai, ei]
                w = aw * ew
                mask = ~np.isnan(value)
                total[mask] += value[mask] * w[mask]
                weight[mask] += w[mask]
        out = np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), np.nan)
        return np.where(np.isnan(out), 1.0, out)


def lookup_from_rows(rows: list[dict]) -> ShadingLookup | None:
    if not rows:
        return None
    grid = np.full((AZ_COUNT, EL_COUNT), np.nan)
    counts = np.zeros((AZ_COUNT, EL_COUNT))
    for row in rows:
        ai = int(row["az_bucket"]) // AZ_STEP
        ei = int(row["el_bucket"]) // EL_STEP
        if 0 <= ai < AZ_COUNT and 0 <= ei < EL_COUNT:
            grid[ai, ei] = float(row["k_beam"])
            counts[ai, ei] = float(row["sample_count"])
    return ShadingLookup(grid, counts)


async def load(db: Database, system_id: str) -> ShadingLookup | None:
    return lookup_from_rows(await db.get_shading(system_id))


def learn(
    telemetry: pd.DataFrame,
    clear: SimResult,
    calibration_c: float,
    system: SystemConfig,
    timezone: str,
    updated_at: int,
    clear_days: int,
    clear_dates: set | None = None,
) -> list[dict]:
    """Learn K from measured power and the retrospective clear-sky run."""
    if clear_days < MIN_CLEAR_DAYS or telemetry.empty:
        return []
    tel = telemetry.copy()
    if "period_start" in tel.columns:
        tel.index = pd.to_datetime(tel.pop("period_start"), unit="s", utc=True)
    tel = tel.sort_index()
    clear_p = instant_to_period_mean(clear.pac).reindex(tel.index)
    actual = tel["mean_watts"].astype("float64")
    f_b = clear.f_b.reindex(tel.index)
    f_d = clear.f_d.reindex(tel.index)
    elevation = clear.elevation.reindex(tel.index)
    azimuth = clear.azimuth.reindex(tel.index)
    valid = (
        (tel["curtailed"].fillna(0).astype(int) == 0)
        & (clear_p > 0)
        & (elevation > 3.0)
        & (f_b >= 0.5)
        & np.isfinite(actual)
        & np.isfinite(f_b)
    )
    idx = valid[valid].index
    if len(idx) == 0:
        return []
    local_days = pd.Series(idx.tz_convert(timezone).date, index=idx)
    if clear_dates is not None:
        allowed = local_days.isin(clear_dates)
        idx = idx[allowed.to_numpy()]
        if len(idx) == 0:
            return []
        local_days = local_days.reindex(idx)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = actual[idx] / (calibration_c * clear_p[idx])
        k = np.clip((ratio - f_d[idx]) / f_b[idx], 0.0, 1.1)
    samples = pd.DataFrame(
        {
            "az": azimuth[idx].to_numpy(),
            "el": elevation[idx].to_numpy(),
            "k": np.asarray(k, dtype="float64"),
            "day": local_days.to_numpy(),
        },
        index=idx,
    ).dropna()
    samples["az_bucket"] = (
        (samples["az"] % 360.0) // AZ_STEP * AZ_STEP
    ).astype(int)
    samples["el_bucket"] = (
        np.clip(samples["el"] // EL_STEP * EL_STEP, 0, 88)
    ).astype(int)

    rows: list[dict] = []
    for (az_bucket, el_bucket), group in samples.groupby(["az_bucket", "el_bucket"]):
        day_count = group["day"].nunique()
        n = len(group)
        if n < MIN_SAMPLES or day_count < MIN_DAYS_PER_BUCKET:
            continue
        k_med = float(group["k"].median())
        k_shrunk = (n * k_med + SHRINK * 1.0) / (n + SHRINK)
        rows.append(
            {
                "az_bucket": int(az_bucket),
                "el_bucket": int(el_bucket),
                "k_beam": float(np.clip(k_shrunk, 0.0, 1.1)),
                "sample_count": int(n),
                "day_count": int(day_count),
                "updated_at": int(updated_at),
            }
        )
    return rows
