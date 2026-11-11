"""Learn Open-Meteo blending weights from telemetry and per-model GHI."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .clearsky import clearsky_ghi, simulate_clearsky
from .database import Database
from .pipeline import system_config
from .pvgis_client import Horizon
from .timeconv import instant_to_period_mean

LOG = logging.getLogger(__name__)

MIN_SAMPLES = 200
MIN_DAYS = 10
GATE_RATIO = 0.99


def fit_weights(
    x: np.ndarray,
    y: np.ndarray,
    default_weights: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    weights, _ = nnls(x, y)
    total = float(weights.sum())
    if total <= 0:
        return None
    weights = weights / total
    learned_mae = float(np.mean(np.abs(x @ weights - y)))
    defaults = np.asarray(default_weights, dtype="float64")
    default_total = float(defaults.sum())
    if default_total <= 0:
        return None
    defaults = defaults / default_total
    default_mae = float(np.mean(np.abs(x @ defaults - y)))
    return weights, learned_mae, default_mae


async def learn_for_location(
    db: Database,
    system_row: dict,
    horizon: Horizon,
    models: list[str],
    default_weights: list[float],
    updated_at: int,
) -> dict[str, float] | None:
    system = system_config(system_row)
    start_ts = int(updated_at) - 60 * 86400
    telemetry = await db.telemetry(
        system.system_id, start_ts, int(updated_at) + 86400
    )
    if not telemetry:
        return None
    tel = pd.DataFrame(telemetry)
    tel.index = pd.to_datetime(tel.pop("period_start"), unit="s", utc=True)
    tel = tel[
        (tel["period_min"] == 15)
        & (tel["curtailed"] == 0)
        & np.isfinite(tel["mean_watts"])
        & (tel["mean_watts"] >= 0)
    ].copy()
    if tel.empty:
        return None

    cs = simulate_clearsky(tel.index, system, horizon)
    cs_mean = instant_to_period_mean(cs.pac).reindex(tel.index)
    valid = (
        (cs_mean > 0)
        & (cs.elevation.reindex(tel.index) > 5)
        & (tel["mean_watts"] <= 1.05 * cs.paco)
    )
    tel = tel[valid]
    cs_mean = cs_mean[valid]
    if tel.empty:
        return None
    k_actual = tel["mean_watts"] / cs_mean

    model_rows = await db.model_ghi_latest(
        round(system.lat, 3), round(system.lon, 3), start_ts
    )
    if not model_rows:
        return None
    model_frame = pd.DataFrame(model_rows)
    model_frame["target_time"] = pd.to_datetime(
        model_frame["target_time"], unit="s", utc=True
    )
    model_ghi = model_frame.pivot(
        index="target_time", columns="model", values="ghi"
    )
    if any(model not in model_ghi for model in models):
        return None
    model_ghi = model_ghi[models].astype("float64")
    clear_ghi = clearsky_ghi(model_ghi.index, system).replace(0.0, np.nan)
    k_model = model_ghi.div(clear_ghi, axis=0)

    joined = pd.concat(
        [k_actual.rename("actual"), k_model],
        axis=1,
        join="inner",
    ).dropna()
    if len(joined) < MIN_SAMPLES:
        return None
    local_dates = pd.DatetimeIndex(joined.index).tz_convert(system.timezone).date
    if len(set(local_dates)) < MIN_DAYS:
        return None

    x = joined[models].to_numpy(dtype="float64")
    y = joined["actual"].to_numpy(dtype="float64")
    result = fit_weights(x, y, np.asarray(default_weights, dtype="float64"))
    if result is None:
        return None
    weights, learned_mae, default_mae = result
    if learned_mae > GATE_RATIO * default_mae:
        LOG.info(
            "weights rejected at %.3f,%.3f: learned MAE %.4f vs default %.4f",
            system.lat,
            system.lon,
            learned_mae,
            default_mae,
        )
        return None
    learned = {model: float(weight) for model, weight in zip(models, weights)}
    await db.put_weights(
        system.lat,
        system.lon,
        [
            {
                "model": model,
                "weight": learned[model],
                "samples": len(joined),
            }
            for model in models
        ],
        updated_at,
    )
    LOG.info(
        "weights learned at %.3f,%.3f: %d samples, MAE %.4f vs %.4f",
        system.lat,
        system.lon,
        len(joined),
        learned_mae,
        default_mae,
    )
    return learned
