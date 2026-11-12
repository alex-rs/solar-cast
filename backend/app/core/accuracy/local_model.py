"""LightGBM forecast from local telemetry only, used when remote weather is gone.

The model predicts AC power at fixed lead times from what the installation already
knows: recent power telemetry plus the clear-sky simulation. No weather data, no
network, no new sensors. Research background and measured cross-site skill live in
``research/local_sensors/RESULTS.md``; this module is the production form of that
experiment, with the physics engine providing the clear-sky reference.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..clearsky import simulate_clearsky
from ..database import Database
from ..physics_engine import SimResult, SystemConfig
from ..pvgis_client import Horizon

LOG = logging.getLogger(__name__)

MODEL_TYPE = "local_lgbm"
STEP_MIN = 15
HORIZONS_MIN: tuple[int, ...] = (15, 30, 45, 60, 90, 120, 180, 240, 300, 360)
MIN_DAYS = 14
HOLDOUT_DAYS = 7
GATE_RATIO = 0.95
DECISION_HORIZON = 60
K_LIMIT = 1.5
CLEAR_FLOOR = 0.02  # normalized clear-sky power below this counts as night
MAX_ANCHOR_AGE_MIN = 180.0
CLIM_CLIP = (0.05, 1.2)
WARMUP_STEPS = 4  # 60 minutes of history needed for the rolling features

FEATURES = [
    "zenith_sin",
    "zenith_cos",
    "azimuth_sin",
    "azimuth_cos",
    "cs_now",
    "cs_h",
    "p_now",
    "p_r30",
    "p_r60",
    "p_s60",
    "p_d15",
    "p_d60",
    "k_now",
    "k_r60",
    "clim_k",
    "doy_sin",
    "doy_cos",
    "tod_sin",
    "tod_cos",
]
ANCHOR_FEATURES = [name for name in FEATURES if name != "cs_h"]

LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=12,
    max_depth=5,
    min_child_samples=50,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


@dataclass
class LocalForecast:
    models: dict[int, lgb.LGBMRegressor]
    p_max: float
    clim_table: pd.Series
    clim_default: float
    trained_at: int
    holdout_skill: dict[str, float]

    @property
    def horizon_min(self) -> int:
        """Furthest lead time the gate kept."""
        return max(self.models) if self.models else 0


# -- telemetry ---------------------------------------------------------------


def telemetry_series(rows: list[dict]) -> pd.Series:
    """Resample stored telemetry onto the 15-minute grid, dropping curtailed rows."""
    usable = [row for row in rows if not row.get("curtailed")]
    if not usable:
        return pd.Series(dtype="float64")
    frame = pd.DataFrame(usable)
    frame.index = pd.to_datetime(frame["period_start"], unit="s", utc=True)
    series = frame["mean_watts"].astype("float64")
    series = series.resample(f"{STEP_MIN}min", label="right", closed="right").mean()
    return series.dropna()


# -- clear-sky index and climatology ----------------------------------------


def clear_norm(clear: SimResult, p_max: float) -> pd.Series:
    return (clear.pac / p_max).clip(lower=0.0)


def clear_sky_index(actual_norm: pd.Series, cs_norm: pd.Series) -> pd.Series:
    ratio = actual_norm / cs_norm.where(cs_norm > CLEAR_FLOOR)
    return ratio.clip(0.0, K_LIMIT)


def _slots(index: pd.DatetimeIndex) -> pd.MultiIndex:
    minutes = index.hour * 60 + index.minute
    return pd.MultiIndex.from_arrays([index.month, minutes // STEP_MIN])


def climatology(k: pd.Series) -> tuple[pd.Series, float]:
    """Median clear-sky index per (month, 15-minute slot), with a scalar fallback."""
    usable = k.dropna()
    if usable.empty:
        return pd.Series(dtype="float64"), 0.0
    table = usable.groupby(_slots(usable.index)).median()
    return table.clip(*CLIM_CLIP), float(usable.median())


def climatology_series(
    index: pd.DatetimeIndex, table: pd.Series, fallback: float
) -> pd.Series:
    if len(table) == 0:
        return pd.Series(fallback, index=index)
    values = table.reindex(_slots(index)).to_numpy(dtype="float64")
    values = np.where(np.isfinite(values), values, fallback)
    return pd.Series(values, index=index)


# -- features ----------------------------------------------------------------


def base_features(
    actual_norm: pd.Series,
    cs_norm: pd.Series,
    clim_k: pd.Series,
    clear: SimResult,
) -> pd.DataFrame:
    """Features that depend only on the anchor time; ``cs_h`` is added per horizon."""
    index = actual_norm.index
    k = clear_sky_index(actual_norm, cs_norm)
    zenith = np.deg2rad(clear.zenith.reindex(index).to_numpy(dtype="float64"))
    azimuth = np.deg2rad(clear.azimuth.reindex(index).to_numpy(dtype="float64"))
    day_of_year = index.day_of_year.to_numpy(dtype="float64")
    minute_of_day = (index.hour * 60 + index.minute).to_numpy(dtype="float64")
    return pd.DataFrame(
        {
            "zenith_sin": np.sin(zenith),
            "zenith_cos": np.cos(zenith),
            "azimuth_sin": np.sin(azimuth),
            "azimuth_cos": np.cos(azimuth),
            "cs_now": cs_norm.reindex(index).to_numpy(dtype="float64"),
            "p_now": actual_norm.to_numpy(dtype="float64"),
            "p_r30": actual_norm.rolling(2, min_periods=1).mean().to_numpy(),
            "p_r60": actual_norm.rolling(4, min_periods=1).mean().to_numpy(),
            "p_s60": actual_norm.rolling(4, min_periods=2).std().to_numpy(),
            "p_d15": (actual_norm - actual_norm.shift(1)).to_numpy(),
            "p_d60": (actual_norm - actual_norm.shift(WARMUP_STEPS)).to_numpy(),
            "k_now": k.to_numpy(),
            "k_r60": k.rolling(4, min_periods=1).mean().to_numpy(),
            "clim_k": clim_k.reindex(index).to_numpy(dtype="float64"),
            "doy_sin": np.sin(2 * np.pi * (day_of_year - 1) / 365.25),
            "doy_cos": np.cos(2 * np.pi * (day_of_year - 1) / 365.25),
            "tod_sin": np.sin(2 * np.pi * minute_of_day / 1440.0),
            "tod_cos": np.cos(2 * np.pi * minute_of_day / 1440.0),
        },
        index=index,
    )[ANCHOR_FEATURES]


# -- training ----------------------------------------------------------------


def horizon_matrix(
    base: pd.DataFrame,
    actual_norm: pd.Series,
    cs_norm: pd.Series,
    steps: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target for ``steps`` ahead: ``y(t+h)`` from anchor features at ``t``."""
    matrix = base.copy()
    matrix["cs_h"] = cs_norm.shift(-steps).reindex(base.index).to_numpy()
    target = actual_norm.shift(-steps).reindex(base.index).rename("y")
    usable = target.notna() & (matrix["cs_now"] > CLEAR_FLOOR) & matrix["cs_h"].notna()
    return matrix.loc[usable, FEATURES], target.loc[usable]


def _nmae(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.mean(np.abs(actual)))
    if denominator <= 0 or not np.isfinite(denominator):
        return float("inf")
    return float(np.mean(np.abs(actual - predicted)) / denominator)


async def train(
    db: Database,
    system_row: dict,
    system: SystemConfig,
    horizon: Horizon,
    models_dir: Path,
    *,
    train_days: int = 180,
    min_days: int = MIN_DAYS,
) -> dict | None:
    """Train, gate and register a local model. Returns the registry record or ``None``."""
    system_id = system_row["system_id"]
    now = int(time.time())
    rows = await db.telemetry(system_id, now - train_days * 86400, now + 3600)
    watts = telemetry_series(rows)
    if watts.empty:
        LOG.info("%s: local model skipped, no telemetry", system_id)
        return None
    days = int(watts.index.normalize().nunique())
    if days < min_days:
        LOG.info("%s: local model skipped, %d/%d days", system_id, days, min_days)
        return None

    p_max = float(np.percentile(watts.to_numpy(), 99.9))
    if not np.isfinite(p_max) or p_max <= 0:
        p_max = float(system.kwp_total) * 1000.0
    if p_max <= 0:
        LOG.info("%s: local model skipped, no power scale", system_id)
        return None

    clear = simulate_clearsky(watts.index, system, horizon)
    cs = clear_norm(clear, p_max)
    actual = watts / p_max

    holdout_start = watts.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train_mask = pd.Series(watts.index < holdout_start, index=watts.index)
    holdout_mask = pd.Series(~train_mask, index=watts.index)
    if int(holdout_mask.sum()) < 50 or int(train_mask.sum()) < 200:
        LOG.info("%s: local model skipped, not enough rows to split", system_id)
        return None

    table, fallback = climatology(clear_sky_index(actual[train_mask], cs[train_mask]))
    if len(table) == 0:
        LOG.info("%s: local model skipped, empty climatology", system_id)
        return None
    clim_all = climatology_series(watts.index, table, fallback)
    base = base_features(actual, cs, clim_all, clear)

    models: dict[int, lgb.LGBMRegressor] = {}
    accepted_model: list[float] = []
    accepted_baseline: list[float] = []
    rejected: list[int] = []
    for horizon_min in HORIZONS_MIN:
        steps = horizon_min // STEP_MIN
        matrix, target = horizon_matrix(base, actual, cs, steps)
        if matrix.empty:
            continue
        train_rows = train_mask.reindex(matrix.index).fillna(False).to_numpy()
        hold_rows = holdout_mask.reindex(matrix.index).fillna(False).to_numpy()
        if int(train_rows.sum()) < 200 or int(hold_rows.sum()) < 30:
            continue
        estimator = lgb.LGBMRegressor(**LGBM_PARAMS)
        estimator.fit(matrix[train_rows], target[train_rows])
        predicted = np.clip(estimator.predict(matrix[hold_rows]), 0.0, None)
        truth = target[hold_rows].to_numpy()
        baseline = (cs.shift(-steps) * clim_all).reindex(matrix.index[hold_rows]).to_numpy()
        nmae = _nmae(truth, predicted)
        nmae_base = _nmae(truth, np.nan_to_num(baseline, nan=0.0))
        # Publish only the lead times where the model actually beats the trivial
        # clear-sky x climatology curve; a horizon that loses is worse than useless.
        if np.isfinite(nmae) and nmae <= GATE_RATIO * nmae_base:
            models[horizon_min] = estimator
            accepted_model.append(nmae)
            accepted_baseline.append(nmae_base)
        else:
            rejected.append(horizon_min)

    # The 60-minute horizon is the product promise; require it plus two more buckets.
    if len(models) < 3 or DECISION_HORIZON not in models:
        await db.deactivate_models(system_id, MODEL_TYPE)
        LOG.info(
            "%s: local model rejected, only %s horizons passed (%s)",
            system_id,
            sorted(models),
            rejected,
        )
        return None

    model_score = float(np.mean(accepted_model))
    baseline_score = float(np.mean(accepted_baseline))

    trained_at = int(time.time())
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = models_dir / f"{system_id}_local_{trained_at}.joblib"
    payload = LocalForecast(
        models=models,
        p_max=p_max,
        clim_table=table,
        clim_default=fallback,
        trained_at=trained_at,
        holdout_skill={
            "nmae_model": model_score,
            "nmae_baseline": baseline_score,
            "skill": float(1.0 - model_score / baseline_score) if baseline_score else 0.0,
        },
    )
    await asyncio.to_thread(joblib.dump, payload, artifact)
    record = {
        "system_id": system_id,
        "model_type": MODEL_TYPE,
        "trained_at": trained_at,
        "artifact_path": str(artifact),
        "train_rows": int(train_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "holdout_nmae_physics": baseline_score,
        "holdout_nmae_hybrid": model_score,
        "active": True,
    }
    await db.register_model(record)
    LOG.info(
        "%s: local model active nMAE=%.3f baseline=%.3f over %d horizons",
        system_id,
        model_score,
        baseline_score,
        len(models),
    )
    return record


# -- serving -----------------------------------------------------------------


def load_sync(artifact_path: str) -> LocalForecast | None:
    try:
        payload = joblib.load(artifact_path)
    except (OSError, ValueError, EOFError) as exc:
        LOG.warning("local model artifact unreadable: %s", exc)
        return None
    if not isinstance(payload, LocalForecast):
        LOG.warning("local model artifact has unexpected type: %s", type(payload))
        return None
    return payload


async def load(db: Database, system_id: str) -> LocalForecast | None:
    record = await db.active_model(system_id, MODEL_TYPE)
    if record is None:
        return None
    return await asyncio.to_thread(load_sync, record["artifact_path"])


def anchor_features(
    history: pd.Series,
    clear: SimResult,
    model: LocalForecast,
    anchor: pd.Timestamp,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Feature row at ``anchor``; ``None`` when telemetry is missing or too old."""
    reference = now if now is not None else pd.Timestamp.now(tz="UTC")
    if (reference - anchor) > pd.Timedelta(minutes=MAX_ANCHOR_AGE_MIN):
        return None
    window = history[history.index <= anchor].tail(16)
    if window.empty:
        return None
    if (anchor - window.index.max()) > pd.Timedelta(minutes=MAX_ANCHOR_AGE_MIN):
        return None
    start = window.index.min() - pd.Timedelta(minutes=STEP_MIN * WARMUP_STEPS)
    grid = pd.date_range(start, anchor, freq=f"{STEP_MIN}min")
    actual = (window / model.p_max).reindex(grid).astype("float64")
    cs = clear_norm(clear, model.p_max).reindex(grid)
    clim = climatology_series(grid, model.clim_table, model.clim_default)
    frame = base_features(actual, cs, clim, clear)
    return frame.loc[[anchor]] if anchor in frame.index else None


def predict_power(
    model: LocalForecast,
    clear: SimResult,
    features: pd.DataFrame,
    anchor: pd.Timestamp,
    anchor_power: float,
    paco: float,
) -> pd.Series:
    """Predicted AC power from ``anchor`` to the last modelled horizon."""
    index = clear.pac.index
    cs = clear_norm(clear, model.p_max)
    points: dict[pd.Timestamp, float] = {anchor: float(anchor_power)}
    for horizon_min, estimator in model.models.items():
        target = anchor + pd.Timedelta(minutes=horizon_min)
        if target not in index:
            continue
        row = features.copy()
        row["cs_h"] = float(cs.get(target, 0.0))
        points[target] = float(np.clip(estimator.predict(row[FEATURES])[0], 0.0, None)) * model.p_max
    curve = pd.Series(points).sort_index()
    target_index = index[index >= anchor]
    aligned = curve.reindex(curve.index.union(target_index)).interpolate(method="time")
    head = aligned.reindex(target_index).ffill().bfill()
    return head.clip(lower=0.0, upper=paco)


def with_tail(model: LocalForecast, clear: SimResult, head: pd.Series) -> pd.Series:
    """Beyond the modelled horizon keep the seasonal clear-sky shape."""
    index = clear.pac.index
    clim = climatology_series(index, model.clim_table, model.clim_default)
    tail = (clear.pac * clim).clip(lower=0.0)
    combined = head.reindex(index).combine_first(tail)
    return combined.clip(lower=0.0, upper=clear.paco)


def climatology_power(clear: SimResult, ratio: float) -> pd.Series:
    """Cold-start fallback: clear-sky shape scaled by the recent clear-sky index."""
    return (clear.pac * float(ratio)).clip(lower=0.0, upper=clear.paco)
