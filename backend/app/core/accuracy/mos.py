"""LightGBM model output statistics on physics residuals."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pvlib

from ..clearsky import clearsky_ghi, simulate_clearsky
from ..database import Database
from ..physics_engine import SimResult, SystemConfig
from ..pvgis_client import Horizon
from ..timeconv import instant_to_period_mean

LOG = logging.getLogger(__name__)

MODEL_TYPE = "mos_lgbm"
MIN_DAYS = 30
HOLDOUT_DAYS = 7
GATE_RATIO = 0.98

FEATURES = [
    "physics_norm",
    "clearsky_norm",
    "kt_fc",
    "cloud_total",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "temp_2m_c",
    "wind_10m_ms",
    "solar_zenith",
    "sin_az",
    "cos_az",
    "lead_hours",
    "doy_sin",
    "doy_cos",
    "days_since_rain",
    "snow_depth_m",
]

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def _days_since_rain(frame: pd.DataFrame, timezone: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    local_dates = pd.DatetimeIndex(frame.index).tz_convert(timezone).date
    daily = frame["precip_mm"].groupby(local_dates).sum()
    since: dict[object, float] = {}
    counter = 7.0
    for day in sorted(daily.index):
        counter = 0.0 if float(daily.loc[day]) > 2.0 else counter + 1.0
        since[day] = counter
    return pd.Series([since[day] for day in local_dates], index=frame.index, dtype="float64")


def build_features(
    weather: pd.DataFrame,
    system: SystemConfig,
    fetched_at: int,
    physics_pac: pd.Series,
    clearsky_pac: pd.Series,
    clear_ghi: pd.Series,
) -> pd.DataFrame:
    index = weather.index
    position = pvlib.solarposition.get_solarposition(
        index, system.lat, system.lon, altitude=system.elevation_m or 0.0
    )
    normalized = system.kwp_total * 1000.0
    features = pd.DataFrame(index=index)
    features["physics_norm"] = physics_pac.reindex(index).fillna(0.0) / normalized
    features["clearsky_norm"] = clearsky_pac.reindex(index).fillna(0.0) / normalized
    ghi_cs = clear_ghi.reindex(index).replace(0.0, np.nan)
    features["kt_fc"] = (weather["ghi"] / ghi_cs).clip(0.0, 2.0).fillna(0.0)
    for column in ("cloud_total", "cloud_low", "cloud_mid", "cloud_high"):
        features[column] = weather[column].ffill().fillna(0.0)
    features["temp_2m_c"] = weather["temp_2m_c"].ffill().fillna(15.0)
    features["wind_10m_ms"] = weather["wind_10m_ms"].ffill().fillna(1.0)
    features["solar_zenith"] = position["apparent_zenith"].to_numpy()
    azimuth_rad = np.deg2rad(position["azimuth"].to_numpy())
    features["sin_az"] = np.sin(azimuth_rad)
    features["cos_az"] = np.cos(azimuth_rad)
    lead = (pd.DatetimeIndex(index).asi8 / 1e9 - fetched_at) / 3600.0
    features["lead_hours"] = lead
    day_of_year = pd.DatetimeIndex(index).dayofyear
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features["days_since_rain"] = _days_since_rain(weather, system.timezone)
    features["snow_depth_m"] = weather["snow_depth_m"].ffill().fillna(0.0)
    return features[FEATURES]


class MosModel:
    def __init__(self, model, metadata: dict) -> None:
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(cls, path: str | Path) -> "MosModel":
        payload = joblib.load(path)
        return cls(payload["model"], payload.get("metadata", {}))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(features[self.model.feature_name_]), dtype="float64")


def apply_hybrid(
    physics: SimResult,
    base: SimResult,
    system: SystemConfig,
    model: MosModel | None,
    features: pd.DataFrame | None,
) -> pd.Series:
    if model is None or features is None:
        return physics.pac.rename("hybrid")
    residual = model.predict(features)
    hybrid = physics.pac.to_numpy() + residual * system.kwp_total * 1000.0
    hybrid = np.clip(hybrid, 0.0, physics.paco)
    hybrid = np.where(base.pac.to_numpy() == 0.0, 0.0, hybrid)
    return pd.Series(hybrid, index=physics.pac.index, name="hybrid")


async def load_model(db: Database, system_id: str) -> MosModel | None:
    record = await db.active_model(system_id)
    if not record:
        return None
    path = Path(record["artifact_path"])
    if not path.exists():
        LOG.warning("active MOS model artifact missing: %s", path)
        return None
    try:
        return await __import__("asyncio").to_thread(MosModel.load, path)
    except Exception as exc:
        LOG.warning("failed to load MOS model %s: %s", path, exc)
        return None


def _nmae(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.mean(np.abs(actual)))
    if denominator <= 0:
        return float("nan")
    return float(np.mean(np.abs(predicted - actual)) / denominator)


def _training_frame(
    db: Database,
    system_row: dict,
    system: SystemConfig,
    horizon: Horizon,
) -> pd.DataFrame:
    import asyncio

    async def _gather() -> pd.DataFrame:
        telemetry = await db.telemetry(system_row["system_id"])
        if not telemetry:
            return pd.DataFrame()
        tel = pd.DataFrame(telemetry)
        tel.index = pd.to_datetime(tel.pop("period_start"), unit="s", utc=True)
        runs = await db.query(
            "SELECT * FROM forecast_runs WHERE system_id=? AND source='physics'", (system_row["system_id"],)
        )
        if not runs:
            return pd.DataFrame()
        runs_df = pd.DataFrame(runs)
        runs_df.index = pd.to_datetime(runs_df.pop("period_start"), unit="s", utc=True)
        frames = []
        for fetched_at, run_group in runs_df.groupby("fetched_at"):
            matching = tel[tel.index.isin(run_group.index)]
            matching = matching[matching["period_min"] == 15]
            if matching.empty or (matching["curtailed"] == 1).all():
                continue
            weather = await db.load_weather(system_row["lat"], system_row["lon"], int(fetched_at))
            if weather is None:
                continue
            run_index = pd.DatetimeIndex(run_group.index)
            cs = simulate_clearsky(run_index, system, horizon)
            physics_pac = pd.Series(
                np.nan, index=run_index, dtype="float64"
            )
            physics_pac.loc[run_group.index] = run_group["mean_watts"].to_numpy()
            physics_pac = physics_pac.interpolate(limit_direction="both")
            features = build_features(
                weather,
                system,
                int(fetched_at),
                physics_pac,
                cs.pac,
                clearsky_ghi(run_index, system),
            )
            features = features.reindex(matching.index).dropna()
            actual = matching["mean_watts"].astype("float64")
            valid = (
                (matching["curtailed"].fillna(0).astype(int) == 0)
                & (actual > 0)
                & np.isfinite(actual)
                & (actual <= 1.05 * cs.paco)
            )
            frame = features[valid.reindex(features.index).fillna(False)].copy()
            frame["actual_w"] = actual.reindex(frame.index)
            frame["physics_w"] = pd.Series(
                run_group["mean_watts"].to_numpy(), index=run_index
            ).reindex(frame.index)
            frame["fetched_at"] = int(fetched_at)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    return asyncio.run(_gather())


def train(
    db: Database,
    system_row: dict,
    system: SystemConfig,
    horizon: Horizon,
    models_dir: Path,
) -> dict | None:
    """Train, gate and register a new MOS model. Returns the registry record."""
    rows = _training_frame(db, system_row, system, horizon)
    if rows.empty:
        LOG.info("%s: no MOS training rows", system.kwp_total and system_row["system_id"])
        return None
    tz = system.timezone
    days = pd.Series(rows.index.tz_convert(tz).date, index=rows.index)
    distinct_days = sorted(days.unique())
    if len(distinct_days) < MIN_DAYS:
        LOG.info("%s: %d days < %d, MOS pass-through", system_row["system_id"], len(distinct_days), MIN_DAYS)
        return None
    holdout_days = set(distinct_days[-HOLDOUT_DAYS:])
    is_holdout = days.isin(holdout_days)
    train = rows[~is_holdout]
    holdout = rows[is_holdout]
    if holdout.empty or train.empty:
        return None

    target = (
        (rows["actual_w"] - rows["physics_w"]) / (system.kwp_total * 1000.0)
    ).astype("float64")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[FEATURES], target[train.index])
    residual = model.predict(holdout[FEATURES])
    hybrid_w = np.clip(holdout["physics_w"].to_numpy() + residual * system.kwp_total * 1000.0, 0, None)
    nmae_physics = _nmae(holdout["actual_w"].to_numpy(), holdout["physics_w"].to_numpy())
    nmae_hybrid = _nmae(holdout["actual_w"].to_numpy(), hybrid_w)

    active = db_query_active_sync(db, system_row["system_id"])
    gate_passes = np.isfinite(nmae_hybrid) and nmae_hybrid <= GATE_RATIO * nmae_physics

    if not gate_passes:
        if active:
            previous = MosModel.load(active["artifact_path"])
            previous_hybrid = np.clip(
                holdout["physics_w"].to_numpy()
                + previous.predict(holdout[FEATURES]) * system.kwp_total * 1000.0,
                0,
                None,
            )
            prev_nmae_hybrid = _nmae(holdout["actual_w"].to_numpy(), previous_hybrid)
            if np.isfinite(prev_nmae_hybrid) and prev_nmae_hybrid <= GATE_RATIO * nmae_physics:
                LOG.info("%s: gate rejected new MOS, keeping previous", system_row["system_id"])
                return None
        db_deactivate_sync(db, system_row["system_id"])
        LOG.info("%s: gate rejected MOS and previous failed; deactivated", system_row["system_id"])
        return None

    model.fit(rows[FEATURES], target)
    trained_at = int(time.time())
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = models_dir / f"{system_row['system_id']}_mos_{trained_at}.joblib"
    joblib.dump(
        {"model": model, "metadata": {"features": FEATURES, "trained_at": trained_at}},
        artifact,
    )
    record = {
        "system_id": system_row["system_id"],
        "model_type": MODEL_TYPE,
        "trained_at": trained_at,
        "artifact_path": str(artifact),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "holdout_nmae_physics": nmae_physics,
        "holdout_nmae_hybrid": nmae_hybrid,
        "active": True,
    }
    db_register_sync(db, record)
    LOG.info(
        "%s: MOS activated nMAE physics=%.3f hybrid=%.3f", system_row["system_id"], nmae_physics, nmae_hybrid
    )
    return record


def db_query_active_sync(db: Database, system_id: str) -> dict | None:
    rows = db._query(
        "SELECT * FROM model_registry WHERE system_id=? AND active=1 ORDER BY trained_at DESC LIMIT 1",
        (system_id,),
    )
    return dict(rows[0]) if rows else None


def db_deactivate_sync(db: Database, system_id: str) -> None:
    db._execute(
        "UPDATE model_registry SET active=0 WHERE system_id=? AND model_type=?",
        (system_id, MODEL_TYPE),
    )


def db_register_sync(db: Database, record: dict) -> int:
    import sqlite3

    conn = db.conn
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "UPDATE model_registry SET active=0 WHERE system_id=? AND model_type=?",
            (record["system_id"], record["model_type"]),
        )
        cur = conn.execute(
            "INSERT INTO model_registry (system_id,model_type,trained_at,artifact_path,"
            "train_rows,holdout_rows,holdout_nmae_physics,holdout_nmae_hybrid,active) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                record["system_id"],
                record["model_type"],
                record["trained_at"],
                record["artifact_path"],
                record["train_rows"],
                record["holdout_rows"],
                record["holdout_nmae_physics"],
                record["holdout_nmae_hybrid"],
                1,
            ),
        )
        conn.execute("COMMIT")
        return int(cur.lastrowid)
    except Exception:
        conn.execute("ROLLBACK")
        raise
