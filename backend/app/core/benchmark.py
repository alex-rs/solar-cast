"""Forecast accuracy benchmark on a common hourly sample set."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib

from .database import Database
from .timeconv import UTC

BUCKETS = {
    "intraday": (0.0, 12.0),
    "day_ahead": (12.0, 36.0),
    "d2_3": (36.0, 84.0),
    "d4_7": (84.0, 168.0),
}
SOURCES = ["forecast_solar", "physics_base", "physics", "hybrid", "final"]
METRIC_KEYS = [
    "nMAE_pct",
    "nRMSE_pct",
    "nMAE_kwp_pct",
    "MBE_wh",
    "skill_vs_forecast_solar",
    "skill_vs_physics_base",
    "daily_energy_mape_pct",
    "n_samples",
]


def _empty_metrics() -> dict:
    return {key: None for key in METRIC_KEYS} | {"n_samples": 0}


def metrics(errors: np.ndarray, actual: np.ndarray, kwp_total: float) -> dict:
    errors = np.asarray(errors, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    n = len(errors)
    if n == 0:
        return _empty_metrics()
    mean_actual = float(np.mean(actual))
    nmae = 100.0 * float(np.mean(np.abs(errors))) / mean_actual if mean_actual > 0 else None
    nrmse = (
        100.0 * float(np.sqrt(np.mean(errors**2))) / mean_actual if mean_actual > 0 else None
    )
    return {
        "nMAE_pct": nmae,
        "nRMSE_pct": nrmse,
        "nMAE_kwp_pct": 100.0 * float(np.mean(np.abs(errors))) / (kwp_total * 1000.0)
        if kwp_total > 0
        else None,
        "MBE_wh": float(np.mean(errors)),
        "skill_vs_forecast_solar": None,
        "skill_vs_physics_base": None,
        "daily_energy_mape_pct": None,
        "n_samples": int(n),
    }


def assign_bucket(lead_hours: float) -> str | None:
    for name, (low, high) in BUCKETS.items():
        if low <= lead_hours < high:
            return name
    return None


def _hourly_actual(telemetry: list[dict], timezone: str) -> pd.DataFrame:
    if not telemetry:
        return pd.DataFrame(columns=["actual_wh", "complete"])
    frame = pd.DataFrame(telemetry)
    frame.index = pd.to_datetime(frame.pop("period_start"), unit="s", utc=True)
    rows = []
    for hour_start, group in frame.groupby(frame.index.floor("h")):
        curtailed = bool(group["curtailed"].any())
        if len(group) == 1 and int(group["period_min"].iloc[0]) == 60:
            complete = True
            wh = float(group["energy_wh"].iloc[0])
        else:
            quarters = set((group.index - hour_start).total_seconds().astype(int) // 60)
            complete = {0, 15, 30, 45}.issubset(quarters) and len(group) >= 4
            wh = float(group["energy_wh"].sum())
        rows.append(
            {
                "hour_start": hour_start,
                "actual_wh": wh,
                "complete": complete and not curtailed,
            }
        )
    return pd.DataFrame(rows).set_index("hour_start")


def _hourly_runs(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["energy_wh", "fetched_at", "lead_hours"])
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame.pop("period_start"), unit="s", utc=True)
    aggregated = []
    for hour_start, group in frame.groupby(frame.index.floor("h")):
        lead = float(group["lead_hours"].min())
        aggregated.append(
            {
                "hour_start": hour_start,
                "energy_wh": float(group["energy_wh"].sum()),
                "fetched_at": int(group["fetched_at"].max()),
                "lead_hours": lead,
            }
        )
    return pd.DataFrame(aggregated).set_index("hour_start")


def _daylight_hours(system_row: dict, timerange: pd.DatetimeIndex, timezone: str) -> set[pd.Timestamp]:
    if len(timerange) == 0:
        return set()
    position = pvlib.solarposition.get_solarposition(
        timerange + pd.Timedelta(minutes=30),
        system_row["lat"],
        system_row["lon"],
        altitude=system_row.get("elevation_m") or 0.0,
    )
    return set(timerange[position["apparent_elevation"].to_numpy() > 0])


def _skill(source_rmse: float, reference_rmse: float) -> float | None:
    if reference_rmse is None or reference_rmse <= 0 or not math.isfinite(reference_rmse):
        return None
    return 1.0 - source_rmse / reference_rmse


async def benchmark(db: Database, system_row: dict, days: int, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(UTC)
    end = now
    start = now - dt.timedelta(days=days)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    system_id = system_row["system_id"]
    kwp_total = sum(float(p["kwp"]) for p in system_row["planes"])
    timezone = system_row["timezone"]

    telemetry = await db.telemetry(system_id, start_ts, end_ts + 86400)
    actual = _hourly_actual(telemetry, timezone)
    if actual.empty:
        return _response(system_id, days, kwp_total, {})

    runs_by_source: dict[str, pd.DataFrame] = {}
    for source in SOURCES:
        rows = await db.query(
            "SELECT fetched_at,period_start,period_min,lead_hours,energy_wh FROM forecast_runs "
            "WHERE system_id=? AND source=? AND period_start>=? AND period_start<?",
            (system_id, source, start_ts, end_ts + 86400),
        )
        runs_by_source[source] = _hourly_runs(rows)

    actual_hours = set(actual.index[actual["complete"]])
    solar_hours = _daylight_hours(system_row, pd.DatetimeIndex(sorted(actual_hours)), timezone)
    valid_hours = actual_hours & solar_hours

    all_hours = pd.date_range(
        pd.Timestamp(start).floor("h"), pd.Timestamp(end).ceil("h"), freq="h", tz="UTC", inclusive="left"
    )
    daylight_all = _daylight_hours(system_row, all_hours, timezone)
    daylight_by_day = pd.Series(list(daylight_all)).groupby(
        [pd.Timestamp(h).tz_convert(timezone).date() for h in daylight_all]
    ).size().to_dict()
    if not valid_hours:
        return _response(system_id, days, kwp_total, {})

    result_buckets: dict[str, dict] = {}
    for bucket, (low, high) in BUCKETS.items():
        selections: dict[str, pd.Series] = {}
        for source in SOURCES:
            frame = runs_by_source[source]
            if frame.empty:
                continue
            mask = (frame["lead_hours"] >= low) & (frame["lead_hours"] < high)
            usable = frame[mask]
            if usable.empty:
                continue
            # Most recent run per hour within the bucket.
            usable = usable.sort_values(["fetched_at"])
            latest = usable.groupby(level=0).tail(1)
            selections[source] = latest["energy_wh"]
        if not selections:
            result_buckets[bucket] = {"n_samples": 0, "sources": {}}
            continue
        common = set(valid_hours)
        for series in selections.values():
            common &= set(series.index)
        common = sorted(common)
        if not common:
            result_buckets[bucket] = {"n_samples": 0, "sources": {}}
            continue
        common_index = pd.DatetimeIndex(common)
        actual_wh = actual.reindex(common_index)["actual_wh"].to_numpy(dtype="float64")
        sources_metrics = {}
        errors_by_source = {}
        for source, series in selections.items():
            predicted = series.reindex(common_index).to_numpy(dtype="float64")
            errors = predicted - actual_wh
            errors_by_source[source] = errors
            source_metrics = metrics(errors, actual_wh, kwp_total)
            sources_metrics[source] = source_metrics
        fs_rmse = (
            float(np.sqrt(np.mean(errors_by_source["forecast_solar"] ** 2)))
            if "forecast_solar" in errors_by_source
            else None
        )
        physics_rmse = (
            float(np.sqrt(np.mean(errors_by_source["physics_base"] ** 2)))
            if "physics_base" in errors_by_source
            else None
        )
        for source, source_metrics in sources_metrics.items():
            rmse = float(np.sqrt(np.mean(errors_by_source[source] ** 2)))
            source_metrics["skill_vs_forecast_solar"] = (
                _skill(rmse, fs_rmse) if source != "forecast_solar" else None
            )
            source_metrics["skill_vs_physics_base"] = (
                _skill(rmse, physics_rmse) if source != "physics_base" else None
            )
            local_dates = common_index.tz_convert(timezone).date
            daily = pd.DataFrame(
                {"actual": actual_wh, "predicted": selections[source].reindex(common_index).to_numpy()},
                index=pd.Index(local_dates, name="date"),
            ).groupby(level=0).sum()
            counts = pd.Series(local_dates).value_counts()
            complete_days = [
                day for day in daily.index if counts[day] == daylight_by_day.get(day, 0)
            ]
            if complete_days:
                subset = daily.loc[complete_days]
                denominator = subset["actual"].replace(0.0, np.nan)
                source_metrics["daily_energy_mape_pct"] = float(
                    100.0 * np.nanmean(np.abs(subset["predicted"] - subset["actual"]) / denominator)
                )
        result_buckets[bucket] = {
            "n_samples": len(common),
            "sources": sources_metrics,
        }
    return _response(system_id, days, kwp_total, result_buckets)


def _response(system_id: str, days: int, kwp_total: float, buckets: dict) -> dict:
    full = {}
    for bucket in BUCKETS:
        if bucket in buckets:
            full[bucket] = buckets[bucket]
        else:
            full[bucket] = {"n_samples": 0, "sources": {}}
    return {
        "system_id": system_id,
        "window_days": days,
        "kwp_total": kwp_total,
        "buckets": full,
    }
