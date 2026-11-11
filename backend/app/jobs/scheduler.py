"""APScheduler wiring. Run uvicorn with exactly one worker."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import Settings

from ..core import benchmark as benchmark_module
from ..core.accuracy import calibration, mos, shading
from ..core.clearsky import simulate_clearsky
from ..core.database import Database
from ..core.pipeline import ForecastPipeline, system_config
from ..core.timeconv import UTC

LOG = logging.getLogger(__name__)

_PROCESS_POOL: ProcessPoolExecutor | None = None


def _get_process_pool() -> ProcessPoolExecutor:
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=1)
    return _PROCESS_POOL


def _train_mos_process(db_path: str, system_row: dict, system, horizon, models_dir) -> dict | None:
    database = Database(db_path)
    asyncio.run(database.initialize())
    return mos.train(database, system_row, system, horizon, models_dir)


def resolve_scheduler_tz(settings: Settings, systems: list[dict]) -> str:
    if settings.scheduler_tz:
        return settings.scheduler_tz
    env_tz = os.environ.get("TZ")
    if env_tz:
        return env_tz
    if systems:
        return systems[0]["timezone"]
    return "UTC"


async def retrain_system(db: Database, pipeline: ForecastPipeline, system_row: dict) -> None:
    system = system_config(system_row)
    now = int(dt.datetime.now(UTC).timestamp())
    telemetry = await db.telemetry(system_row["system_id"], now - 60 * 86400, now + 86400)
    if not telemetry:
        LOG.info("retrain %s: no telemetry", system.system_id)
        return
    import pandas as pd

    tel = pd.DataFrame(telemetry)
    tel.index = pd.to_datetime(tel.pop("period_start"), unit="s", utc=True)
    weather, _ = await pipeline.ensure_weather(system.lat, system.lon)
    horizon = await pipeline.ensure_horizon(system.lat, system.lon)
    clear = simulate_clearsky(tel.index, system, horizon)
    previous = await db.get_calibration(system.system_id)
    previous_c = float(previous["factor_c"]) if previous else 1.0
    result = await asyncio.to_thread(
        calibration.learn,
        tel.reset_index().rename(columns={"index": "period_start"}),
        clear,
        system,
        system.timezone,
        previous_c,
        now,
    )
    await db.put_calibration(
        system.system_id,
        result["factor_c"],
        result["clear_days"],
        result["samples"],
        result["updated_at"],
    )
    LOG.info("retrain %s: C=%.4f (%d clear days)", system.system_id, result["factor_c"], result["clear_days"])
    shade_rows = await asyncio.to_thread(
        shading.learn,
        tel.reset_index().rename(columns={"index": "period_start"}),
        clear,
        result["factor_c"],
        system,
        system.timezone,
        now,
        result["clear_days"],
        set(result.get("clear_dates", [])),
    )
    await db.replace_shading(system.system_id, shade_rows)
    LOG.info("retrain %s: %d shading buckets", system.system_id, len(shade_rows))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        _get_process_pool(),
        _train_mos_process,
        db.path,
        system_row,
        system,
        horizon,
        pipeline.settings.models_dir,
    )
    pipeline.invalidate()


async def nightly_retrain(db: Database, pipeline: ForecastPipeline) -> None:
    for row in await db.list_systems():
        try:
            await retrain_system(db, pipeline, row)
        except Exception:
            LOG.exception("retrain failed for %s", row["system_id"])


async def housekeeping(db: Database) -> None:
    now = int(dt.datetime.now(UTC).timestamp())
    deleted = await db.housekeeping(now)
    LOG.info("housekeeping deleted %s", deleted)
    if dt.datetime.now(UTC).day <= 7:
        await db.vacuum()


def create_scheduler(
    settings: Settings,
    db: Database,
    pipeline: ForecastPipeline,
    timezone: str = "UTC",
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(
        timezone=timezone,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 900},
    )
    scheduler.add_job(
        pipeline.refresh_weather,
        IntervalTrigger(minutes=60),
        id="weather_refresh",
        next_run_time=dt.datetime.now(ZoneInfo(timezone)),
    )
    scheduler.add_job(
        pipeline.snapshot_all,
        CronTrigger(hour="5,11,17", minute=0, timezone=timezone),
        id="snapshot_runs",
    )
    scheduler.add_job(
        pipeline.snapshot_nowcast,
        CronTrigger(minute=5, timezone=timezone),
        id="nowcast_snapshot",
    )
    scheduler.add_job(
        nightly_retrain,
        CronTrigger(hour=1, minute=30, timezone=timezone),
        id="nightly_retrain",
        args=[db, pipeline],
    )
    scheduler.add_job(
        housekeeping,
        CronTrigger(day_of_week="sun", hour=3, minute=0, timezone=timezone),
        id="housekeeping",
        args=[db],
    )
    return scheduler
