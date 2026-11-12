from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_db, get_pipeline, require_auth
from app.core.accuracy import local_model, mos
from app.core.database import Database
from app.core.pipeline import ForecastPipeline
from app.core.timeconv import ts_to_dt

router = APIRouter(tags=["health"], dependencies=[Depends(require_auth)])


@router.get("/api/v1/health")
async def health(
    request: Request,
    db: Database = Depends(get_db),
    pipeline: ForecastPipeline = Depends(get_pipeline),
):
    database = "ok"
    try:
        await db.query("SELECT 1")
    except Exception as exc:  # pragma: no cover - hard failure path
        database = f"error: {exc}"
    scheduler = getattr(request.app.state, "scheduler", None)
    jobs = []
    if scheduler is not None:
        for job in scheduler.get_jobs():
            next_run = getattr(job, "next_run_time", None)
            jobs.append(
                {
                    "id": job.id,
                    "next_run": next_run.isoformat() if next_run else None,
                }
            )
    systems = []
    for system in await db.list_systems():
        calibration = await db.get_calibration(system["system_id"])
        shading_rows = await db.get_shading(system["system_id"])
        model = await db.active_model(system["system_id"], mos.MODEL_TYPE)
        local = await db.active_model(system["system_id"], local_model.MODEL_TYPE)
        last = await db.last_telemetry(system["system_id"])
        weather_ts = await db.latest_weather_fetched_at(system["lat"], system["lon"])
        systems.append(
            {
                "system_id": system["system_id"],
                "weather_fetched_at": ts_to_dt(weather_ts).isoformat() if weather_ts else None,
                "last_telemetry": ts_to_dt(last["period_start"]).isoformat() if last else None,
                "layers": {
                    "calibration": bool(calibration),
                    "shading": bool(shading_rows),
                    "mos": bool(model),
                    "local": bool(local),
                },
                "local_model": (
                    {
                        "trained_at": ts_to_dt(local["trained_at"]).isoformat(),
                        "holdout_nmae_baseline": local["holdout_nmae_physics"],
                        "holdout_nmae_local": local["holdout_nmae_hybrid"],
                    }
                    if local
                    else None
                ),
                "active_model": (
                    {
                        "trained_at": ts_to_dt(model["trained_at"]).isoformat(),
                        "holdout_nmae_physics": model["holdout_nmae_physics"],
                        "holdout_nmae_hybrid": model["holdout_nmae_hybrid"],
                    }
                    if model
                    else None
                ),
            }
        )
    return {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "scheduler_timezone": getattr(request.app.state, "scheduler_tz", None),
        "jobs": jobs,
        "systems": systems,
    }
