from __future__ import annotations

import re

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.deps import get_db, get_pipeline, require_auth
from app.core.database import Database
from app.core.pipeline import ForecastPipeline
from app.schemas.native import SystemIn, SystemOut

router = APIRouter(prefix="/api/v1/systems", tags=["systems"], dependencies=[Depends(require_auth)])

SLUG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _to_out(system: dict) -> SystemOut:
    payload = dict(system)
    payload["planes"] = [
        {
            "plane_id": plane["plane_id"],
            "ordinal": plane["ordinal"],
            "dec": plane["dec"],
            "az": plane["az_fs"],
            "kwp": plane["kwp"],
        }
        for plane in system["planes"]
    ]
    return SystemOut.model_validate(payload)


@router.put("/{system_id}", response_model=SystemOut)
async def put_system(
    system_id: str,
    payload: SystemIn,
    db: Database = Depends(get_db),
    pipeline: ForecastPipeline = Depends(get_pipeline),
) -> SystemOut:
    if not SLUG.match(system_id):
        raise HTTPException(status_code=422, detail="system_id must be a slug")
    timezone = payload.timezone or pipeline.timezone_for(payload.lat, payload.lon)
    system = {
        "system_id": system_id,
        "name": payload.name,
        "lat": payload.lat,
        "lon": payload.lon,
        "elevation_m": payload.elevation_m,
        "timezone": timezone,
        "inverter_kw": payload.inverter_kw,
        "system_loss": payload.system_loss,
        "gamma_pdc": payload.gamma_pdc,
        "damping_morning": payload.damping_morning,
        "damping_evening": payload.damping_evening,
    }
    planes = [
        {"dec": plane.dec, "az_fs": plane.az, "kwp": plane.kwp} for plane in payload.planes
    ]
    await db.upsert_system(system, planes)
    stored = await db.get_system(system_id)
    if stored is None:
        raise HTTPException(status_code=500, detail="system was not persisted")
    pipeline.invalidate()
    return _to_out(stored)


@router.get("", response_model=list[SystemOut])
async def list_systems(db: Database = Depends(get_db)) -> list[SystemOut]:
    return [_to_out(system) for system in await db.list_systems()]


@router.get("/{system_id}", response_model=SystemOut)
async def get_system(system_id: str, db: Database = Depends(get_db)) -> SystemOut:
    system = await db.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail="system not found")
    return _to_out(system)


@router.post("/{system_id}/retrain", status_code=202)
async def retrain(
    system_id: str,
    background: BackgroundTasks,
    db: Database = Depends(get_db),
    pipeline: ForecastPipeline = Depends(get_pipeline),
) -> dict:
    system = await db.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail="system not found")
    from app.jobs.scheduler import retrain_system

    background.add_task(retrain_system, db, pipeline, system)
    return {"status": "scheduled"}


@router.delete("/{system_id}", status_code=204)
async def delete_system(
    system_id: str, db: Database = Depends(get_db), pipeline: ForecastPipeline = Depends(get_pipeline)
) -> None:
    if not await db.delete_system(system_id):
        raise HTTPException(status_code=404, detail="system not found")
    pipeline.invalidate()
