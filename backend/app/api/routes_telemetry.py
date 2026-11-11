from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_db, get_pipeline, require_auth
from app.core.database import Database
from app.core.pipeline import ForecastPipeline
from app.core.timeconv import is_aligned
from app.schemas.native import (
    RejectedItem,
    TelemetryBatchIn,
    TelemetryBatchOut,
    TelemetryDeleteOut,
)

LOG = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/systems", tags=["telemetry"], dependencies=[Depends(require_auth)])


@router.post("/{system_id}/telemetry", response_model=TelemetryBatchOut)
async def post_telemetry(
    system_id: str,
    batch: TelemetryBatchIn,
    db: Database = Depends(get_db),
    pipeline: ForecastPipeline = Depends(get_pipeline),
) -> TelemetryBatchOut:
    system = await db.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail="system not found")
    if system.get("inverters"):
        paco = sum(float(inv["ac_kw"]) for inv in system["inverters"]) * 1000.0
    elif system.get("inverter_kw"):
        paco = float(system["inverter_kw"]) * 1000.0
    else:
        paco = sum(float(p["kwp"]) for p in system["planes"]) * 1000.0
    now = dt.datetime.now(dt.timezone.utc)
    accepted_items = []
    rejected: list[RejectedItem] = []
    for index, item in enumerate(batch.items):
        start = item.period_start.astimezone(dt.timezone.utc)
        if not is_aligned(start, item.period_min):
            rejected.append(RejectedItem(index=index, reason="period_start not aligned"))
            continue
        if start > now:
            rejected.append(RejectedItem(index=index, reason="future period"))
            continue
        if item.energy_wh < 0:
            rejected.append(RejectedItem(index=index, reason="negative energy"))
            continue
        hours = item.period_min / 60.0
        mean_watts = item.energy_wh / hours
        cap = 1.1 * paco * hours
        if item.energy_wh > cap:
            LOG.warning(
                "telemetry above cap for %s at %s: %.1f Wh > %.1f Wh",
                system_id,
                start.isoformat(),
                item.energy_wh,
                cap,
            )
        accepted_items.append(
            {
                "period_start": int(start.timestamp()),
                "period_min": item.period_min,
                "energy_wh": float(item.energy_wh),
                "mean_watts": float(mean_watts),
                "curtailed": bool(item.curtailed),
                "inverter_temp_c": item.inverter_temp_c,
                "origin": batch.origin,
            }
        )
    accepted = await db.upsert_telemetry(system_id, accepted_items) if accepted_items else 0
    pipeline.invalidate()
    return TelemetryBatchOut(accepted=accepted, rejected=rejected)


@router.delete("/{system_id}/telemetry", response_model=TelemetryDeleteOut)
async def delete_telemetry(
    system_id: str,
    start: dt.datetime = Query(..., description="Inclusive ISO 8601 start of the range"),
    end: dt.datetime = Query(..., description="Exclusive ISO 8601 end of the range"),
    db: Database = Depends(get_db),
    pipeline: ForecastPipeline = Depends(get_pipeline),
) -> TelemetryDeleteOut:
    if start.tzinfo is None or end.tzinfo is None:
        raise HTTPException(status_code=422, detail="start and end must be timezone aware")
    start_utc = start.astimezone(dt.timezone.utc)
    end_utc = end.astimezone(dt.timezone.utc)
    if end_utc <= start_utc:
        raise HTTPException(status_code=422, detail="end must be after start")
    if await db.get_system(system_id) is None:
        raise HTTPException(status_code=404, detail="system not found")
    deleted = await db.delete_telemetry(
        system_id, int(start_utc.timestamp()), int(end_utc.timestamp())
    )
    pipeline.invalidate()
    LOG.info("deleted %s telemetry rows for %s in [%s, %s)", deleted, system_id, start_utc, end_utc)
    return TelemetryDeleteOut(
        deleted=deleted, start=start_utc.isoformat(), end=end_utc.isoformat()
    )
