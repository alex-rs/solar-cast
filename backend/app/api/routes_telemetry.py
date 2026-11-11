from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_db, get_pipeline, require_auth
from app.core.database import Database
from app.core.pipeline import ForecastPipeline
from app.core.timeconv import is_aligned
from app.schemas.native import TelemetryBatchIn, TelemetryBatchOut, RejectedItem

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
    paco = (
        float(system["inverter_kw"]) * 1000.0
        if system.get("inverter_kw")
        else sum(float(p["kwp"]) for p in system["planes"]) * 1000.0
    )
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
