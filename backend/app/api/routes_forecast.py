from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_pipeline, require_auth
from app.core.pipeline import ForecastPipeline
from app.schemas.native import ForecastOut

router = APIRouter(prefix="/api/v1/systems", tags=["forecast"], dependencies=[Depends(require_auth)])

Layer = Literal["final", "hybrid", "physics", "physics_base"]


@router.get("/{system_id}/forecast", response_model=ForecastOut)
async def get_forecast(
    system_id: str,
    resolution: int = Query(default=60),
    days: int = Query(default=7, ge=1, le=7),
    layer: Layer = Query(default="final"),
    pipeline: ForecastPipeline = Depends(get_pipeline),
) -> ForecastOut:
    if resolution not in (15, 60):
        raise HTTPException(status_code=422, detail="resolution must be 15 or 60")
    try:
        payload = await pipeline.get_forecast(system_id, resolution, days, layer)
    except KeyError:
        raise HTTPException(status_code=404, detail="system not found") from None
    return ForecastOut.model_validate(payload)
