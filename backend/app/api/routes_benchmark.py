from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_db, require_auth
from app.core.benchmark import benchmark
from app.core.database import Database

router = APIRouter(prefix="/api/v1/systems", tags=["benchmark"], dependencies=[Depends(require_auth)])


@router.get("/{system_id}/benchmark")
async def get_benchmark(
    system_id: str,
    days: int = Query(default=14, ge=1, le=400),
    db: Database = Depends(get_db),
):
    system = await db.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail="system not found")
    return await benchmark(db, system, days)
