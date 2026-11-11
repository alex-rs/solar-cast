from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import get_pipeline

router = APIRouter(tags=["compat"])

PlaneGroups = list[tuple[float, float, float]]


def _parse_planes(raw: str) -> PlaneGroups:
    parts = [part for part in raw.split("/") if part != ""]
    if len(parts) % 3 != 0 or not 1 <= len(parts) // 3 <= 4:
        raise HTTPException(
            status_code=422, detail="planes must be 1-4 groups of dec/az/kwp"
        )
    planes: PlaneGroups = []
    for i in range(0, len(parts), 3):
        try:
            dec, az, kwp = (float(parts[i]), float(parts[i + 1]), float(parts[i + 2]))
        except ValueError:
            raise HTTPException(status_code=422, detail="plane values must be numeric") from None
        if not 0 <= dec <= 90 or not -180 <= az <= 180 or kwp <= 0:
            raise HTTPException(status_code=422, detail="plane values out of range")
        planes.append((dec, az, kwp))
    return planes


async def _estimate(
    request: Request,
    lat: float,
    lon: float,
    planes_raw: str,
    inverter: float | None,
    damping: str | None,
    damping_morning: float | None,
    damping_evening: float | None,
):
    settings = request.app.state.settings
    if not settings.compat_public:
        from app.api.deps import require_auth

        await require_auth(request)
    planes = _parse_planes(planes_raw)
    morning = damping_morning or 0.0
    evening = damping_evening or 0.0
    if damping:
        try:
            morning_part, evening_part = (float(v) for v in damping.split(","))
            morning, evening = morning_part, evening_part
        except ValueError:
            raise HTTPException(status_code=422, detail="damping must be 'morning,evening'") from None
    pipeline = get_pipeline(request)
    return await pipeline.compat_estimate(
        lat=lat,
        lon=lon,
        planes=planes,
        inverter_kw=inverter,
        damping_morning=morning,
        damping_evening=evening,
    )


@router.get("/estimate/{lat}/{lon}/{planes:path}")
async def estimate(
    request: Request,
    lat: float,
    lon: float,
    planes: str,
    inverter: float | None = Query(default=None),
    damping: str | None = Query(default=None),
    damping_morning: float | None = Query(default=None),
    damping_evening: float | None = Query(default=None),
):
    return await _estimate(request, lat, lon, planes, inverter, damping, damping_morning, damping_evening)


@router.get("/{apikey}/estimate/{lat}/{lon}/{planes:path}")
async def estimate_with_key(
    request: Request,
    apikey: str,
    lat: float,
    lon: float,
    planes: str,
    inverter: float | None = Query(default=None),
    damping: str | None = Query(default=None),
    damping_morning: float | None = Query(default=None),
    damping_evening: float | None = Query(default=None),
):
    return await _estimate(request, lat, lon, planes, inverter, damping, damping_morning, damping_evening)
