from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator


class PlaneIn(BaseModel):
    dec: float = Field(ge=0, le=90)
    az: float = Field(ge=-180, le=180)
    kwp: float = Field(gt=0)


class SystemIn(BaseModel):
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    elevation_m: float | None = None
    timezone: str | None = None
    inverter_kw: float | None = Field(default=None, gt=0)
    system_loss: float = Field(default=0.10, ge=0, lt=1)
    gamma_pdc: float = -0.004
    damping_morning: float = Field(default=0.0, ge=0, le=1)
    damping_evening: float = Field(default=0.0, ge=0, le=1)
    planes: list[PlaneIn] = Field(min_length=1, max_length=4)

    @field_validator("planes")
    @classmethod
    def _unique_ordinals(cls, planes: list[PlaneIn]) -> list[PlaneIn]:
        return planes


class PlaneOut(PlaneIn):
    plane_id: str
    ordinal: int


class SystemOut(BaseModel):
    system_id: str
    name: str
    lat: float
    lon: float
    elevation_m: float | None
    timezone: str
    inverter_kw: float | None
    system_loss: float
    gamma_pdc: float
    damping_morning: float
    damping_evening: float
    planes: list[PlaneOut]


class PeriodOut(BaseModel):
    start: str
    end: str
    wh: float
    mean_watts: float


class DailyOut(BaseModel):
    date: dt.date
    wh: float


class ForecastOut(BaseModel):
    system_id: str
    generated_at: str
    weather_fetched_at: str
    timezone: str
    resolution_minutes: int
    layer: str
    kwp_total: float
    ac_limit_w: float
    layers_active: dict[str, bool]
    periods: list[PeriodOut]
    daily: list[DailyOut]


class TelemetryItemIn(BaseModel):
    period_start: dt.datetime
    period_min: int = 15
    energy_wh: float
    curtailed: bool = False
    inverter_temp_c: float | None = None

    @field_validator("period_start")
    @classmethod
    def _require_tz(cls, value: dt.datetime) -> dt.datetime:
        if value.tzinfo is None:
            raise ValueError("period_start must be timezone aware")
        return value

    @field_validator("period_min")
    @classmethod
    def _allowed_period(cls, value: int) -> int:
        if value not in (15, 60):
            raise ValueError("period_min must be 15 or 60")
        return value


class TelemetryBatchIn(BaseModel):
    items: list[TelemetryItemIn] = Field(min_length=1, max_length=5000)
    origin: str = "push"

    @field_validator("origin")
    @classmethod
    def _origin(cls, value: str) -> str:
        if value not in ("push", "backfill"):
            raise ValueError("origin must be push or backfill")
        return value


class RejectedItem(BaseModel):
    index: int
    reason: str


class TelemetryBatchOut(BaseModel):
    accepted: int
    rejected: list[RejectedItem]


class HealthOut(BaseModel):
    status: str
    database: str
    scheduler_timezone: str | None
    jobs: list[dict]
    systems: list[dict]
