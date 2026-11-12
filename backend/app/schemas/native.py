from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class InverterIn(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}$")
    ac_kw: float = Field(gt=0)


class PlaneIn(BaseModel):
    dec: float = Field(ge=0, le=90)
    az: float = Field(ge=-180, le=180)
    kwp: float = Field(gt=0)
    inverter: str | None = None


class SystemIn(BaseModel):
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    elevation_m: float | None = None
    timezone: str | None = None
    inverter_kw: float | None = Field(default=None, gt=0)
    inverters: list[InverterIn] = Field(default_factory=list, max_length=8)
    system_loss: float = Field(default=0.10, ge=0, lt=1)
    gamma_pdc: float = -0.004
    damping_morning: float = Field(default=0.0, ge=0, le=1)
    damping_evening: float = Field(default=0.0, ge=0, le=1)
    local_fallback: bool = True
    planes: list[PlaneIn] = Field(min_length=1, max_length=4)

    @field_validator("planes")
    @classmethod
    def _distinct_planes(cls, planes: list[PlaneIn]) -> list[PlaneIn]:
        seen: set[tuple[str, str, str, str]] = set()
        for plane in planes:
            # Mirrors plane_id_for, which is what the database keys planes by.
            key = (f"{plane.dec:.1f}", f"{plane.az:.1f}", f"{plane.kwp:.3f}", plane.inverter or "")
            if key in seen:
                raise ValueError(
                    "duplicate plane: identical tilt, azimuth and kWp on the same "
                    "inverter; merge them or give each plane its own inverter"
                )
            seen.add(key)
        return planes

    @model_validator(mode="after")
    def _validate_inverters(self) -> "SystemIn":
        ids = [inverter.id for inverter in self.inverters]
        if self.inverter_kw is not None and self.inverters:
            raise ValueError("use either inverter_kw or inverters")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate inverter id")
        if self.inverters and any(plane.inverter is None for plane in self.planes):
            raise ValueError("every plane must reference an inverter")
        if self.inverters and any(plane.inverter not in ids for plane in self.planes):
            raise ValueError("plane references unknown inverter")
        if not self.inverters and any(plane.inverter is not None for plane in self.planes):
            raise ValueError("plane references unknown inverter")
        if self.inverter_kw is not None and any(
            plane.inverter is not None for plane in self.planes
        ):
            raise ValueError("plane references unknown inverter")
        return self


class InverterOut(InverterIn):
    pass


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
    inverters: list[InverterOut]
    system_loss: float
    gamma_pdc: float
    damping_morning: float
    damping_evening: float
    local_fallback: bool = True
    planes: list[PlaneOut]


class PeriodOut(BaseModel):
    start: str
    end: str
    wh: float
    mean_watts: float
    confidence: str | None = None


class TodayOut(BaseModel):
    date: str
    wh: float
    elapsed_wh: float
    remaining_wh: float
    measured_wh: float | None = None
    simulated_wh: float | None = None
    source: str = "forecast"


class DailyOut(BaseModel):
    date: dt.date
    wh: float


class HorizonPoint(BaseModel):
    azimuth_deg: float
    elevation_deg: float


class HorizonOut(BaseModel):
    system_id: str
    source: Literal["user"]
    points: list[HorizonPoint]


class LearningOut(BaseModel):
    calibration_c: float
    calibration_days: int
    mos: bool
    shading: bool
    nowcast: bool
    nowcast_buckets: int
    telemetry_last_seen: str | None
    telemetry_age_minutes: float | None
    local: bool = False
    local_skill: float | None = None
    local_trained_at: int | None = None


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
    learning: LearningOut
    provenance: str = "remote"
    degraded: bool = False
    reason: str | None = None
    local_horizon_minutes: int = 0
    today: TodayOut | None = None
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
        if value not in ("push", "backfill", "manual"):
            raise ValueError("origin must be push, backfill or manual")
        return value


class RejectedItem(BaseModel):
    index: int
    reason: str


class TelemetryBatchOut(BaseModel):
    accepted: int
    rejected: list[RejectedItem]


class TelemetryDeleteOut(BaseModel):
    deleted: int
    start: str
    end: str


class HealthOut(BaseModel):
    status: str
    database: str
    scheduler_timezone: str | None
    jobs: list[dict]
    systems: list[dict]
