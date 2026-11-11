"""DataUpdateCoordinator for SolarCast."""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import SolarCastApi, SolarCastError
from .const import ACCURACY_INTERVAL_SECONDS, DOMAIN, UPDATE_INTERVAL_MINUTES

LOG = logging.getLogger(__name__)


def _parse_dt(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"naive timestamp from backend: {value}")
    return parsed


@dataclass(frozen=True)
class ForecastPeriod:
    start: dt.datetime
    end: dt.datetime
    wh: float
    mean_watts: float


@dataclass
class ForecastData:
    hourly: list[ForecastPeriod]
    intraday: list[ForecastPeriod]
    timezone: str
    layer: str
    layers_active: dict[str, bool]
    generated_at: dt.datetime
    ac_limit_w: float
    accuracy_nmae_14d: float | None = None
    accuracy_detail: dict = field(default_factory=dict)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def local_day(self, period: ForecastPeriod, offset: int = 0) -> bool:
        target = (dt.datetime.now(self.tz) + dt.timedelta(days=offset)).date()
        return period.start.astimezone(self.tz).date() == target


def _periods(payload: dict) -> list[ForecastPeriod]:
    return [
        ForecastPeriod(
            start=_parse_dt(item["start"]),
            end=_parse_dt(item["end"]),
            wh=float(item["wh"]),
            mean_watts=float(item["mean_watts"]),
        )
        for item in payload.get("periods", [])
    ]


class SolarCastCoordinator(DataUpdateCoordinator[ForecastData]):
    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, api: SolarCastApi, system_id: str
    ) -> None:
        super().__init__(
            hass,
            LOG,
            name=f"{DOMAIN} {system_id}",
            update_interval=dt.timedelta(minutes=UPDATE_INTERVAL_MINUTES),
        )
        self.api = api
        self.system_id = system_id
        self.entry = entry
        self._accuracy: dict | None = None
        self._accuracy_fetched = 0.0

    async def _async_update_data(self) -> ForecastData:
        try:
            hourly, intraday = await self._fetch_both()
        except SolarCastError as exc:
            raise UpdateFailed(str(exc)) from exc
        now = time.monotonic()
        if now - self._accuracy_fetched > ACCURACY_INTERVAL_SECONDS:
            try:
                self._accuracy = await self.api.benchmark(self.system_id, 14)
                self._accuracy_fetched = now
            except SolarCastError as exc:
                LOG.debug("benchmark unavailable: %s", exc)
        accuracy = self._accuracy or {}
        nmae = None
        for bucket in ("day_ahead", "intraday", "d2_3", "d4_7"):
            sources = accuracy.get("buckets", {}).get(bucket, {}).get("sources", {})
            for source in ("final", "hybrid", "physics", "physics_base"):
                if sources.get(source, {}).get("nMAE_pct") is not None:
                    nmae = float(sources[source]["nMAE_pct"])
                    break
            if nmae is not None:
                break
        return ForecastData(
            hourly=_periods(hourly),
            intraday=_periods(intraday),
            timezone=hourly["timezone"],
            layer=hourly["layer"],
            layers_active=hourly.get("layers_active", {}),
            generated_at=_parse_dt(hourly["generated_at"]),
            ac_limit_w=float(hourly.get("ac_limit_w", 0.0)),
            accuracy_nmae_14d=nmae,
            accuracy_detail=accuracy,
        )

    async def _fetch_both(self):
        hourly = await self.api.forecast(self.system_id, 60, 7, "final")
        intraday = await self.api.forecast(self.system_id, 15, 2, "final")
        return hourly, intraday
