"""Forecast.Solar polling for shadow benchmarking."""

from __future__ import annotations

import collections
import logging
import time
from zoneinfo import ZoneInfo

import httpx

from .database import Database, plane_id_for

LOG = logging.getLogger(__name__)


class RateLimited(Exception):
    pass


class ForecastSolarClient:
    def __init__(
        self,
        base_url: str = "https://api.forecast.solar",
        http_client: httpx.AsyncClient | None = None,
        now=time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = http_client
        self._calls: collections.deque[float] = collections.deque()
        self._disabled_until = 0.0
        self._now = now

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _check_budget(self) -> None:
        now = self._now()
        if now < self._disabled_until:
            raise RateLimited("disabled until next slot")
        while self._calls and now - self._calls[0] > 3600:
            self._calls.popleft()
        if len(self._calls) >= 3:
            raise RateLimited("3 calls per hour budget exhausted")

    async def estimate_plane(self, lat: float, lon: float, dec: float, az: float, kwp: float):
        self._check_budget()
        http = await self._http()
        url = f"{self.base_url}/estimate/{lat}/{lon}/{dec}/{az}/{kwp}"
        self._calls.append(self._now())
        response = await http.get(url)
        if response.status_code == 429:
            try:
                ratelimit = response.json().get("message", {}).get("ratelimit", {})
                period = float(ratelimit.get("period", 3600))
            except Exception:
                period = 3600.0
            self._disabled_until = self._now() + period
            raise RateLimited("upstream 429")
        return response


def parse_estimate(payload: dict) -> dict[int, float]:
    """Convert ``watt_hours_period`` (local period-end labels) to UTC period starts."""
    timezone = payload.get("message", {}).get("info", {}).get("timezone", "UTC")
    tz = ZoneInfo(timezone)
    import datetime as dt

    periods: dict[int, float] = {}
    values = payload.get("result", {}).get("watt_hours_period", {})
    for label, energy in values.items():
        stamp = dt.datetime.strptime(label, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
        start = stamp - dt.timedelta(hours=1)
        key = int(start.timestamp())
        periods[key] = periods.get(key, 0.0) + float(energy)
    return periods


async def fetch_system_shadow(
    db: Database,
    system_row: dict,
    client: ForecastSolarClient,
    inverter_kw: float | None = None,
    fetched_at: int | None = None,
) -> int:
    """Poll every plane, persist raw responses and merged hourly runs."""
    import json
    import time as time_module

    fetched_at = fetched_at or int(time_module.time())
    totals: dict[int, float] = {}
    for plane in system_row["planes"]:
        try:
            response = await client.estimate_plane(
                system_row["lat"],
                system_row["lon"],
                plane["dec"],
                plane["az_fs"],
                plane["kwp"],
            )
        except RateLimited as exc:
            LOG.warning("%s: Forecast.Solar skipped: %s", system_row["system_id"], exc)
            break
        pid = plane.get("plane_id") or plane_id_for(
            system_row["system_id"],
            plane["dec"],
            plane["az_fs"],
            plane["kwp"],
            plane.get("inverter_id"),
        )
        await db.insert_forecast_solar_raw(
            system_row["system_id"], pid, fetched_at, response.status_code, response.text
        )
        if response.status_code != 200:
            LOG.warning("Forecast.Solar %s for plane %s", response.status_code, pid)
            continue
        try:
            periods = parse_estimate(response.json())
        except Exception as exc:
            LOG.warning("cannot parse Forecast.Solar response: %s", exc)
            continue
        for start, energy in periods.items():
            totals[start] = totals.get(start, 0.0) + energy

    if not totals:
        return 0
    if inverter_kw:
        cap = float(inverter_kw) * 1000.0
        totals = {start: min(energy, cap) for start, energy in totals.items()}
    rows = [
        {
            "period_start": start,
            "energy_wh": energy,
            "mean_watts": energy,
        }
        for start, energy in sorted(totals.items())
    ]
    await db.insert_forecast_solar_runs(system_row["system_id"], fetched_at, rows)
    return len(rows)
