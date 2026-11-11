"""One-time telemetry history import from recorder statistics."""

from __future__ import annotations

import datetime as dt
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .api import LocalSolarCastApi, LocalSolarCastError
from .const import DOMAIN

LOG = logging.getLogger(__name__)

SHORT_TERM_DAYS = 10
BATCH_SIZE = 500


def _statistics(hass: HomeAssistant, entity_id: str, start: dt.datetime, end: dt.datetime, period: str):
    from homeassistant.components.recorder.statistics import statistics_during_period

    result = statistics_during_period(
        hass,
        start,
        end,
        {entity_id},
        period=period,
        units={"energy": "Wh"},
        types={"sum"},
    )
    return result.get(entity_id, []) if result else []


def _deltas(rows: list[dict]) -> list[tuple[dt.datetime, float]]:
    out = []
    previous = None
    for row in rows:
        value = row.get("sum")
        if value is None:
            continue
        if previous is not None:
            delta = float(value) - previous
            if delta >= 0:
                out.append((row["start"], delta))
        previous = float(value)
    return out


async def async_backfill(
    hass: HomeAssistant, api: LocalSolarCastApi, system_id: str, entity_id: str, days: int = 60
) -> int:
    days = max(1, min(int(days), 60))
    now = dt.datetime.now(dt.timezone.utc)
    short_start = max(now - dt.timedelta(days=SHORT_TERM_DAYS), now - dt.timedelta(days=days))
    items: list[dict] = []

    try:
        five_min = await hass.async_add_executor_job(
            _statistics, hass, entity_id, short_start, now, "5minute"
        )
        buckets: dict[dt.datetime, float] = {}
        for stamp, energy in _deltas(five_min):
            aligned = stamp.replace(second=0, microsecond=0)
            aligned -= dt.timedelta(minutes=aligned.minute % 15)
            buckets[aligned] = buckets.get(aligned, 0.0) + energy
        for stamp, energy in sorted(buckets.items()):
            items.append(
                {
                    "period_start": stamp.isoformat(),
                    "period_min": 15,
                    "energy_wh": round(energy, 3),
                }
            )

        if days > SHORT_TERM_DAYS:
            hour_start = now - dt.timedelta(days=days)
            hour_end = now - dt.timedelta(days=SHORT_TERM_DAYS)
            hourly = await hass.async_add_executor_job(
                _statistics, hass, entity_id, hour_start, hour_end, "hour"
            )
            for stamp, energy in _deltas(hourly):
                items.append(
                    {
                        "period_start": stamp.isoformat(),
                        "period_min": 60,
                        "energy_wh": round(energy, 3),
                    }
                )
    except Exception as exc:
        LOG.warning("recorder statistics unavailable: %s", exc)
        return 0

    accepted = 0
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        try:
            result = await api.telemetry(system_id, batch, origin="backfill")
            accepted += int(result.get("accepted", 0))
        except LocalSolarCastError as exc:
            LOG.warning("backfill batch failed: %s", exc)
            break
    return accepted


async def async_maybe_backfill(
    hass: HomeAssistant, api: LocalSolarCastApi, entry, entity_id: str, days: int = 60
) -> None:
    store = Store(hass, 1, f"{DOMAIN}.{entry.entry_id}.backfill")
    done = await store.async_load()
    if done:
        return
    accepted = await async_backfill(hass, api, entry.data["system_id"], entity_id, days)
    if accepted:
        await store.async_save({"completed": True, "accepted": accepted})
    LOG.info("backfill complete: %d rows", accepted)
