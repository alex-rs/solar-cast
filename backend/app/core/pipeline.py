"""Forecast orchestration: weather, physics, learning layers and payloads."""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib

from config.settings import Settings

from .accuracy import calibration, local_model, mos, nowcast, shading
from .clearsky import clearsky_ghi, simulate_clearsky
from .database import Database
from .openmeteo_client import OpenMeteoClient
from .physics_engine import (
    InverterConfig,
    PlaneConfig,
    SimResult,
    SystemConfig,
    simulate_system,
)
from .pvgis_client import Horizon, fetch_horizon, flat_fallback
from .timeconv import (
    UTC,
    aggregate_periods,
    ensure_utc,
    instant_energy,
    local_daily_energy,
    period_start,
    ts_to_dt,
)

LOG = logging.getLogger(__name__)

CACHE_TTL_S = 300
WEATHER_TTL_S = 3600
HORIZON_RETRY_S = 7 * 86400


@dataclass
class LayerResult:
    base: SimResult
    physics: SimResult
    hybrid: SimResult
    final: SimResult
    calibration_c: float
    active: dict[str, bool]
    nowcast_active: bool
    nowcast_buckets: int = 0
    learning: dict[str, object] = field(default_factory=dict)
    provenance: str = "remote"
    degraded: bool = False
    reason: str | None = None
    local_horizon_min: int = 0


def system_config(row: dict) -> SystemConfig:
    return SystemConfig(
        system_id=row["system_id"],
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        elevation_m=float(row["elevation_m"] or 0.0),
        inverters=tuple(
            InverterConfig(
                inverter_id=inverter["inverter_id"],
                ac_kw=float(inverter["ac_kw"]),
            )
            for inverter in row["inverters"]
        ),
        system_loss=float(row["system_loss"]),
        gamma_pdc=float(row["gamma_pdc"]),
        damping_morning=float(row["damping_morning"]),
        damping_evening=float(row["damping_evening"]),
        timezone=row["timezone"],
        planes=tuple(
            PlaneConfig(
                dec=float(plane["dec"]),
                az_fs=float(plane["az_fs"]),
                kwp=float(plane["kwp"]),
                plane_id=plane.get("plane_id"),
                inverter_id=plane.get("inverter_id"),
            )
            for plane in row["planes"]
        ),
    )


def _with_pac(source: SimResult, pac: pd.Series) -> SimResult:
    return SimResult(
        pac=pac.rename("pac"),
        pdc=source.pdc,
        paco=source.paco,
        f_b=source.f_b,
        f_d=source.f_d,
        zenith=source.zenith,
        azimuth=source.azimuth,
        elevation=source.elevation,
        planes=source.planes,
        inverters=source.inverters,
    )


class ForecastPipeline:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        weather_client: OpenMeteoClient | None = None,
        horizon_fetcher=None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.weather = weather_client or OpenMeteoClient(
            base_url=settings.openmeteo_base_url,
            models=settings.openmeteo_models,
            weights=settings.openmeteo_weights,
            api_key=settings.openmeteo_api_key,
        )
        self.horizon_fetcher = horizon_fetcher or fetch_horizon
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self._tzfinder = None
        from .forecast_solar_client import ForecastSolarClient

        self.fs_client = ForecastSolarClient(settings.forecast_solar_base_url)

    # -- caches ---------------------------------------------------------
    def invalidate(self) -> None:
        self._cache.clear()
        self.weather.invalidate()

    def timezone_for(self, lat: float, lon: float) -> str:
        if self._tzfinder is None:
            from timezonefinder import TimezoneFinder

            self._tzfinder = TimezoneFinder()
        return self._tzfinder.timezone_at(lat=lat, lng=lon) or "UTC"

    # -- data -----------------------------------------------------------
    async def ensure_weather(self, lat: float, lon: float):
        frame, fetched_at = await self.db.latest_weather(lat, lon)
        now = int(time.time())
        weights_updated = await self.db.weights_updated_at(lat, lon)
        if (
            frame is not None
            and fetched_at is not None
            and now - fetched_at < WEATHER_TTL_S
            and (weights_updated is None or weights_updated <= fetched_at)
        ):
            return frame, fetched_at
        try:
            stored = await self.db.get_weights(lat, lon)
            weights = None
            if stored and all(model in stored for model in self.settings.openmeteo_models):
                weights = [stored[model] for model in self.settings.openmeteo_models]
            result = await self.weather.get(lat, lon, weights=weights)
            fetched_at = now
            await self.db.store_weather(
                lat, lon, fetched_at, result.frame, result.model_ghi
            )
            if result.elevation_m is not None:
                for system in await self.db.list_systems():
                    if round(system["lat"], 3) == round(lat, 3) and round(
                        system["lon"], 3
                    ) == round(lon, 3):
                        await self.db.set_elevation(system["system_id"], result.elevation_m)
            return result.frame, fetched_at
        except Exception as exc:
            if frame is not None and fetched_at is not None:
                LOG.warning("weather refresh failed, using stale snapshot: %s", exc)
                return frame, fetched_at
            raise

    async def refresh_weather(self) -> dict[str, int]:
        locations = {(round(s["lat"], 3), round(s["lon"], 3)) for s in await self.db.list_systems()}
        refreshed = 0
        for lat, lon in locations:
            try:
                stored = await self.db.get_weights(lat, lon)
                weights = None
                if stored and all(
                    model in stored for model in self.settings.openmeteo_models
                ):
                    weights = [
                        stored[model] for model in self.settings.openmeteo_models
                    ]
                result = await self.weather.get(lat, lon, weights=weights)
                fetched_at = int(time.time())
                await self.db.store_weather(
                    lat, lon, fetched_at, result.frame, result.model_ghi
                )
                for system in await self.db.list_systems():
                    if round(system["lat"], 3) == lat and round(system["lon"], 3) == lon:
                        if result.elevation_m is not None:
                            await self.db.set_elevation(system["system_id"], result.elevation_m)
                refreshed += 1
            except Exception as exc:
                LOG.error("weather refresh failed for %s,%s: %s", lat, lon, exc)
        self.invalidate()
        return {"locations": len(locations), "refreshed": refreshed}

    async def ensure_horizon(self, lat: float, lon: float) -> Horizon:
        rows = await self.db.get_horizon(lat, lon)
        now = int(time.time())
        if rows:
            source = rows[0]["source"]
            fetched_at = max(int(row["fetched_at"]) for row in rows)
            if source != "flat_fallback" or now - fetched_at < HORIZON_RETRY_S:
                return Horizon(
                    azimuth=np.array([float(row["azimuth_deg"]) for row in rows]),
                    elevation=np.array([float(row["elevation_deg"]) for row in rows]),
                    source=source,
                )
        horizon = await self.horizon_fetcher(
            lat, lon, base_url=self.settings.pvgis_base_url
        )
        await self.db.put_horizon(
            lat, lon, horizon.azimuth, horizon.elevation, horizon.source
        )
        return horizon

    async def ensure_horizon_for_system(self, system_row: dict) -> Horizon:
        rows = await self.db.get_system_horizon(system_row["system_id"])
        if rows:
            return Horizon(
                azimuth=np.array([float(r["azimuth_deg"]) for r in rows]),
                elevation=np.array([float(r["elevation_deg"]) for r in rows]),
                source="user",
            )
        return await self.ensure_horizon(system_row["lat"], system_row["lon"])

    # -- layers ---------------------------------------------------------
    async def compute_layers(self, system_row: dict, regenerate: bool = False) -> LayerResult:
        system = system_config(system_row)
        try:
            weather, fetched_at = await self.ensure_weather(system.lat, system.lon)
        except Exception as exc:
            return await self.local_layers(system_row, system, exc)
        weather_age = int(time.time()) - int(fetched_at)
        provenance = "remote" if weather_age <= WEATHER_TTL_S else "remote_stale"
        horizon = await self.ensure_horizon_for_system(system_row)

        base = simulate_system(weather, system, horizon, calibration_c=1.0)
        c_row = await self.db.get_calibration(system.system_id)
        calibration_c = float(c_row["factor_c"]) if c_row else 1.0
        lookup = await shading.load(self.db, system.system_id)
        physics = simulate_system(weather, system, horizon, shading=lookup, calibration_c=calibration_c)

        model = await mos.load_model(self.db, system.system_id)
        features = None
        cs_pac = None
        if model is not None:
            cs = simulate_clearsky(weather.index, system, horizon)
            cs_pac = cs.pac
            clear_ghi = clearsky_ghi(weather.index, system)
            features = mos.build_features(
                weather, system, fetched_at, physics.pac, cs_pac, clear_ghi
            )
        hybrid_pac = mos.apply_hybrid(physics, base, system, model, features)
        hybrid = _with_pac(physics, hybrid_pac)

        telemetry = await self.db.last_telemetry(system.system_id)
        now = dt.datetime.now(UTC)
        last_seen = None
        if telemetry is not None:
            last_seen = ts_to_dt(telemetry["period_start"]) + dt.timedelta(
                minutes=int(telemetry["period_min"])
            )
        age_minutes = (
            round(max(0.0, (now - last_seen).total_seconds() / 60.0), 1)
            if last_seen is not None
            else None
        )
        nowcast_buckets = 0
        nowcast_active = False
        final = hybrid
        if telemetry is not None:
            cs_ref = simulate_clearsky(
                weather.index, system, horizon, shading=lookup, calibration_c=calibration_c
            )
            # The eligibility rule is "ended within 45 min"; a finished 60-min
            # bucket therefore starts up to 45 + 60 = 105 min ago.
            recent = await self.db.telemetry(
                system.system_id,
                int(now.timestamp()) - 105 * 60,
                int(now.timestamp()) + 1,
            )
            ref = nowcast.deltas_from_telemetry(
                recent,
                now,
                cs_ref.pac,
                hybrid.pac,
                cs_ref.zenith,
                0.1 * system.kwp_total * 1000.0,
            )
            if ref is not None:
                anchor_start, delta_k, nowcast_buckets = ref
                final_pac, nowcast_active = nowcast.apply(
                    hybrid.pac,
                    cs_ref.pac,
                    anchor_start,
                    delta_k,
                    physics.paco,
                )
                final = _with_pac(physics, final_pac)

        return LayerResult(
            base=base,
            physics=physics,
            hybrid=hybrid,
            final=final,
            calibration_c=calibration_c,
            active={
                "calibration": c_row is not None,
                "shading": lookup is not None,
                "mos": model is not None,
                "nowcast": nowcast_active,
            },
            nowcast_active=nowcast_active,
            nowcast_buckets=nowcast_buckets,
            provenance=provenance,
            degraded=provenance != "remote",
            reason=None if provenance == "remote" else "remote_stale",
            learning={
                "calibration_c": round(calibration_c, 4),
                "calibration_days": int(c_row["clear_days"]) if c_row else 0,
                "mos": model is not None,
                "shading": lookup is not None,
                "nowcast": nowcast_active,
                "nowcast_buckets": nowcast_buckets,
                "telemetry_last_seen": last_seen.isoformat() if last_seen else None,
                "telemetry_age_minutes": age_minutes,
            },
        )

    # -- offline / local layers ----------------------------------------
    async def offline_horizon(self, system_row: dict) -> Horizon:
        """Horizon for a weather-free run; a flat horizon if none is cached."""
        try:
            return await self.ensure_horizon_for_system(system_row)
        except Exception as exc:  # noqa: BLE001 - any failure means "no horizon"
            LOG.warning("horizon unavailable offline, assuming flat: %s", exc)
            return flat_fallback()

    async def telemetry_power(self, system_id: str, now: dt.datetime) -> pd.Series:
        rows = await self.db.telemetry(
            system_id, int(now.timestamp()) - 6 * 3600, int(now.timestamp()) + 1
        )
        return local_model.telemetry_series(rows)

    async def local_layers(
        self, system_row: dict, system: SystemConfig, exc: Exception | None = None
    ) -> LayerResult:
        """Build a weather-free forecast from telemetry, or raise when disabled."""
        if not self.settings.enable_local_forecast or not system_row.get("local_fallback", 1):
            raise exc if exc is not None else RuntimeError("local fallback disabled")

        now = dt.datetime.now(UTC)
        horizon = await self.offline_horizon(system_row)
        grid = pd.date_range(
            # Six hours of history for the anchor features and the cold-start ratio.
            pd.Timestamp(now).floor("15min") - pd.Timedelta(hours=7),
            now + dt.timedelta(days=7),
            freq="15min",
        )
        clear = simulate_clearsky(grid, system, horizon)
        history = await self.telemetry_power(system.system_id, now)
        window = history[history.index >= grid[0]] if not history.empty else history

        model = await local_model.load(self.db, system.system_id)
        local_horizon = int(self.settings.local_forecast_hours) * 60
        pac: pd.Series | None = None
        provenance = "clearsky_climatology"
        reason = "no_local_model"
        used_local = False
        cutoff: pd.Timestamp | None = now

        if model is not None and not window.empty:
            anchor = window.index.max()
            features = local_model.anchor_features(
                window, clear, model, anchor, now=pd.Timestamp(now)
            )
            if features is not None:
                head = local_model.predict_power(
                    model,
                    clear,
                    features,
                    anchor,
                    float(window.loc[anchor]),
                    clear.paco,
                )
                head = head[head.index <= anchor + pd.Timedelta(minutes=local_horizon)]
                pac = local_model.with_tail(model, clear, head)
                provenance = "local_model"
                reason = None
                used_local = True
                local_horizon = model.horizon_min
                cutoff = anchor + pd.Timedelta(minutes=local_horizon)
            elif model is not None:
                reason = "no_recent_telemetry"

        if pac is None:
            ratio = 1.0
            if history.empty:
                reason = "no_telemetry"
            elif not window.empty:
                clear_window = clear.pac.reindex(window.index).clip(lower=0.0)
                usable = (
                    (window / clear_window.where(clear_window > 1e-6))
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if not usable.empty:
                    ratio = float(np.clip(usable.median(), *local_model.CLIM_CLIP))
            pac = local_model.climatology_power(clear, ratio)

        final = _with_pac(clear, pac)
        last_seen = window.index.max().to_pydatetime() if not window.empty else None
        age = round(max(0.0, (now - last_seen).total_seconds() / 60.0), 1) if last_seen else None
        return LayerResult(
            base=clear,
            physics=clear,
            hybrid=clear,
            final=final,
            calibration_c=1.0,
            active={
                "calibration": False,
                "shading": False,
                "mos": False,
                "nowcast": False,
            },
            nowcast_active=False,
            nowcast_buckets=0,
            learning={
                "calibration_c": 1.0,
                "calibration_days": 0,
                "mos": False,
                "shading": False,
                "nowcast": False,
                "nowcast_buckets": 0,
                "telemetry_last_seen": last_seen.isoformat() if last_seen else None,
                "telemetry_age_minutes": age,
                "local": used_local,
                "local_skill": (
                    model.holdout_skill.get("skill") if used_local and model else None
                ),
                "local_trained_at": model.trained_at if used_local and model else None,
            },
            provenance=provenance,
            degraded=True,
            reason=reason,
            local_horizon_min=local_horizon if used_local else 0,
        )

    def _layer(self, layers: LayerResult, name: str) -> SimResult:
        return {
            "physics_base": layers.base,
            "physics": layers.physics,
            "hybrid": layers.hybrid,
            "final": layers.final,
        }[name]

    # -- forecast -------------------------------------------------------
    async def get_forecast(
        self, system_id: str, resolution: int = 60, days: int = 7, layer: str = "final"
    ) -> dict:
        system_row = await self.db.get_system(system_id)
        if system_row is None:
            raise KeyError(system_id)
        key = (system_id, resolution, days, layer)
        cached = self._cache.get(key)
        now_mono = time.monotonic()
        if cached and cached[0] > now_mono:
            return cached[1]
        layers = await self.compute_layers(system_row)
        system = system_config(system_row)
        sim = self._layer(layers, layer)
        now = dt.datetime.now(UTC)
        confidence_cutoff = None
        if layers.degraded:
            confidence_cutoff = (
                pd.Timestamp(now) + pd.Timedelta(minutes=layers.local_horizon_min)
                if layers.local_horizon_min
                else pd.Timestamp(now)
            )
        periods, daily = self._periods_payload(
            sim, resolution, days, now, system.timezone, confidence_cutoff
        )
        today = self._today_block(
            sim,
            now,
            system.timezone,
            await self._measured_today(system.system_id, now, system.timezone),
        )
        daily = [entry for entry in daily if entry["date"] != today["date"]]
        daily.insert(0, {"date": today["date"], "wh": today["wh"]})
        weather_ts = await self.db.latest_weather_fetched_at(system.lat, system.lon)
        payload = {
            "system_id": system_id,
            "generated_at": now.isoformat(),
            "weather_fetched_at": ts_to_dt(weather_ts).isoformat() if weather_ts else now.isoformat(),
            "timezone": system.timezone,
            "resolution_minutes": resolution,
            "layer": layer,
            "kwp_total": system.kwp_total,
            "ac_limit_w": sim.paco,
            "layers_active": layers.active,
            "learning": layers.learning,
            "provenance": layers.provenance,
            "degraded": layers.degraded,
            "reason": layers.reason,
            "local_horizon_minutes": layers.local_horizon_min,
            "periods": periods,
            "daily": daily,
            "today": today,
        }
        self._cache[key] = (now_mono + CACHE_TTL_S, payload)
        return payload

    async def _measured_today(
        self, system_id: str, now: dt.datetime, timezone: str
    ) -> float | None:
        """Energy actually reported since local midnight, or ``None`` without telemetry."""
        tz = ZoneInfo(timezone)
        start = dt.datetime.combine(now.astimezone(tz).date(), dt.time(0), tzinfo=tz)
        rows = await self.db.telemetry(
            system_id, int(start.timestamp()), int(now.timestamp()) + 1
        )
        if not rows:
            return None
        return float(sum(float(row["energy_wh"]) for row in rows))

    def _today_block(
        self, sim: SimResult, now: dt.datetime, timezone: str, measured_wh: float | None
    ) -> dict:
        """Whole-day total for the current local date."""
        tz = ZoneInfo(timezone)
        day = now.astimezone(tz).date()
        start = pd.Timestamp(dt.datetime.combine(day, dt.time(0), tzinfo=tz))
        end = start + pd.Timedelta(days=1)
        energy = instant_energy(sim.pac).tz_convert(timezone)
        day_energy = energy[(energy.index >= start) & (energy.index < end)]
        simulated = float(day_energy.sum()) if len(day_energy) else 0.0
        now_ts = pd.Timestamp(now)
        elapsed_sim = (
            float(day_energy[day_energy.index <= now_ts].sum()) if len(day_energy) else 0.0
        )
        remaining = max(0.0, simulated - elapsed_sim)
        if measured_wh is None:
            return {
                "date": day.isoformat(),
                "wh": round(simulated, 3),
                "elapsed_wh": round(elapsed_sim, 3),
                "remaining_wh": round(remaining, 3),
                "measured_wh": None,
                "simulated_wh": round(simulated, 3),
                "source": "forecast",
            }
        return {
            "date": day.isoformat(),
            "wh": round(measured_wh + remaining, 3),
            "elapsed_wh": round(measured_wh, 3),
            "remaining_wh": round(remaining, 3),
            "measured_wh": round(measured_wh, 3),
            "simulated_wh": round(simulated, 3),
            "source": "measured+forecast",
        }

    def _periods_payload(
        self,
        sim: SimResult,
        resolution: int,
        days: int,
        now: dt.datetime,
        timezone: str,
        confidence_cutoff: pd.Timestamp | None = None,
    ) -> tuple[list[dict], list[dict]]:
        energy = instant_energy(sim.pac)
        full = aggregate_periods(energy, 15, 60) if resolution == 60 else energy
        start_floor = pd.Timestamp(now).floor(f"{resolution}min")
        end = pd.Timestamp(now + dt.timedelta(days=days))
        series = full[(full.index >= start_floor) & (full.index < end)]
        mean = series / (resolution / 60.0)
        elevation = sim.elevation.resample(f"{resolution}min", label="left", closed="left").max()
        sun = elevation.reindex(series.index).fillna(-90.0) > 0

        local_dates = pd.Series(
            pd.DatetimeIndex(series.index).tz_convert(timezone).date, index=series.index
        )
        keep: list[pd.Timestamp] = []
        for _, positions in local_dates.groupby(local_dates).groups.items():
            idx = pd.DatetimeIndex(positions)
            up = sun.reindex(idx).fillna(False).to_numpy()
            if not up.any():
                continue
            first = int(np.argmax(up))
            last = len(up) - 1 - int(np.argmax(up[::-1]))
            lo = max(first - 1, 0)
            hi = min(last + 1, len(idx) - 1)
            keep.extend(idx[lo : hi + 1].tolist())
        if keep:
            series = series.reindex(sorted(keep))
            mean = mean.reindex(series.index)
        periods = []
        for stamp, wh in series.items():
            if not np.isfinite(wh):
                continue
            local_start = stamp.tz_convert(timezone)
            local_end = (stamp + pd.Timedelta(minutes=resolution)).tz_convert(timezone)
            low = confidence_cutoff is not None and stamp > confidence_cutoff
            periods.append(
                {
                    "start": local_start.isoformat(),
                    "end": local_end.isoformat(),
                    "wh": float(wh),
                    "mean_watts": float(mean.loc[stamp]),
                    "confidence": "low" if low else None,
                }
            )
        # Daily totals cover whole local days: the current day keeps the hours that
        # have already elapsed, so "today" does not shrink towards zero in the evening.
        daily_series = local_daily_energy(full[full.index < end], timezone)
        today_local = pd.Timestamp(now).tz_convert(timezone).date()
        last_local = pd.Timestamp(end).tz_convert(timezone).date()
        daily = [
            {"date": day.isoformat(), "wh": float(value)}
            for day, value in daily_series.items()
            if np.isfinite(value) and today_local <= day <= last_local
        ]
        return periods, daily

    # -- snapshots ------------------------------------------------------
    async def snapshot_layers(self) -> int:
        written = 0
        for system_row in await self.db.list_systems():
            system = system_config(system_row)
            try:
                weather, fetched_at = await self.ensure_weather(system.lat, system.lon)
            except Exception as exc:  # noqa: BLE001 - an offline site must not stop the sweep
                LOG.warning("snapshot skipped for %s: %s", system.system_id, exc)
                continue
            layers = await self.compute_layers(system_row)
            if layers.degraded:
                # Benchmark rows must be the remote stack; the local layer is
                # snapshotted separately with source="local".
                continue
            now = dt.datetime.now(UTC)
            rows = []
            for name, sim in (
                ("physics_base", layers.base),
                ("physics", layers.physics),
                ("hybrid", layers.hybrid),
            ):
                energy = instant_energy(sim.pac)
                for stamp, wh in energy.items():
                    rows.append(
                        {
                            "system_id": system.system_id,
                            "fetched_at": int(fetched_at),
                            "period_start": int(stamp.timestamp()),
                            "period_min": 15,
                            "lead_hours": (stamp.timestamp() - fetched_at) / 3600.0,
                            "source": name,
                            "energy_wh": float(wh),
                            "mean_watts": float(wh / 0.25),
                        }
                    )
            await self.db.insert_runs(rows)
            written += len(rows)
        return written

    async def snapshot_local(self) -> int:
        """Store the local layer while remote weather is healthy (shadow mode)."""
        if not self.settings.enable_local_forecast:
            return 0
        written = 0
        now = dt.datetime.now(UTC)
        for system_row in await self.db.list_systems():
            if not system_row.get("local_fallback", 1):
                continue
            system = system_config(system_row)
            if await local_model.load(self.db, system.system_id) is None:
                continue
            try:
                layers = await self.local_layers(system_row, system)
            except Exception as exc:  # noqa: BLE001 - shadow work must never break a run
                LOG.warning("%s: local shadow failed: %s", system.system_id, exc)
                continue
            energy = instant_energy(layers.final.pac)
            rows = []
            for stamp, wh in energy.items():
                lead = (stamp.timestamp() - now.timestamp()) / 3600.0
                if lead > self.settings.local_forecast_hours or lead < 0:
                    continue
                rows.append(
                    {
                        "system_id": system.system_id,
                        "fetched_at": int(now.timestamp()),
                        "period_start": int(stamp.timestamp()),
                        "period_min": 15,
                        "lead_hours": lead,
                        "source": "local",
                        "energy_wh": float(wh),
                        "mean_watts": float(wh / 0.25),
                    }
                )
            await self.db.insert_runs(rows)
            written += len(rows)
        return written

    async def snapshot_forecast_solar(self) -> int:
        if not self.settings.enable_fs_shadow_benchmark:
            return 0
        from .forecast_solar_client import fetch_system_shadow

        written = 0
        for system_row in await self.db.list_systems():
            try:
                inverters = system_row.get("inverters") or []
                written += await fetch_system_shadow(
                    self.db,
                    system_row,
                    self.fs_client,
                    inverter_kw=(
                        sum(float(inv["ac_kw"]) for inv in inverters)
                        if inverters
                        else system_row.get("inverter_kw")
                    ),
                )
            except Exception as exc:
                LOG.warning("Forecast.Solar shadow failed for %s: %s", system_row["system_id"], exc)
        return written

    async def snapshot_all(self) -> dict[str, int]:
        layers = await self.snapshot_layers()
        fs_rows = await self.snapshot_forecast_solar()
        return {"forecast_rows": layers, "forecast_solar_rows": fs_rows}

    async def snapshot_nowcast(self) -> int:
        written = 0
        now = dt.datetime.now(UTC)
        for system_row in await self.db.list_systems():
            layers = await self.compute_layers(system_row)
            if not layers.nowcast_active:
                continue
            system = system_config(system_row)
            energy = instant_energy(layers.final.pac)
            rows = []
            for stamp, wh in energy.items():
                lead = (stamp.timestamp() - now.timestamp()) / 3600.0
                if lead > 3.0 or lead < 0:
                    continue
                rows.append(
                    {
                        "system_id": system.system_id,
                        "fetched_at": int(now.timestamp()),
                        "period_start": int(stamp.timestamp()),
                        "period_min": 15,
                        "lead_hours": lead,
                        "source": "final",
                        "energy_wh": float(wh),
                        "mean_watts": float(wh / 0.25),
                    }
                )
            await self.db.insert_runs(rows)
            written += len(rows)
        return written

    # -- Forecast.Solar compat ----------------------------------------
    async def compat_estimate(
        self,
        lat: float,
        lon: float,
        planes: list[tuple[float, float, float]],
        inverter_kw: float | None = None,
        damping_morning: float = 0.0,
        damping_evening: float = 0.0,
    ) -> dict:
        timezone = self.timezone_for(lat, lon)
        weather, fetched_at = await self.ensure_weather(lat, lon)
        horizon = await self.ensure_horizon(lat, lon)
        registered = await self._match_registered(lat, lon, planes, inverter_kw)
        matched_system = None
        layers_active: dict[str, bool] = {}
        if registered is not None:
            layers = await self.compute_layers(registered)
            sim = layers.final
            system = system_config(registered)
            matched_system = registered["system_id"]
            layers_active = layers.active
        else:
            system = SystemConfig(
                system_id="compat",
                lat=lat,
                lon=lon,
                elevation_m=0.0,
                inverters=(
                    (InverterConfig("main", inverter_kw),)
                    if inverter_kw is not None
                    else ()
                ),
                timezone=timezone,
                damping_morning=damping_morning,
                damping_evening=damping_evening,
                planes=tuple(
                    PlaneConfig(
                        dec=dec,
                        az_fs=az,
                        kwp=kwp,
                        inverter_id="main" if inverter_kw is not None else None,
                    )
                    for dec, az, kwp in planes
                ),
            )
            sim = simulate_system(weather, system, horizon, calibration_c=1.0)
        return self._compat_payload(
            sim,
            system,
            timezone,
            matched_system=matched_system,
            layers_active=layers_active,
        )

    async def _match_registered(
        self, lat: float, lon: float, planes: list[tuple[float, float, float]], inverter_kw
    ) -> dict | None:
        wanted = sorted((round(d, 1), round(a, 1), round(k, 3)) for d, a, k in planes)
        for system in await self.db.list_systems():
            if abs(system["lat"] - lat) > 1e-6 or abs(system["lon"] - lon) > 1e-6:
                continue
            have = sorted(
                (round(p["dec"], 1), round(p["az_fs"], 1), round(p["kwp"], 3))
                for p in system["planes"]
            )
            if have != wanted:
                continue
            registered_inverters = system.get("inverters") or []
            if len(registered_inverters) == 1:
                registered_inverter = float(registered_inverters[0]["ac_kw"])
            elif not registered_inverters:
                registered_inverter = system.get("inverter_kw")
            else:
                registered_inverter = None
            if inverter_kw is not None:
                if registered_inverter is None or abs(
                    float(inverter_kw) - float(registered_inverter)
                ) > 1e-6:
                    continue
            elif len(registered_inverters) > 1:
                continue
            return system
        return None

    def _compat_payload(
        self,
        sim: SimResult,
        system: SystemConfig,
        timezone: str,
        matched_system: str | None = None,
        layers_active: dict[str, bool] | None = None,
    ) -> dict:
        tz = ZoneInfo(timezone)
        now = dt.datetime.now(UTC)
        energy = instant_energy(sim.pac)
        local_energy = energy.tz_convert(timezone)
        watts: dict[str, float] = {}
        wh_period: dict[str, float] = {}
        wh_cumulative: dict[str, float] = {}
        wh_day: dict[str, float] = {}
        inverter_cap = (
            sum(inverter.ac_kw for inverter in system.inverters) * 1000.0
            if system.inverters
            else None
        )

        for offset in (0, 1):
            day = (now.astimezone(tz) + dt.timedelta(days=offset)).date()
            day_start = pd.Timestamp(dt.datetime.combine(day, dt.time(0), tzinfo=tz))
            day_end = day_start + pd.Timedelta(days=1)
            day_energy = local_energy[
                (local_energy.index >= day_start) & (local_energy.index < day_end)
            ]
            if day_energy.empty:
                wh_day[day.isoformat()] = 0.0
                continue
            hours = day_energy.resample("1h", label="left", closed="left").sum()
            if inverter_cap is not None:
                hours = hours.clip(upper=inverter_cap)
            sunrise, sunset = self._sun_events(day, system)
            if sunrise is None or sunset is None:
                continue

            day_keys: list[tuple[pd.Timestamp, float, float]] = []
            sunrise_local = pd.Timestamp(sunrise).tz_convert(timezone) if sunrise.tzinfo else pd.Timestamp(sunrise)
            sunset_local = pd.Timestamp(sunset).tz_convert(timezone) if sunset.tzinfo else pd.Timestamp(sunset)

            if sunrise_local > day_start:
                day_keys.append((sunrise_local, 0.0, 0.0))

            labels = {sunrise_local}
            for hour_start, wh in hours.items():
                hour_end = hour_start + pd.Timedelta(hours=1)
                if hour_end <= sunrise_local or hour_start >= sunset_local:
                    continue
                if float(wh) <= 0:
                    continue
                if hour_end <= sunset_local:
                    day_keys.append((hour_end, float(wh), float(wh)))
                else:
                    day_keys.append((sunset_local, float(wh), 0.0))
                labels.add(day_keys[-1][0])
            if sunset_local not in labels:
                day_keys.append((sunset_local, 0.0, 0.0))

            day_keys.sort(key=lambda item: item[0])
            cumulative = 0.0
            for stamp, value, watts_value in day_keys:
                key = stamp.strftime("%Y-%m-%d %H:%M:%S")
                wh_period[key] = round(value, 3)
                watts[key] = round(watts_value, 3)
                cumulative += value
                wh_cumulative[key] = round(cumulative, 3)
            wh_day[day.isoformat()] = round(cumulative, 3)

        info = {
            "latitude": system.lat,
            "longitude": system.lon,
            "distance": 0,
            "place": "SolarCast",
            "timezone": timezone,
            "time": now.astimezone(tz).isoformat(),
            "time_utc": now.isoformat(),
            "solarcast": {
                "matched_system": matched_system,
                "layers_active": dict(layers_active or {}),
            },
        }
        return {
            "result": {
                "watts": watts,
                "watt_hours_period": wh_period,
                "watt_hours": wh_cumulative,
                "watt_hours_day": wh_day,
            },
            "message": {
                "code": 0,
                "type": "success",
                "text": "",
                "info": info,
                "ratelimit": {
                    "zone": "local",
                    "period": 3600,
                    "limit": 1000,
                    "remaining": 999,
                },
            },
        }

    @staticmethod
    def _sun_events(day: dt.date, system: SystemConfig):
        tz = ZoneInfo(system.timezone)
        noon = pd.Timestamp(dt.datetime.combine(day, dt.time(12), tzinfo=tz)).tz_convert("UTC")
        try:
            events = pvlib.solarposition.sun_rise_set_transit_spa(
                pd.DatetimeIndex([noon]), system.lat, system.lon
            )
        except Exception:
            return None, None
        sunrise = events["sunrise"].iloc[0]
        sunset = events["sunset"].iloc[0]
        if pd.isna(sunrise) or pd.isna(sunset):
            return None, None
        return sunrise, sunset
