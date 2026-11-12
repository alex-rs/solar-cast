from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

import yaml
from fastapi import FastAPI

from config.settings import Settings, get_settings

from .api import routes_benchmark, routes_compat, routes_forecast, routes_health, routes_systems, routes_telemetry
from .core.database import Database
from .core.openmeteo_client import OpenMeteoClient
from .core.pipeline import ForecastPipeline

LOG = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def _load_static_systems(db: Database, pipeline: ForecastPipeline, path) -> None:
    if not path.exists():
        return
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        LOG.error("cannot parse %s: %s", path, exc)
        return
    for entry in payload.get("systems", []):
        system_id = entry["system_id"]
        timezone = entry.get("timezone") or pipeline.timezone_for(entry["lat"], entry["lon"])
        system = {
            "system_id": system_id,
            "name": entry.get("name", system_id),
            "lat": entry["lat"],
            "lon": entry["lon"],
            "elevation_m": entry.get("elevation_m"),
            "timezone": timezone,
            "inverter_kw": entry.get("inverter_kw"),
            "system_loss": entry.get("system_loss", 0.10),
            "gamma_pdc": entry.get("gamma_pdc", -0.004),
            "damping_morning": entry.get("damping_morning", 0.0),
            "damping_evening": entry.get("damping_evening", 0.0),
            "local_fallback": bool(entry.get("local_fallback", True)),
        }
        raw_inverters = entry.get("inverters")
        if raw_inverters is not None:
            if any(plane.get("inverter") is None for plane in entry.get("planes", [])):
                LOG.error(
                    "static system %s: every plane must reference an inverter",
                    system_id,
                )
                continue
            inverter_rows = [
                {"inverter_id": inverter["id"], "ac_kw": inverter["ac_kw"]}
                for inverter in raw_inverters
            ]
            planes = [
                {
                    "dec": plane["dec"],
                    "az_fs": plane["az"],
                    "kwp": plane["kwp"],
                    "inverter_id": plane.get("inverter"),
                }
                for plane in entry.get("planes", [])
            ]
            if planes:
                try:
                    await db.upsert_system(system, planes, inverter_rows)
                except Exception as exc:
                    LOG.error("cannot load static system %s: %s", system_id, exc)
                    continue
                LOG.info("loaded static system %s", system_id)
            continue
        planes = [
            {"dec": plane["dec"], "az_fs": plane["az"], "kwp": plane["kwp"]}
            for plane in entry.get("planes", [])
        ]
        if planes:
            try:
                await db.upsert_system(system, planes)
            except Exception as exc:
                LOG.error("cannot load static system %s: %s", system_id, exc)
                continue
            LOG.info("loaded static system %s", system_id)


def create_app(
    settings: Settings | None = None,
    weather_client: OpenMeteoClient | None = None,
    horizon_fetcher=None,
    enable_scheduler: bool = True,
    db: Database | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    _configure_logging(app_settings.log_level)
    database = db or Database(app_settings.database_path)
    mcp_server = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_settings.ensure_dirs()
        await database.initialize()
        app.state.settings = app_settings
        app.state.db = database
        app.state.pipeline = ForecastPipeline(
            database,
            app_settings,
            weather_client=weather_client,
            horizon_fetcher=horizon_fetcher,
        )
        await _load_static_systems(database, app.state.pipeline, app_settings.systems_file)
        async with AsyncExitStack() as stack:
            if mcp_server is not None:
                await stack.enter_async_context(mcp_server.session_manager.run())
            scheduler = None
            if enable_scheduler:
                from .jobs.scheduler import create_scheduler, resolve_scheduler_tz

                timezone = resolve_scheduler_tz(app_settings, await database.list_systems())
                scheduler = create_scheduler(app_settings, database, app.state.pipeline, timezone)
                scheduler.start()
                app.state.scheduler = scheduler
                app.state.scheduler_tz = str(scheduler.timezone)
                LOG.info("scheduler started in timezone %s", scheduler.timezone)
            yield
            if scheduler is not None:
                scheduler.shutdown(wait=False)
        await database.close()

    app = FastAPI(title="SolarCast", version="1.3.0", lifespan=lifespan)
    app.include_router(routes_systems.router)
    app.include_router(routes_forecast.router)
    app.include_router(routes_compat.router)
    app.include_router(routes_telemetry.router)
    app.include_router(routes_benchmark.router)
    app.include_router(routes_health.router)
    if app_settings.enable_mcp:
        from mcp.server.transport_security import TransportSecuritySettings

        from .mcp_server import MCPAuthMiddleware, build_mcp_server

        mcp_server = build_mcp_server(app)
        mcp_app = mcp_server.streamable_http_app(
            json_response=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            ),
        )
        if app_settings.api_token:
            mcp_app.add_middleware(MCPAuthMiddleware, token=app_settings.api_token)
        app.mount("/", mcp_app)
    return app


app = create_app()
