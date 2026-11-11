import json
from contextlib import asynccontextmanager

import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.core.database import Database
from app.main import create_app
from config.settings import Settings
from tests.conftest import flat_horizon, system_body

AUTH = {"Authorization": "Bearer secret"}


def mcp_settings(settings) -> Settings:
    data = settings.model_dump()
    data.update(enable_mcp=True, api_token="secret")
    return Settings(**data)


@asynccontextmanager
async def running_app(settings, weather_stub, headers):
    """Run an MCP-enabled app and yield a client bound to it.

    The lifespan and client are entered in the test's own task: the MCP session
    manager uses an AnyIO task group that must exit in the task that entered it.
    """
    app = create_app(
        settings=settings,
        weather_client=weather_stub,
        horizon_fetcher=flat_horizon,
        enable_scheduler=False,
        db=Database(settings.database_path),
    )
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test", headers=headers
        ) as client:
            yield client


async def test_mcp_disabled_returns_404(api):
    response = await api.post("/mcp")
    assert response.status_code == 404


async def test_mcp_requires_bearer_token(settings, weather_stub):
    async with running_app(mcp_settings(settings), weather_stub, {}) as client:
        response = await client.post("/mcp")
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid token"}
        assert response.headers["www-authenticate"] == "Bearer"
        response = await client.post("/mcp", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401


async def test_mcp_tools(settings, weather_stub):
    async with running_app(mcp_settings(settings), weather_stub, AUTH) as client:
        created = await client.put("/api/v1/systems/home", json=system_body())
        assert created.status_code == 200, created.text

        async with streamable_http_client(
            "http://test/mcp", http_client=client
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "list_systems",
                    "get_forecast",
                }

                listed = await session.call_tool("list_systems", {})
                assert listed.is_error is False, listed.content
                systems = json.loads(listed.content[0].text)["systems"]
                assert [system["system_id"] for system in systems] == ["home"]
                assert systems[0]["kwp_total"] == pytest.approx(5.0)

                forecast = await session.call_tool(
                    "get_forecast", {"system_id": "home", "days": 1, "resolution": 60}
                )
                assert forecast.is_error is False, forecast.content
                payload = json.loads(forecast.content[0].text)
                assert payload["system_id"] == "home"
                assert payload["resolution_minutes"] == 60
                assert payload["periods"]
                assert len(payload["periods"]) <= 24

                missing = await session.call_tool(
                    "get_forecast", {"system_id": "missing"}
                )
                assert missing.is_error is True
                assert "unknown system: missing" in missing.content[0].text

                invalid_days = await session.call_tool(
                    "get_forecast", {"system_id": "home", "days": 8}
                )
                assert invalid_days.is_error is True
                assert "days must be between 1 and 7" in invalid_days.content[0].text
