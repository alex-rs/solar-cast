"""Optional read-only MCP server exposing SolarCast forecasts."""

from __future__ import annotations

import hmac
from typing import Any, Literal

from fastapi import FastAPI
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class MCPAuthMiddleware:
    """Pure ASGI bearer-token gate for the MCP endpoint.

    MCP responses can stream, so this must not buffer bodies through
    ``BaseHTTPMiddleware``.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            raw = headers.get(b"authorization", b"").decode("latin-1")
            scheme, _, token = raw.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(token, self.token):
                response = JSONResponse(
                    {"detail": "invalid token"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_mcp_server(app: FastAPI) -> MCPServer:
    """Build the read-only SolarCast MCP server bound to a running FastAPI app."""
    server = MCPServer(
        "SolarCast",
        version="1.3.0",
        instructions=(
            "Read-only access to SolarCast PV forecasts. Call list_systems first, "
            "then get_forecast. Periods are ISO 8601 with UTC offset; wh is period "
            "energy and daily is local-calendar-date energy."
        ),
    )

    @server.tool()
    async def list_systems() -> dict[str, Any]:
        """List configured PV systems available for forecasting."""
        systems = await app.state.db.list_systems()
        return {
            "systems": [
                {
                    "system_id": system["system_id"],
                    "name": system["name"],
                    "timezone": system["timezone"],
                    "lat": float(system["lat"]),
                    "lon": float(system["lon"]),
                    "kwp_total": sum(float(plane["kwp"]) for plane in system["planes"]),
                }
                for system in systems
            ]
        }

    @server.tool()
    async def get_forecast(
        system_id: str,
        days: int = 3,
        resolution: Literal[15, 60] = 60,
        layer: Literal["final", "hybrid", "physics", "physics_base"] = "final",
    ) -> dict[str, Any]:
        """Return the SolarCast forecast for one system. Defaults: 3 days, 60-minute periods, final layer. Use layer=physics for the unlearned physical baseline."""
        if not 1 <= days <= 7:
            raise ToolError("days must be between 1 and 7")
        try:
            return await app.state.pipeline.get_forecast(system_id, resolution, days, layer)
        except KeyError:
            raise ToolError(f"unknown system: {system_id}") from None

    return server
