"""Async client for the SolarCast backend."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

TIMEOUT = aiohttp.ClientTimeout(total=30)


class SolarCastError(Exception):
    """Backend communication error."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class SolarCastAuthError(SolarCastError):
    """Authentication rejected."""


class SolarCastApi:
    def __init__(
        self, session: aiohttp.ClientSession, base_url: str, token: str | None = None
    ) -> None:
        self._session = session
        self.base_url = base_url.rstrip("/")
        self._token = token

    @property
    def _headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    async def _request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}{path}"
        request_headers = {**self._headers, **(headers or {})}
        try:
            async with asyncio.timeout(35):
                response = await self._session.request(
                    method, url, headers=request_headers, timeout=TIMEOUT, **kwargs
                )
                if response.status == 401:
                    raise SolarCastAuthError("invalid token", 401)
                if response.status >= 400:
                    body = await response.text()
                    raise SolarCastError(
                        f"backend returned {response.status}: {body[:200]}", response.status
                    )
                if response.status == 204:
                    return None
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError) as exc:
            raise SolarCastError(str(exc)) from exc

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/api/v1/health")

    async def put_system(self, system_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("PUT", f"/api/v1/systems/{system_id}", json=payload)

    async def forecast(
        self, system_id: str, resolution: int = 60, days: int = 7, layer: str = "final"
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/systems/{system_id}/forecast",
            params={"resolution": resolution, "days": days, "layer": layer},
        )

    async def benchmark(self, system_id: str, days: int = 14) -> dict[str, Any]:
        return await self._request(
            "GET", f"/api/v1/systems/{system_id}/benchmark", params={"days": days}
        )

    async def telemetry(self, system_id: str, items: list[dict], origin: str = "push") -> dict:
        return await self._request(
            "POST",
            f"/api/v1/systems/{system_id}/telemetry",
            json={"items": items, "origin": origin},
        )

    async def retrain(self, system_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/api/v1/systems/{system_id}/retrain")

    async def put_horizon(self, system_id: str, csv_text: str) -> None:
        await self._request(
            "PUT",
            f"/api/v1/systems/{system_id}/horizon",
            data=csv_text,
            headers={"Content-Type": "text/csv"},
        )

    async def delete_horizon(self, system_id: str) -> None:
        await self._request("DELETE", f"/api/v1/systems/{system_id}/horizon")
