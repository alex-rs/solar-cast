"""PVGIS terrain horizon client."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
import numpy as np

from .timeconv import fs_az_to_pvlib

LOG = logging.getLogger(__name__)

PVGIS_VERSIONS = ("v5_3", "v5_2")
_TIMEOUT = 20.0


@dataclass(frozen=True)
class Horizon:
    azimuth: np.ndarray
    elevation: np.ndarray
    source: str

    def lookup(self, sun_azimuth: np.ndarray) -> np.ndarray:
        """Interpolate horizon elevation, wrapping at 360 degrees."""
        return np.interp(
            np.asarray(sun_azimuth, dtype="float64"),
            self.azimuth,
            self.elevation,
            period=360.0,
        )


def parse_horizon(payload: dict) -> Horizon:
    profile = payload["outputs"]["horizon_profile"]
    points: dict[float, float] = {}
    for entry in profile:
        az = round(fs_az_to_pvlib(float(entry["A"])), 3)
        if az in points:
            continue
        points[az] = float(entry["H_hor"])
    if not points:
        raise ValueError("empty PVGIS horizon profile")
    azimuth = np.array(sorted(points), dtype="float64")
    elevation = np.array([points[az] for az in azimuth], dtype="float64")
    return Horizon(azimuth=azimuth, elevation=elevation, source="pvgis")


def flat_fallback() -> Horizon:
    return Horizon(
        azimuth=np.array([0.0, 90.0, 180.0, 270.0]),
        elevation=np.zeros(4),
        source="flat_fallback",
    )


def user_horizon(azimuth_fs: list[float], elevation: list[float]) -> Horizon:
    """Build a horizon from a user CSV-style pair of arrays (Forecast.Solar azimuth)."""
    points: dict[float, float] = {}
    for az, el in zip(azimuth_fs, elevation):
        points[round(fs_az_to_pvlib(float(az)), 3)] = float(el)
    azimuth = np.array(sorted(points), dtype="float64")
    return Horizon(
        azimuth=azimuth,
        elevation=np.array([points[a] for a in azimuth], dtype="float64"),
        source="user",
    )


async def fetch_horizon(
    lat: float,
    lon: float,
    *,
    base_url: str = "https://re.jrc.ec.europa.eu/api",
    client: httpx.AsyncClient | None = None,
    retries: int = 2,
) -> Horizon:
    """Fetch the terrain horizon, trying PVGIS v5_3 then v5_2.

    Never raises for upstream failures; falls back to a flat horizon.
    """
    lat_r, lon_r = round(lat, 3), round(lon, 3)
    own = client is None
    http = client or httpx.AsyncClient(timeout=_TIMEOUT)
    try:
        for version in PVGIS_VERSIONS:
            for attempt in range(retries + 1):
                try:
                    response = await http.get(
                        f"{base_url}/{version}/printhorizon",
                        params={"lat": lat_r, "lon": lon_r, "outputformat": "json"},
                    )
                except httpx.HTTPError as exc:
                    LOG.warning("PVGIS %s request failed: %s", version, exc)
                    break
                if response.status_code == 200:
                    try:
                        horizon = parse_horizon(response.json())
                    except (ValueError, KeyError, TypeError) as exc:
                        LOG.warning("PVGIS %s payload invalid: %s", version, exc)
                        break
                    LOG.info(
                        "PVGIS horizon %s: %d points for %s,%s",
                        version,
                        len(horizon.azimuth),
                        lat_r,
                        lon_r,
                    )
                    return Horizon(
                        azimuth=horizon.azimuth,
                        elevation=horizon.elevation,
                        source=f"pvgis_{version}",
                    )
                if response.status_code not in (404, 429, 500, 502, 503, 504):
                    LOG.warning("PVGIS %s returned %s", version, response.status_code)
                    break
                await asyncio.sleep(0.5 * (2**attempt))
        LOG.warning("PVGIS unavailable for %s,%s; using flat horizon", lat_r, lon_r)
        return flat_fallback()
    finally:
        if own:
            await http.aclose()
