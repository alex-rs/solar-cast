import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx

from app.core.pvgis_client import fetch_horizon, flat_fallback, parse_horizon

FIXTURE = Path(__file__).parent / "fixtures" / "pvgis_horizon.json"


def test_parse_deduplicates_pm_180():
    horizon = parse_horizon(json.loads(FIXTURE.read_text()))
    assert horizon.source == "pvgis"
    assert len(horizon.azimuth) == len(set(horizon.azimuth.tolist()))
    assert 0.0 in horizon.azimuth
    assert horizon.azimuth.min() >= 0.0 and horizon.azimuth.max() < 360.0
    assert np.all(np.diff(horizon.azimuth) > 0)
    # Sources fixture has 49 entries; the +180/-180 pair collapses to one.
    assert len(horizon.azimuth) == 48


def test_horizon_interpolation_wraps():
    horizon = flat_fallback()
    horizon.elevation[:] = [1.0, 2.0, 3.0, 4.0]
    values = horizon.lookup(np.array([-10.0, 0.0, 355.0, 360.0]))
    assert values[0] == pytest.approx(1.3333333333)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(1.1666666667)
    assert values[3] == pytest.approx(1.0)


@respx.mock
@pytest.mark.asyncio
async def test_fetch_uses_v5_3_and_returns_source():
    body = json.loads(FIXTURE.read_text())
    route = respx.get(
        "https://re.jrc.ec.europa.eu/api/v5_3/printhorizon",
        params={"lat": 44.812, "lon": 20.461, "outputformat": "json"},
    ).mock(return_value=httpx.Response(200, json=body))
    horizon = await fetch_horizon(44.8125, 20.4612)
    assert route.called
    assert horizon.source == "pvgis_v5_3"


@respx.mock
@pytest.mark.asyncio
async def test_fetch_falls_back_to_flat_without_crashing():
    respx.get("https://re.jrc.ec.europa.eu/api/v5_3/printhorizon").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://re.jrc.ec.europa.eu/api/v5_2/printhorizon").mock(
        return_value=httpx.Response(503)
    )
    horizon = await fetch_horizon(0.0, 0.0, retries=0)
    assert horizon.source == "flat_fallback"
    assert horizon.elevation.sum() == 0.0


@respx.mock
@pytest.mark.asyncio
async def test_fetch_falls_back_to_v5_2():
    body = json.loads(FIXTURE.read_text())
    respx.get("https://re.jrc.ec.europa.eu/api/v5_3/printhorizon").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://re.jrc.ec.europa.eu/api/v5_2/printhorizon").mock(
        return_value=httpx.Response(200, json=body)
    )
    horizon = await fetch_horizon(44.8125, 20.4612, retries=0)
    assert horizon.source == "pvgis_v5_2"
