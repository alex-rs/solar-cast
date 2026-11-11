import numpy as np
import pytest

from tests.conftest import system_body

CSV = """azimuth_fs,elevation
-180,0
-90,30
0,5
90,0
175,0
"""


@pytest.mark.asyncio
async def test_horizon_upload_get_delete(api):
    assert (await api.put("/api/v1/systems/home", json=system_body())).status_code == 200
    response = await api.put(
        "/api/v1/systems/home/horizon",
        content=CSV,
        headers={"Content-Type": "text/csv"},
    )
    assert response.status_code == 204, response.text

    response = await api.get("/api/v1/systems/home/horizon")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["system_id"] == "home"
    assert payload["source"] == "user"
    assert [point["azimuth_deg"] for point in payload["points"]] == [0.0, 90.0, 180.0, 270.0, 355.0]
    assert [point["elevation_deg"] for point in payload["points"]] == [0.0, 30.0, 5.0, 0.0, 0.0]

    pipeline = api.app.state.pipeline
    db = api.app.state.db
    row = await db.get_system("home")
    horizon = await pipeline.ensure_horizon_for_system(row)
    assert horizon.source == "user"
    assert horizon.lookup(np.array([90.0]))[0] == pytest.approx(30.0)

    assert (await api.delete("/api/v1/systems/home/horizon")).status_code == 204
    assert (await api.get("/api/v1/systems/home/horizon")).status_code == 404
    assert (await api.delete("/api/v1/systems/home/horizon")).status_code == 404


@pytest.mark.asyncio
async def test_horizon_validation_errors(api):
    assert (await api.put("/api/v1/systems/home", json=system_body())).status_code == 200
    cases = [
        ("elevation,azimuth_fs\n0,0\n", "horizon must have header 'azimuth_fs,elevation'"),
        ("azimuth_fs,elevation\n0,0\n90,1\n", "horizon needs 3-360 points"),
        (
            "azimuth_fs,elevation\n0,0\n90,1\n190,0\n",
            "azimuth_fs must be between -180 and 180",
        ),
        (
            "azimuth_fs,elevation\n0,0\n90,1\n180,91\n",
            "elevation must be between -5 and 90",
        ),
        ("azimuth_fs,elevation\n0,0\n90\n180,0\n", "horizon row must contain two numbers"),
    ]
    for text, detail in cases:
        response = await api.put(
            "/api/v1/systems/home/horizon",
            content=text,
            headers={"Content-Type": "text/csv"},
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"] == detail


@pytest.mark.asyncio
async def test_horizon_unknown_system(api):
    assert (
        await api.put(
            "/api/v1/systems/nope/horizon",
            content=CSV,
            headers={"Content-Type": "text/csv"},
        )
    ).status_code == 404
    assert (await api.get("/api/v1/systems/nope/horizon")).status_code == 404
