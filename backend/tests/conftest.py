import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pvlib
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import Database
from app.core.openmeteo_client import CANONICAL_COLUMNS, OpenMeteoResult
from app.core.pvgis_client import flat_fallback
from config.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_weather(lat: float = 44.8125, lon: float = 20.4612, days: int = 10) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    index = pd.date_range(now - pd.Timedelta(days=2), periods=days * 96, freq="15min")
    clear = pvlib.location.Location(lat, lon, "UTC").get_clearsky(index, model="ineichen")
    frame = pd.DataFrame(index=index)
    frame["ghi"] = clear["ghi"].to_numpy() * 0.85
    frame["dni"] = clear["dni"].to_numpy() * 0.85
    frame["dhi"] = clear["dhi"].to_numpy() * 0.85
    frame["temp_2m_c"] = 15.0
    frame["wind_10m_ms"] = 2.0
    frame["cloud_total"] = 10.0
    frame["cloud_low"] = 5.0
    frame["cloud_mid"] = 3.0
    frame["cloud_high"] = 2.0
    frame["precip_mm"] = 0.0
    frame["snowfall_cm"] = 0.0
    frame["snow_depth_m"] = 0.0
    return frame[CANONICAL_COLUMNS]


class StubWeather:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    async def get(self, lat: float, lon: float, forecast_days: int = 7) -> OpenMeteoResult:
        self.calls += 1
        return OpenMeteoResult(self.frame, 100.0, ("stub",), {})

    def invalidate(self) -> None:
        pass


async def flat_horizon(lat, lon, base_url=None):
    return flat_fallback()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        database_path=tmp_path / "test.db",
        api_token=None,
        compat_public=True,
        scheduler_tz="UTC",
        systems_file=tmp_path / "missing.yaml",
        enable_fs_shadow_benchmark=False,
    )


@pytest.fixture
def weather_stub() -> StubWeather:
    return StubWeather(synthetic_weather())


@pytest.fixture
async def api(settings, weather_stub):
    from app.main import create_app

    app = create_app(
        settings=settings,
        weather_client=weather_stub,
        horizon_fetcher=flat_horizon,
        enable_scheduler=False,
        db=Database(settings.database_path),
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            client.app = app
            yield client


@pytest.fixture
def openmeteo_payload() -> dict:
    return json.loads((FIXTURES / "openmeteo_multimodel.json").read_text())


@pytest.fixture
def fs_fixture() -> dict:
    return json.loads((FIXTURES / "forecast_solar_estimate.json").read_text())


def system_body(planes=None, **overrides) -> dict:
    body = {
        "name": "Home",
        "lat": 44.8125,
        "lon": 20.4612,
        "timezone": "Europe/Belgrade",
        "inverter_kw": 5.0,
        "system_loss": 0.10,
        "planes": planes or [{"dec": 30, "az": 0, "kwp": 5.0}],
    }
    body.update(overrides)
    return body
