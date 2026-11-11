import pytest

from custom_components.localsolarcast.api import (
    LocalSolarCastAuthError,
    LocalSolarCastError,
)
from custom_components.localsolarcast.const import (
    CONF_AZ,
    CONF_DEC,
    CONF_KWP,
    CONF_NAME,
    CONF_PLANES,
    CONF_SYSTEM_ID,
    CONF_URL,
    DOMAIN,
)

pytestmark = pytest.mark.asyncio


class FakeApi:
    put_calls: list = []

    def __init__(self, session, url, token=None):
        self.url = url
        self.token = token

    async def health(self):
        if "bad-host" in self.url:
            raise LocalSolarCastError("cannot connect")
        if self.token == "bad":
            raise LocalSolarCastAuthError("invalid")
        return {"status": "ok"}

    async def put_system(self, system_id, payload):
        FakeApi.put_calls.append((system_id, payload))
        return {"system_id": system_id}


@pytest.fixture(autouse=True)
def patch_api(monkeypatch):
    monkeypatch.setattr(
        "custom_components.localsolarcast.config_flow.LocalSolarCastApi", FakeApi
    )
    FakeApi.put_calls = []


async def _start(hass, url="http://backend:8000", token=""):
    return await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )


async def test_happy_path_with_four_planes(hass):
    result = await _start(hass)
    assert result["step_id"] == "user"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_URL: "http://backend:8000", "token": ""}
    )
    assert result["step_id"] == "system"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        user_input={
            CONF_NAME: "Home",
            CONF_SYSTEM_ID: "home",
            "lat": 44.8125,
            "lon": 20.4612,
            "inverter_kw": 5.0,
            "system_loss": 0.1,
            "damping_morning": 0.0,
            "damping_evening": 0.0,
        },
    )
    assert result["step_id"] == "plane"
    for index in range(4):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                CONF_DEC: 30,
                CONF_AZ: 0,
                CONF_KWP: 1.25,
                "add_another": index < 3,
            },
        )
    assert result["step_id"] == "telemetry"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], user_input={})
    assert result["type"] == "create_entry"
    assert len(result["options"][CONF_PLANES]) == 4
    assert result["data"][CONF_SYSTEM_ID] == "home"
    assert FakeApi.put_calls[-1][1]["planes"][0]["dec"] == 30


async def test_four_plane_limit_ignores_extra_add(hass):
    result = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_URL: "http://backend:8000", "token": ""}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        user_input={
            CONF_NAME: "Home",
            CONF_SYSTEM_ID: "home",
            "lat": 44.8,
            "lon": 20.4,
            "inverter_kw": None,
            "system_loss": 0.1,
            "damping_morning": 0.0,
            "damping_evening": 0.0,
        },
    )
    for index in range(4):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={CONF_DEC: 30, CONF_AZ: 0, CONF_KWP: 1.0, "add_another": True},
        )
    assert result["step_id"] == "telemetry"


async def test_bad_url_and_token(hass):
    result = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_URL: "http://bad-host:8000", "token": ""}
    )
    assert result["errors"]["base"] == "cannot_connect"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_URL: "http://backend:8000", "token": "bad"}
    )
    assert result["errors"]["base"] == "invalid_auth"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_URL: "http://backend:8000", "token": ""}
    )
    assert result["step_id"] == "system"


async def test_options_flow_re_puts(hass, monkeypatch):
    from tests_ha.fixtures import make_entry

    monkeypatch.setattr(
        "custom_components.localsolarcast.config_flow.LocalSolarCastApi", FakeApi
    )
    entry = make_entry()
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["step_id"] == "init"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_NAME: "Cabin",
            "lat": 45.0,
            "lon": 20.0,
            "inverter_kw": 4.0,
            "system_loss": 0.12,
            "damping_morning": 0.0,
            "damping_evening": 0.0,
            "planes_count": 2,
        },
    )
    assert result["step_id"] == "planes"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            "dec_0": 30,
            "az_0": 0,
            "kwp_0": 2.0,
            "dec_1": 35,
            "az_1": 90,
            "kwp_1": 2.0,
        },
    )
    assert result["step_id"] == "telemetry"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"soc_threshold": 98}
    )
    assert result["type"] == "create_entry"
    assert FakeApi.put_calls
    payload = FakeApi.put_calls[-1][1]
    assert payload["name"] == "Cabin"
    assert len(payload["planes"]) == 2
    assert payload["planes"][1]["az"] == 90
