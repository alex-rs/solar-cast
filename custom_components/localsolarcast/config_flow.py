from __future__ import annotations

import logging
import re
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import slugify

from .api import LocalSolarCastApi, LocalSolarCastAuthError, LocalSolarCastError
from .const import (
    CONF_ADD_ANOTHER,
    CONF_AZ,
    CONF_BATTERY_SOC_ENTITY,
    CONF_CURTAILMENT_ENTITY,
    CONF_DAMPING_EVENING,
    CONF_DAMPING_MORNING,
    CONF_DEC,
    CONF_ENERGY_ENTITY,
    CONF_EXPORT_LIMITED,
    CONF_INVERTER_KW,
    CONF_INVERTER_TEMP_ENTITY,
    CONF_KWP,
    CONF_LAT,
    CONF_LON,
    CONF_NAME,
    CONF_PLANES,
    CONF_PLANES_COUNT,
    CONF_POWER_ENTITY,
    CONF_SOC_THRESHOLD,
    CONF_SYSTEM_ID,
    CONF_SYSTEM_LOSS,
    CONF_TOKEN,
    CONF_URL,
    DEFAULT_SOC_THRESHOLD,
    DEFAULT_SYSTEM_LOSS,
    DEFAULT_URL,
    DOMAIN,
)

LOG = logging.getLogger(__name__)
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _system_schema(defaults: dict[str, Any], *, with_planes_count: bool = False) -> vol.Schema:
    fields: dict[Any, Any] = {
        vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, "Home")): str,
        vol.Required(CONF_LAT, default=defaults.get(CONF_LAT, 0.0)): vol.All(
            vol.Coerce(float), vol.Range(min=-90, max=90)
        ),
        vol.Required(CONF_LON, default=defaults.get(CONF_LON, 0.0)): vol.All(
            vol.Coerce(float), vol.Range(min=-180, max=180)
        ),
        vol.Optional(CONF_INVERTER_KW, default=defaults.get(CONF_INVERTER_KW)): vol.Any(
            None, "", vol.All(vol.Coerce(float), vol.Range(min=0.01))
        ),
        vol.Optional(
            CONF_SYSTEM_LOSS, default=defaults.get(CONF_SYSTEM_LOSS, DEFAULT_SYSTEM_LOSS)
        ): vol.All(vol.Coerce(float), vol.Range(min=0, max=0.99)),
        vol.Optional(
            CONF_DAMPING_MORNING, default=defaults.get(CONF_DAMPING_MORNING, 0.0)
        ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
        vol.Optional(
            CONF_DAMPING_EVENING, default=defaults.get(CONF_DAMPING_EVENING, 0.0)
        ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
    }
    if with_planes_count:
        fields[
            vol.Required(CONF_PLANES_COUNT, default=defaults.get(CONF_PLANES_COUNT, 1))
        ] = vol.All(vol.Coerce(int), vol.Range(min=1, max=4))
    return vol.Schema(fields)


def _plane_schema(defaults: dict[str, Any] | None = None, *, show_add: bool = True) -> vol.Schema:
    defaults = defaults or {}
    fields: dict[Any, Any] = {
        vol.Required(CONF_DEC, default=defaults.get(CONF_DEC, 30)): vol.All(
            vol.Coerce(float), vol.Range(min=0, max=90)
        ),
        vol.Required(CONF_AZ, default=defaults.get(CONF_AZ, 0)): vol.All(
            vol.Coerce(float), vol.Range(min=-180, max=180)
        ),
        vol.Required(CONF_KWP, default=defaults.get(CONF_KWP, 1.0)): vol.All(
            vol.Coerce(float), vol.Range(min=0.001)
        ),
    }
    if show_add:
        fields[vol.Optional(CONF_ADD_ANOTHER, default=False)] = bool
    return vol.Schema(fields)


def _telemetry_schema(defaults: dict[str, Any]) -> vol.Schema:
    entity = selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor"))
    binary = selector.EntitySelector(selector.EntitySelectorConfig(domain="binary_sensor"))
    return vol.Schema(
        {
            vol.Optional(CONF_ENERGY_ENTITY): entity,
            vol.Optional(CONF_POWER_ENTITY): entity,
            vol.Optional(CONF_CURTAILMENT_ENTITY): binary,
            vol.Optional(CONF_BATTERY_SOC_ENTITY): entity,
            vol.Optional(
                CONF_SOC_THRESHOLD, default=defaults.get(CONF_SOC_THRESHOLD, DEFAULT_SOC_THRESHOLD)
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=100)),
            vol.Optional(
                CONF_EXPORT_LIMITED, default=defaults.get(CONF_EXPORT_LIMITED, False)
            ): bool,
            vol.Optional(CONF_INVERTER_TEMP_ENTITY): entity,
        }
    )


def _system_payload(options: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": options.get(CONF_NAME, data.get(CONF_NAME, "Home")),
        "lat": float(options[CONF_LAT]),
        "lon": float(options[CONF_LON]),
        "inverter_kw": options.get(CONF_INVERTER_KW) or None,
        "system_loss": float(options.get(CONF_SYSTEM_LOSS, DEFAULT_SYSTEM_LOSS)),
        "damping_morning": float(options.get(CONF_DAMPING_MORNING, 0.0)),
        "damping_evening": float(options.get(CONF_DAMPING_EVENING, 0.0)),
        "planes": [
            {"dec": float(p[CONF_DEC]), "az": float(p[CONF_AZ]), "kwp": float(p[CONF_KWP])}
            for p in options.get(CONF_PLANES, [])
        ],
    }


class LocalSolarCastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    def __init__(self) -> None:
        self._url = DEFAULT_URL
        self._token: str | None = None
        self._system: dict[str, Any] = {}
        self._planes: list[dict[str, Any]] = []

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            url = user_input[CONF_URL].strip().rstrip("/")
            token = (user_input.get(CONF_TOKEN) or "").strip() or None
            api = LocalSolarCastApi(async_get_clientsession(self.hass), url, token)
            try:
                await api.health()
            except LocalSolarCastAuthError:
                errors["base"] = "invalid_auth"
            except LocalSolarCastError:
                errors["base"] = "cannot_connect"
            else:
                self._url = url
                self._token = token
                return await self.async_step_system()
        schema = vol.Schema(
            {
                vol.Required(CONF_URL, default=self._url): str,
                vol.Optional(CONF_TOKEN, default=""): str,
            }
        )
        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )

    async def async_step_system(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            system_id = slugify(user_input.get(CONF_SYSTEM_ID) or user_input[CONF_NAME])
            if not SLUG.match(system_id or ""):
                errors["base"] = "invalid_system_id"
            else:
                await self.async_set_unique_id(f"{self._url}|{system_id}")
                self._abort_if_unique_id_configured()
                self._system = {
                    **user_input,
                    CONF_SYSTEM_ID: system_id,
                }
                self._planes = []
                return await self.async_step_plane()
        defaults = {
            CONF_NAME: "Home",
            CONF_LAT: self.hass.config.latitude,
            CONF_LON: self.hass.config.longitude,
        }
        schema = _system_schema(defaults)
        schema = schema.extend(
            {vol.Optional(CONF_SYSTEM_ID, default="home"): str}
        )
        return self.async_show_form(step_id="system", data_schema=schema, errors=errors)

    async def async_step_plane(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            self._planes.append(
                {
                    CONF_DEC: float(user_input[CONF_DEC]),
                    CONF_AZ: float(user_input[CONF_AZ]),
                    CONF_KWP: float(user_input[CONF_KWP]),
                }
            )
            if user_input.get(CONF_ADD_ANOTHER) and len(self._planes) < 4:
                return await self.async_step_plane()
            return await self.async_step_telemetry()
        return self.async_show_form(
            step_id="plane",
            data_schema=_plane_schema(),
            description_placeholders={"count": str(len(self._planes) + 1)},
        )

    async def async_step_telemetry(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            options = {
                **self._system,
                CONF_PLANES: self._planes,
                **user_input,
            }
            options.pop(CONF_SYSTEM_ID, None)
            api = LocalSolarCastApi(
                async_get_clientsession(self.hass), self._url, self._token
            )
            try:
                await api.put_system(
                    self._system[CONF_SYSTEM_ID],
                    _system_payload(options, self._system),
                )
            except LocalSolarCastAuthError:
                errors["base"] = "invalid_auth"
            except LocalSolarCastError:
                errors["base"] = "cannot_connect"
            else:
                return self.async_create_entry(
                    title=options[CONF_NAME],
                    data={
                        CONF_URL: self._url,
                        CONF_TOKEN: self._token,
                        CONF_SYSTEM_ID: self._system[CONF_SYSTEM_ID],
                    },
                    options=options,
                )
        return self.async_show_form(
            step_id="telemetry",
            data_schema=_telemetry_schema({}),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return LocalSolarCastOptionsFlow(config_entry)


class LocalSolarCastOptionsFlow(OptionsFlow):
    def __init__(self, config_entry: ConfigEntry) -> None:
        self._entry = config_entry
        self._options: dict[str, Any] = dict(config_entry.options)
        self._planes: list[dict[str, Any]] = []

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            self._options.update(user_input)
            self._count = int(user_input[CONF_PLANES_COUNT])
            existing = self._entry.options.get(CONF_PLANES, [])
            self._planes = []
            for index in range(self._count):
                base = existing[index] if index < len(existing) else {"dec": 30, "az": 0, "kwp": 1.0}
                self._planes.append({"dec": base.get("dec", 30), "az": base.get("az", 0), "kwp": base.get("kwp", 1.0)})
            return await self.async_step_planes()
        defaults = dict(self._entry.options)
        defaults[CONF_PLANES_COUNT] = len(defaults.get(CONF_PLANES, [])) or 1
        return self.async_show_form(
            step_id="init",
            data_schema=_system_schema(defaults, with_planes_count=True),
        )

    async def async_step_planes(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            for index in range(self._count):
                self._planes[index] = {
                    CONF_DEC: float(user_input[f"dec_{index}"]),
                    CONF_AZ: float(user_input[f"az_{index}"]),
                    CONF_KWP: float(user_input[f"kwp_{index}"]),
                }
            return await self.async_step_telemetry()
        fields: dict[Any, Any] = {}
        for index, plane in enumerate(self._planes):
            fields[
                vol.Required(f"dec_{index}", default=plane[CONF_DEC])
            ] = vol.All(vol.Coerce(float), vol.Range(min=0, max=90))
            fields[
                vol.Required(f"az_{index}", default=plane[CONF_AZ])
            ] = vol.All(vol.Coerce(float), vol.Range(min=-180, max=180))
            fields[
                vol.Required(f"kwp_{index}", default=plane[CONF_KWP])
            ] = vol.All(vol.Coerce(float), vol.Range(min=0.001))
        return self.async_show_form(step_id="planes", data_schema=vol.Schema(fields))

    async def async_step_telemetry(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            options = {**self._options, CONF_PLANES: self._planes, **user_input}
            api = LocalSolarCastApi(
                async_get_clientsession(self.hass),
                self._entry.data[CONF_URL],
                self._entry.data.get(CONF_TOKEN),
            )
            try:
                await api.put_system(
                    self._entry.data[CONF_SYSTEM_ID],
                    _system_payload(options, self._entry.data),
                )
            except LocalSolarCastAuthError:
                errors["base"] = "invalid_auth"
            except LocalSolarCastError:
                errors["base"] = "cannot_connect"
            else:
                return self.async_create_entry(title=options[CONF_NAME], data=options)
        return self.async_show_form(
            step_id="telemetry",
            data_schema=_telemetry_schema(self._entry.options),
            errors=errors,
        )
