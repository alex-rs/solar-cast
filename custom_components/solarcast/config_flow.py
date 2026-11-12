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

from .api import SolarCastApi, SolarCastAuthError, SolarCastError
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
    CONF_AC_KW,
    CONF_INVERTER,
    CONF_INVERTER_COUNT,
    CONF_INVERTER_ID,
    CONF_INVERTER_KW,
    CONF_INVERTER_TEMP_ENTITY,
    CONF_INVERTERS,
    CONF_KWP,
    CONF_LAT,
    CONF_LOCAL_FALLBACK,
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
        vol.Required(
            CONF_INVERTER_COUNT,
            default=(
                len(defaults[CONF_INVERTERS])
                if CONF_INVERTERS in defaults
                else (1 if defaults.get(CONF_INVERTER_KW) else 0)
            ),
        ): vol.All(vol.Coerce(int), vol.Range(min=0, max=8)),
        vol.Optional(
            CONF_SYSTEM_LOSS, default=defaults.get(CONF_SYSTEM_LOSS, DEFAULT_SYSTEM_LOSS)
        ): vol.All(vol.Coerce(float), vol.Range(min=0, max=0.99)),
        vol.Optional(
            CONF_DAMPING_MORNING, default=defaults.get(CONF_DAMPING_MORNING, 0.0)
        ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
        vol.Optional(
            CONF_DAMPING_EVENING, default=defaults.get(CONF_DAMPING_EVENING, 0.0)
        ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
        vol.Optional(
            CONF_LOCAL_FALLBACK, default=defaults.get(CONF_LOCAL_FALLBACK, True)
        ): bool,
    }
    if with_planes_count:
        fields[
            vol.Required(CONF_PLANES_COUNT, default=defaults.get(CONF_PLANES_COUNT, 1))
        ] = vol.All(vol.Coerce(int), vol.Range(min=1, max=4))
    return vol.Schema(fields)


def _inverter_schema(inverters: list[dict[str, Any]]) -> vol.Schema:
    fields: dict[Any, Any] = {}
    for index in range(len(inverters)):
        existing = inverters[index]
        fields[
            vol.Required(
                f"inverter_{index}_id",
                default=existing.get(CONF_INVERTER_ID, "main" if index == 0 else f"inv{index + 1}"),
            )
        ] = str
        fields[
            vol.Required(
                f"inverter_{index}_kw",
                default=existing.get(CONF_AC_KW, 5.0),
            )
        ] = vol.All(vol.Coerce(float), vol.Range(min=0.01, max=1000))
    return vol.Schema(fields)


def _plane_schema(
    defaults: dict[str, Any] | None = None,
    *,
    show_add: bool = True,
    inverters: list[dict[str, Any]] | None = None,
) -> vol.Schema:
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
    if inverters:
        ids = [inverter[CONF_INVERTER_ID] for inverter in inverters]
        default = defaults.get(CONF_INVERTER)
        if default not in ids:
            default = ids[0]
        fields[vol.Required(CONF_INVERTER, default=default)] = vol.In(ids)
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
        "system_loss": float(options.get(CONF_SYSTEM_LOSS, DEFAULT_SYSTEM_LOSS)),
        "damping_morning": float(options.get(CONF_DAMPING_MORNING, 0.0)),
        "damping_evening": float(options.get(CONF_DAMPING_EVENING, 0.0)),
        "local_fallback": bool(options.get(CONF_LOCAL_FALLBACK, True)),
        "planes": [
            {
                "dec": float(p[CONF_DEC]),
                "az": float(p[CONF_AZ]),
                "kwp": float(p[CONF_KWP]),
                "inverter": p.get(CONF_INVERTER),
            }
            for p in options.get(CONF_PLANES, [])
        ],
        "inverters": [
            {
                "id": inverter[CONF_INVERTER_ID],
                "ac_kw": float(inverter[CONF_AC_KW]),
            }
            for inverter in options.get(CONF_INVERTERS, [])
        ],
    }


class SolarCastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 2

    def __init__(self) -> None:
        self._url = DEFAULT_URL
        self._token: str | None = None
        self._system: dict[str, Any] = {}
        self._planes: list[dict[str, Any]] = []
        self._inverters: list[dict[str, Any]] = []
        self._count = 0

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            url = user_input[CONF_URL].strip().rstrip("/")
            token = (user_input.get(CONF_TOKEN) or "").strip() or None
            api = SolarCastApi(async_get_clientsession(self.hass), url, token)
            try:
                await api.health()
            except SolarCastAuthError:
                errors["base"] = "invalid_auth"
            except SolarCastError:
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
                self._inverters = []
                self._count = int(user_input[CONF_INVERTER_COUNT])
                if self._count > 0:
                    return await self.async_step_inverters()
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

    async def async_step_inverters(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        seed = self._inverters or [
            {
                CONF_INVERTER_ID: "main" if index == 0 else f"inv{index + 1}",
                CONF_AC_KW: 5.0,
            }
            for index in range(self._count)
        ]
        if user_input is not None:
            ids = [
                str(user_input[f"inverter_{index}_id"]).strip()
                for index in range(self._count)
            ]
            if any(not re.match(r"^[a-z0-9][a-z0-9_-]{0,31}$", value) for value in ids):
                errors["base"] = "invalid_inverter_id"
            elif len(ids) != len(set(ids)):
                errors["base"] = "duplicate_inverter_id"
            else:
                self._inverters = [
                    {
                        CONF_INVERTER_ID: ids[index],
                        CONF_AC_KW: float(user_input[f"inverter_{index}_kw"]),
                    }
                    for index in range(self._count)
                ]
                return await self.async_step_plane()
            seed = [
                {
                    CONF_INVERTER_ID: ids[index],
                    CONF_AC_KW: user_input[f"inverter_{index}_kw"],
                }
                for index in range(self._count)
            ]
        return self.async_show_form(
            step_id="inverters",
            data_schema=_inverter_schema(seed),
            errors=errors,
        )

    async def async_step_plane(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            self._planes.append(
                {
                    CONF_DEC: float(user_input[CONF_DEC]),
                    CONF_AZ: float(user_input[CONF_AZ]),
                    CONF_KWP: float(user_input[CONF_KWP]),
                    CONF_INVERTER: (
                        user_input.get(CONF_INVERTER) if self._inverters else None
                    ),
                }
            )
            if user_input.get(CONF_ADD_ANOTHER) and len(self._planes) < 4:
                return await self.async_step_plane()
            return await self.async_step_telemetry()
        return self.async_show_form(
            step_id="plane",
            data_schema=_plane_schema(inverters=self._inverters),
            description_placeholders={"count": str(len(self._planes) + 1)},
        )

    async def async_step_telemetry(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            options = {
                **self._system,
                CONF_PLANES: self._planes,
                CONF_INVERTERS: self._inverters,
                **user_input,
            }
            options.pop(CONF_SYSTEM_ID, None)
            api = SolarCastApi(
                async_get_clientsession(self.hass), self._url, self._token
            )
            try:
                await api.put_system(
                    self._system[CONF_SYSTEM_ID],
                    _system_payload(options, self._system),
                )
            except SolarCastAuthError:
                errors["base"] = "invalid_auth"
            except SolarCastError:
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
        return SolarCastOptionsFlow(config_entry)


class SolarCastOptionsFlow(OptionsFlow):
    def __init__(self, config_entry: ConfigEntry) -> None:
        self._entry = config_entry
        self._options: dict[str, Any] = dict(config_entry.options)
        self._planes: list[dict[str, Any]] = []
        self._count = 0
        self._inverters: list[dict[str, Any]] = []

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            self._options.update(user_input)
            self._count = int(user_input[CONF_INVERTER_COUNT])
            existing = self._entry.options.get(CONF_PLANES, [])
            self._planes = []
            for index in range(int(user_input[CONF_PLANES_COUNT])):
                base = (
                    existing[index]
                    if index < len(existing)
                    else {"dec": 30, "az": 0, "kwp": 1.0}
                )
                self._planes.append(
                    {
                        "dec": base.get("dec", 30),
                        "az": base.get("az", 0),
                        "kwp": base.get("kwp", 1.0),
                        CONF_INVERTER: base.get(CONF_INVERTER),
                    }
                )
            existing_inverters = list(self._options.get(CONF_INVERTERS, []))
            if (
                not existing_inverters
                and self._options.get(CONF_INVERTER_KW)
                and self._count > 0
            ):
                existing_inverters = [
                    {
                        CONF_INVERTER_ID: "main",
                        CONF_AC_KW: float(self._options[CONF_INVERTER_KW]),
                    }
                ]
                self._options[CONF_INVERTERS] = existing_inverters
                self._planes = [
                    {**plane, CONF_INVERTER: "main"} for plane in self._planes
                ]
            self._inverters = [
                (
                    existing_inverters[index]
                    if index < len(existing_inverters)
                    else {
                        CONF_INVERTER_ID: "main" if index == 0 else f"inv{index + 1}",
                        CONF_AC_KW: 5.0,
                    }
                )
                for index in range(self._count)
            ]
            if self._count > 0:
                return await self.async_step_inverters()
            return await self.async_step_planes()
        defaults = dict(self._entry.options)
        if not defaults.get(CONF_INVERTERS) and defaults.get(CONF_INVERTER_KW):
            defaults[CONF_INVERTERS] = [
                {
                    CONF_INVERTER_ID: "main",
                    CONF_AC_KW: float(defaults[CONF_INVERTER_KW]),
                }
            ]
            defaults[CONF_PLANES] = [
                {**plane, CONF_INVERTER: "main"}
                for plane in defaults.get(CONF_PLANES, [])
            ]
        defaults[CONF_PLANES_COUNT] = len(defaults.get(CONF_PLANES, [])) or 1
        return self.async_show_form(
            step_id="init",
            data_schema=_system_schema(defaults, with_planes_count=True),
        )

    async def async_step_inverters(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        seed = self._inverters or [
            {
                CONF_INVERTER_ID: "main" if index == 0 else f"inv{index + 1}",
                CONF_AC_KW: 5.0,
            }
            for index in range(self._count)
        ]
        if user_input is not None:
            ids = [
                str(user_input[f"inverter_{index}_id"]).strip()
                for index in range(self._count)
            ]
            if any(not re.match(r"^[a-z0-9][a-z0-9_-]{0,31}$", value) for value in ids):
                errors["base"] = "invalid_inverter_id"
            elif len(ids) != len(set(ids)):
                errors["base"] = "duplicate_inverter_id"
            else:
                self._inverters = [
                    {
                        CONF_INVERTER_ID: ids[index],
                        CONF_AC_KW: float(user_input[f"inverter_{index}_kw"]),
                    }
                    for index in range(self._count)
                ]
                return await self.async_step_planes()
            seed = [
                {
                    CONF_INVERTER_ID: ids[index],
                    CONF_AC_KW: user_input[f"inverter_{index}_kw"],
                }
                for index in range(self._count)
            ]
        return self.async_show_form(
            step_id="inverters",
            data_schema=_inverter_schema(seed),
            errors=errors,
        )

    async def async_step_planes(self, user_input: dict[str, Any] | None = None):
        ids = [inverter[CONF_INVERTER_ID] for inverter in self._inverters]
        if user_input is not None:
            for index in range(len(self._planes)):
                self._planes[index] = {
                    CONF_DEC: float(user_input[f"dec_{index}"]),
                    CONF_AZ: float(user_input[f"az_{index}"]),
                    CONF_KWP: float(user_input[f"kwp_{index}"]),
                    CONF_INVERTER: (
                        user_input[f"inverter_{index}"] if ids else None
                    ),
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
            if ids:
                default = plane.get(CONF_INVERTER)
                if default not in ids:
                    default = ids[0]
                fields[vol.Required(f"inverter_{index}", default=default)] = vol.In(ids)
        return self.async_show_form(step_id="planes", data_schema=vol.Schema(fields))

    async def async_step_telemetry(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            options = {
                **self._options,
                CONF_PLANES: self._planes,
                CONF_INVERTERS: self._inverters,
                **user_input,
            }
            options.pop(CONF_INVERTER_KW, None)
            api = SolarCastApi(
                async_get_clientsession(self.hass),
                self._entry.data[CONF_URL],
                self._entry.data.get(CONF_TOKEN),
            )
            try:
                await api.put_system(
                    self._entry.data[CONF_SYSTEM_ID],
                    _system_payload(options, self._entry.data),
                )
            except SolarCastAuthError:
                errors["base"] = "invalid_auth"
            except SolarCastError:
                errors["base"] = "cannot_connect"
            else:
                return self.async_create_entry(title=options[CONF_NAME], data=options)
        return self.async_show_form(
            step_id="telemetry",
            data_schema=_telemetry_schema(self._entry.options),
            errors=errors,
        )
