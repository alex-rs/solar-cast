# External API verification

Recorded 2026-11-11 against the default Belgrade coordinates
(44.8125, 20.4612). Raw responses are committed under `backend/tests/fixtures/`.

## 1. Open-Meteo

- Request accepted with `forecast_days=7`, `past_days=2`, `timezone=GMT`,
  `wind_speed_unit=ms` and the three configured models.
- All five `minutely_15` variables (radiation instants, `temperature_2m`,
  `wind_speed_10m`) are accepted. With multiple models, every model-suffixed key
  is present, e.g. `shortwave_radiation_instant_icon_seamless`,
  `temperature_2m_ecmwf_ifs025`.
- `hourly` returns all requested cloud, precipitation, snowfall and snow-depth
  variables, likewise suffixed per model.
- Response `elevation` is 114 m for the default location.
- Native 15-minute radiation is available for this location in the fixture
  (ICON-D2 domain); no explicit fallback is needed for the code path.

## 2. PVGIS horizon

- `v5_3/printhorizon` returns HTTP 200 with 49 points at 7.5° spacing.
- `A` follows the Forecast.Solar convention (0 = South, -90 = East, +90 = West).
- Both `-180` and `+180` entries are present and map to pvlib azimuth 0; the
  parser deduplicates them, leaving 48 points.
- `source` is recorded as `pvgis_v5_3`; `v5_2` is the fallback.

## 3. Forecast.Solar estimate

- `GET /estimate/44.8125/20.4612/30/0/5` returns HTTP 200.
- `result.watts`, `result.watt_hours_period`, `result.watt_hours` and
  `result.watt_hours_day` are present, plus the full `message` block
  (`code`, `type`, `text`, `info`, `ratelimit`).
- Key format is local time `"YYYY-MM-DD HH:MM:SS"`; exact sunrise/sunset
  timestamps appear as anchors (`06:29:48` / `16:13:24`).
- `watt_hours` is cumulative within each local day and matches the running sum
  of `watt_hours_period`; the final non-zero period ends at sunset, so
  `watt_hours_period` keys are treated as **period end** labels in the compat
  API.
- Free-tier rate limit observed: 12 requests/hour/IP.

## 4. Home Assistant energy platform

- `async_get_solar_forecast(hass, config_entry_id)` returns
  `{"wh_hours": {iso_hour_start: wh}}` or `None`. Keys are hour-start ISO
  timestamps with UTC offset; values are hourly energy in Wh. Implemented and
  asserted in `tests_ha/test_energy.py`.
