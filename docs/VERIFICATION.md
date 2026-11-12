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

## 4. Local fallback (2026-11-12)

Recorded with `OPENMETEO_BASE_URL` pointed at a closed port, so every weather
call fails at connect time. The database held 45 days of synthetic 15-minute
telemetry (the generator used by the unit tests) and a local model trained
offline from it.

| Check | Result |
|---|---|
| `GET /api/v1/systems/home/forecast` | 200, `provenance=local_model`, `degraded=true`, `local_horizon_minutes=360` |
| `learning` block | `local=true`, `local_skill=0.668` (holdout nMAE 0.159 vs baseline 0.480), telemetry age 19.7 min |
| payload shape | 91 hourly periods and 7 daily entries, peak 2921 W; every period `confidence=low` because the check ran after sunset, so all leads exceeded the 6 h horizon |
| `PUT` with `local_fallback=false`, then `GET` | 500 — previous behaviour restored |
| `PUT` back to `true`, then `GET` | `provenance=local_model` again |
| `GET /api/v1/health` | `layers.local=true`, `local_model.holdout_nmae_local=0.159` |
| `GET /estimate/...` (compat route) | 500 — pre-existing behaviour: that endpoint is stateless and needs live weather; the fallback covers the native API only |

## 5. Whole-day "today" totals (2026-11-12)

Reported symptom: `sensor.solarcast_house_energy_production_today` fell through
the day (1.47 -> 0.62 -> 0.19 -> 0.0 kWh) because the backend returned periods only
from "now" and the integration summed exactly those.

Checked live at 18:58 CET against the same instance:

| Quantity | Value |
|---|---|
| Sum of returned `periods` for today (old sensor value) | 0 Wh |
| `today.wh` | 21,319 Wh |
| `today.elapsed_wh` / `today.remaining_wh` | 21,190 / 129 Wh |
| `today.measured_wh` / `today.simulated_wh` | 21,190 / 9,457 Wh (`source=measured+forecast`) |
| `daily[0]` | `{date: 2026-11-12, wh: 21,319}` |

`daily` now sums whole local calendar days, and `today` is the same number, so the
sensor no longer decays. Fixed-clock tests in `backend/tests/test_today_totals.py`
pin the behaviour at 06:00, 12:00 and 21:00 local.

