# API

Base URL: `http://localhost:8000`. When `API_TOKEN` is set, every `/api/v1/*`
route requires `Authorization: Bearer <token>`. The Forecast.Solar-shaped
routes are public when `COMPAT_PUBLIC=true`.

## Systems

`PUT /api/v1/systems/{system_id}` creates or replaces a system (1-4 planes,
up to 8 inverters). Changes to plane geometry, plane kWp, inverter topology or
inverter AC rating reset the learned calibration, shading and MOS layers.

```json
{
  "name": "Home",
  "lat": 44.8125,
  "lon": 20.4612,
  "timezone": "Europe/Belgrade",
  "system_loss": 0.10,
  "gamma_pdc": -0.004,
  "damping_morning": 0.0,
  "damping_evening": 0.0,
  "inverters": [
    {"id": "east", "ac_kw": 2.0},
    {"id": "west", "ac_kw": 2.0}
  ],
  "planes": [
    {"dec": 30, "az": -90, "kwp": 2.5, "inverter": "east"},
    {"dec": 30, "az": 90, "kwp": 2.5, "inverter": "west"}
  ]
}
```

Use either `inverters` or the legacy single `inverter_kw`, never both. When
`inverters` is present, every plane must reference a declared id. When neither
is present, planes are unclipped (system kWp is the AC cap). Output includes
`inverters` and each plane's `inverter`.
`az` uses Forecast.Solar conventions: 0 = South, -90 = East, 90 = West.
`GET /api/v1/systems`, `GET /api/v1/systems/{id}`, `DELETE /api/v1/systems/{id}`,
and `POST /api/v1/systems/{id}/retrain` are also available.

## Horizon

`PUT /api/v1/systems/{system_id}/horizon` accepts a raw UTF-8 CSV body with
`Content-Type: text/csv` and returns 204. The header must be
`azimuth_fs,elevation`; 3-360 rows use Forecast.Solar azimuths (-180..180) and
elevation (-5..90 degrees). Invalid input returns 422. The user horizon
overrides the PVGIS terrain horizon for this system and persists across restarts.

`GET /api/v1/systems/{system_id}/horizon` returns
`{"system_id", "source": "user", "points": [{"azimuth_deg", "elevation_deg"}]}`
in pvlib azimuths, or 404 when no user horizon exists. `DELETE` removes it and
returns 204, or 404 when absent.

## Native forecast

`GET /api/v1/systems/{system_id}/forecast?resolution=15|60&days=1..7&layer=final|hybrid|physics|physics_base`

Periods are labeled by their start and serialized as ISO 8601 with UTC offset.
`wh` is energy for the period; `mean_watts = wh / (resolution/60)`. `daily` sums
periods by local calendar date.

`learning` states which correction layers are active and how fresh the data is:

```json
{
  "calibration_c": 0.93,
  "calibration_days": 7,
  "mos": true,
  "shading": false,
  "nowcast": true,
  "nowcast_buckets": 3,
  "telemetry_last_seen": "2026-11-12T10:45:00+00:00",
  "telemetry_age_minutes": 12.5
}
```

`calibration_days` is the number of clear days behind `calibration_c` (0 when
uncalibrated). `nowcast_buckets` counts the telemetry buckets that fed the
intraday correction (0-4). `layers_active` is the same information for existing
clients and remains supported.

## MCP (optional)

`ENABLE_MCP=true` mounts a read-only MCP server (Streamable HTTP) at `/mcp` on
the same port. When `API_TOKEN` is set, requests require
`Authorization: Bearer <token>`; an empty `API_TOKEN` leaves the endpoint
unauthenticated, like the rest of the API. With the flag unset, `/mcp` is a
plain 404 and the MCP SDK is not imported.

Tools:

- `list_systems` returns `{"systems": [{"system_id", "name", "timezone", "lat",
  "lon", "kwp_total"}]}`.
- `get_forecast(system_id, days=3, resolution=60, layer="final")` returns the
  native forecast payload. `days` must be 1-7, `resolution` is 15 or 60, and
  `layer` is `final`, `hybrid`, `physics` or `physics_base`. Unknown systems and
  out-of-range days are reported as MCP tool errors.

The server exposes no write tools: no system CRUD, telemetry or retraining.

## Telemetry

`POST /api/v1/systems/{system_id}/telemetry`

```json
{
  "origin": "push",
  "items": [
    {"period_start": "2026-11-11T11:00:00Z", "period_min": 15,
     "energy_wh": 461.4, "curtailed": false, "inverter_temp_c": 41.2}
  ]
}
```

`period_start` must align to `period_min` (15 or 60). `mean_watts` is derived
server-side. `origin` is `push` (HA pusher), `backfill` (recorder import) or
`manual` (user correction); precedence is `manual` > `push` > `backfill`, so a
manual row is never overwritten by a later push or backfill, and a backfill
never overwrites either. Equal origins replace in place (idempotent re-push).
The response is `{"accepted": n, "rejected": [{"index": i, "reason": "..."}]}`.

`DELETE /api/v1/systems/{system_id}/telemetry?start=<iso>&end=<iso>`

Deletes telemetry whose `period_start` falls in `[start, end)`. Both bounds are
required, timezone-aware ISO 8601; `end` must be after `start` (422
otherwise), and an unknown system is 404. Rows are keyed by whole-second
`period_start`, so the half-open range is applied at second precision: a
fractional `start` excludes the second it falls inside. Returns
`{"deleted": n, "start": iso, "end": iso}`. Deleting does not retrain
immediately: calibration, shading and MOS pick up the correction at the next
nightly retrain (01:30 local), while the intraday nowcast ignores telemetry
older than its 45-minute window anyway.

Nightly retraining learns per-model Open-Meteo GHI weights from clear-sky
telemetry plus stored per-model GHI; accepted weights activate automatically on
the next weather fetch. Missing data or a failed quality gate keeps the
configured `OPENMETEO_WEIGHTS`.

## Benchmark

`GET /api/v1/systems/{system_id}/benchmark?days=14`

Scores `forecast_solar`, `physics_base`, `physics`, `hybrid` and `final` on the
common hourly sample set, per lead bucket (`intraday`, `day_ahead`, `d2_3`,
`d4_7`). Metrics: `nMAE_pct`, `nRMSE_pct`, `nMAE_kwp_pct`, `MBE_wh`,
`skill_vs_forecast_solar`, `skill_vs_physics_base`, `daily_energy_mape_pct`,
`n_samples`.

## Forecast.Solar compatibility

`GET /estimate/{lat}/{lon}/{dec}/{az}/{kwp}[/{dec}/{az}/{kwp}...]`
(1-4 planes), optional `/{apikey}/estimate/...` prefix, and query parameters
`inverter`, `damping=morning,evening`, `damping_morning`, `damping_evening`.

The response mirrors Forecast.Solar: `result.watts`,
`result.watt_hours_period`, cumulative `result.watt_hours`, `result.watt_hours_day`
and the `message` block. Keys are local `"YYYY-MM-DD HH:MM:SS"` period ends with
sunrise/sunset anchors. Registered systems that match the coordinates exactly use
the `final` layer; ad-hoc requests use `physics_base`.

`message.info.solarcast` states which branch ran:
`{"matched_system": "home", "layers_active": {"calibration": true, "shading": true,
"mos": false, "nowcast": true}}`, or `{"matched_system": null, "layers_active": {}}`
for an ad-hoc request. A match requires latitude/longitude within `1e-6`, `dec`/`az`
rounded to 0.1 and `kwp` rounded to 0.001 against the registered planes, the same
inverter AC rating when `inverter` is given, and a single-inverter system — so a
tuned system silently falls back to uncorrected physics unless this reports the id.

## Health

`GET /api/v1/health` returns DB status, last weather refresh, scheduler job
next-run times, per-system layer status, last telemetry timestamp and the active
MOS model with holdout metrics.
