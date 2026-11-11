# API

Base URL: `http://localhost:8000`. When `API_TOKEN` is set, every `/api/v1/*`
route requires `Authorization: Bearer <token>`. The Forecast.Solar-shaped
routes are public when `COMPAT_PUBLIC=true`.

## Systems

`PUT /api/v1/systems/{system_id}` creates or replaces a system (1-4 planes).
Changes to geometry or kWp reset the learned calibration, shading and MOS layers.

```json
{
  "name": "Home",
  "lat": 44.8125,
  "lon": 20.4612,
  "timezone": "Europe/Belgrade",
  "inverter_kw": 5.0,
  "system_loss": 0.10,
  "gamma_pdc": -0.004,
  "damping_morning": 0.0,
  "damping_evening": 0.0,
  "planes": [{"dec": 30, "az": 0, "kwp": 5.0}]
}
```

`az` uses Forecast.Solar conventions: 0 = South, -90 = East, 90 = West.
`GET /api/v1/systems`, `GET /api/v1/systems/{id}`, `DELETE /api/v1/systems/{id}`,
and `POST /api/v1/systems/{id}/retrain` are also available.

## Native forecast

`GET /api/v1/systems/{system_id}/forecast?resolution=15|60&days=1..7&layer=final|hybrid|physics|physics_base`

Periods are labeled by their start and serialized as ISO 8601 with UTC offset.
`wh` is energy for the period; `mean_watts = wh / (resolution/60)`. `daily` sums
periods by local calendar date.

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
server-side. Push rows override backfill rows; backfill never overrides push.
The response is `{"accepted": n, "rejected": [{"index": i, "reason": "..."}]}`.

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

## Health

`GET /api/v1/health` returns DB status, last weather refresh, scheduler job
next-run times, per-system layer status, last telemetry timestamp and the active
MOS model with holdout metrics.
