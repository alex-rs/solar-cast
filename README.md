# LocalSolarCast

Self-hosted PV production forecasting: a FastAPI backend that blends Open-Meteo
ensemble weather with `pvlib` physics, plus a Home Assistant (HACS) integration
that feeds the Energy dashboard.

## Quick start

```sh
cd backend
uv sync
uv run uvicorn app.main:app --port 8000
```

Register a system:

```sh
curl -X PUT http://localhost:8000/api/v1/systems/home \
  -H 'content-type: application/json' \
  -d '{"name":"Home","lat":44.8125,"lon":20.4612,"inverter_kw":5.0,
       "planes":[{"dec":30,"az":0,"kwp":5.0}]}'
```

Forecast: `GET /api/v1/systems/home/forecast?resolution=60&days=7`.
Forecast.Solar-compatible: `/estimate/{lat}/{lon}/{dec}/{az}/{kwp}`.

Docker: `cd backend && docker compose up -d`.

## Home Assistant

Install through HACS as a custom repository, then add the **LocalSolarCast**
integration and point it at the backend URL. The integration registers the
system, polls the forecast and (optionally) pushes your PV energy meter back as
telemetry for calibration and shading learning.

## Limitations

- Learned shading is one factor per system; for multi-plane systems it is applied
  to every plane's beam.
- PVGIS horizon covers terrain only; local obstacles are learned over time from
  clear days.
- Snow-on-module loss is approximate; periods after snowfall are excluded from
  learning.
- Open-Meteo's free tier is non-commercial. Forecast.Solar polling uses the free
  tier (~2 days, one plane per call).
- The compat endpoint mirrors Forecast.Solar's response shape, not an official API.
