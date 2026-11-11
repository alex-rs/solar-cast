#!/bin/sh
set -eu
python - <<'PY'
import json, pathlib, shlex
path = pathlib.Path('/data/options.json')
options = json.loads(path.read_text()) if path.exists() else {}
mapping = {
    "api_token": "API_TOKEN",
    "timezone": "TZ",
    "enable_fs_shadow_benchmark": "ENABLE_FS_SHADOW_BENCHMARK",
    "log_level": "LOG_LEVEL",
    "openmeteo_base_url": "OPENMETEO_BASE_URL",
    "openmeteo_api_key": "OPENMETEO_API_KEY",
    "openmeteo_models": "OPENMETEO_MODELS",
    "openmeteo_weights": "OPENMETEO_WEIGHTS",
}
lines = []
for key, env in mapping.items():
    if key not in options:
        continue
    value = options[key]
    if value == "":
        continue
    if isinstance(value, bool):
        value = "true" if value else "false"
    lines.append(f"export {env}={shlex.quote(str(value))}")
pathlib.Path("/tmp/options.env").write_text("\n".join(lines))
PY
. /tmp/options.env
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
