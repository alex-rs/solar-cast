DOMAIN = "solarcast"

PLATFORMS = ["sensor", "binary_sensor"]

CONF_URL = "url"
CONF_TOKEN = "token"
CONF_SYSTEM_ID = "system_id"
CONF_NAME = "name"
CONF_LAT = "lat"
CONF_LON = "lon"
CONF_INVERTER_KW = "inverter_kw"
CONF_INVERTERS = "inverters"
CONF_INVERTER_COUNT = "inverter_count"
CONF_INVERTER = "inverter"
CONF_INVERTER_ID = "inverter_id"
CONF_AC_KW = "ac_kw"
CONF_SYSTEM_LOSS = "system_loss"
CONF_DAMPING_MORNING = "damping_morning"
CONF_DAMPING_EVENING = "damping_evening"
CONF_PLANES = "planes"
CONF_DEC = "dec"
CONF_AZ = "az"
CONF_KWP = "kwp"
CONF_PLANES_COUNT = "planes_count"
CONF_ADD_ANOTHER = "add_another"

CONF_ENERGY_ENTITY = "energy_entity"
CONF_POWER_ENTITY = "power_entity"
CONF_CURTAILMENT_ENTITY = "curtailment_entity"
CONF_BATTERY_SOC_ENTITY = "battery_soc_entity"
CONF_SOC_THRESHOLD = "soc_threshold"
CONF_EXPORT_LIMITED = "export_limited"
CONF_INVERTER_TEMP_ENTITY = "inverter_temp_entity"

DEFAULT_URL = "http://264cce28-solarcast:8000"
DEFAULT_SOC_THRESHOLD = 98
DEFAULT_SYSTEM_LOSS = 0.10
UPDATE_INTERVAL_MINUTES = 15
TELEMETRY_STALE_MINUTES = 45
ACCURACY_INTERVAL_SECONDS = 6 * 3600

SERVICE_GET_FORECAST = "get_forecast"
SERVICE_BACKFILL = "backfill"
SERVICE_RETRAIN = "retrain"
SERVICE_UPLOAD_HORIZON = "upload_horizon"
SERVICE_CLEAR_HORIZON = "clear_horizon"
SERVICE_SET_TELEMETRY = "set_telemetry"
SERVICE_DELETE_TELEMETRY = "delete_telemetry"
