"""Runtime configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
    database_path: Path | None = None
    api_token: str | None = None
    compat_public: bool = True
    scheduler_tz: str | None = None
    openmeteo_base_url: str = "https://api.open-meteo.com"
    openmeteo_api_key: str | None = None
    openmeteo_models: list[str] = Field(
        default_factory=lambda: ["icon_seamless", "ecmwf_ifs025", "gfs_seamless"]
    )
    openmeteo_weights: list[float] = Field(default_factory=lambda: [0.50, 0.35, 0.15])
    enable_fs_shadow_benchmark: bool = True
    systems_file: Path = BASE_DIR / "config" / "systems.yaml"
    log_level: str = "INFO"
    pvgis_base_url: str = "https://re.jrc.ec.europa.eu/api"
    forecast_solar_base_url: str = "https://api.forecast.solar"

    @field_validator("openmeteo_models", "openmeteo_weights", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("openmeteo_weights")
    @classmethod
    def _coerce_weights(cls, value: list[float]) -> list[float]:
        return [float(v) for v in value]

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        if self.database_path is None:
            object.__setattr__(self, "database_path", self.data_dir / "solarcast.db")
        if len(self.openmeteo_models) != len(self.openmeteo_weights):
            raise ValueError("OPENMETEO_MODELS and OPENMETEO_WEIGHTS must have equal length")
        if abs(sum(self.openmeteo_weights)) < 1e-9:
            raise ValueError("OPENMETEO_WEIGHTS must not sum to zero")

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
