"""Retrospective clear-sky runs used by the calibration and shading learners."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .physics_engine import SimResult, SystemConfig, simulate_system
from .pvgis_client import Horizon


def clearsky_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Weather frame with no forecast radiation, suitable for ``clear_sky=True``."""
    frame = pd.DataFrame(
        {
            "ghi": np.nan,
            "dni": np.nan,
            "dhi": np.nan,
            "temp_2m_c": 15.0,
            "wind_10m_ms": 1.0,
            "cloud_total": np.nan,
            "cloud_low": np.nan,
            "cloud_mid": np.nan,
            "cloud_high": np.nan,
            "precip_mm": 0.0,
            "snowfall_cm": 0.0,
            "snow_depth_m": 0.0,
        },
        index=index,
    )
    return frame


def simulate_clearsky(
    index: pd.DatetimeIndex,
    system: SystemConfig,
    horizon: Horizon,
    shading: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    calibration_c: float = 1.0,
) -> SimResult:
    return simulate_system(
        clearsky_frame(index),
        system,
        horizon,
        shading=shading,
        calibration_c=calibration_c,
        clear_sky=True,
    )
