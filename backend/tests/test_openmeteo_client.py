import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.openmeteo_client import CANONICAL_COLUMNS, OpenMeteoClient, parse_openmeteo
from app.core.timeconv import fs_az_to_pvlib

FIXTURE = Path(__file__).parent / "fixtures" / "openmeteo_multimodel.json"


def _payload() -> dict:
    times15 = pd.date_range("2026-06-21 06:00", "2026-06-21 09:00", freq="15min", tz="UTC")
    times_h = pd.date_range("2026-06-21 06:00", "2026-06-21 09:00", freq="h", tz="UTC")
    m15 = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times15]}
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times_h]}
    for model, scale in (("alpha", 1.0), ("beta", 2.0)):
        m15[f"shortwave_radiation_instant_{model}"] = [500.0 * scale] * len(times15)
        m15[f"direct_normal_irradiance_instant_{model}"] = [100.0 * scale] * len(times15)
        m15[f"diffuse_radiation_instant_{model}"] = [50.0 * scale] * len(times15)
        m15[f"temperature_2m_{model}"] = [10.0] * len(times15)
        m15[f"wind_speed_10m_{model}"] = [2.0] * len(times15)
        hourly[f"cloud_cover_{model}"] = [10.0] * len(times_h)
        hourly[f"cloud_cover_low_{model}"] = [1.0] * len(times_h)
        hourly[f"cloud_cover_mid_{model}"] = [2.0] * len(times_h)
        hourly[f"cloud_cover_high_{model}"] = [3.0] * len(times_h)
        hourly[f"precipitation_{model}"] = [0.0, 0.0, 8.0, 0.0]
        hourly[f"snowfall_{model}"] = [0.0, 0.0, 0.0, 2.0]
        hourly[f"snow_depth_{model}"] = [0.1, 0.2, 0.3, 0.4]
        hourly[f"temperature_2m_{model}"] = [20.0] * len(times_h)
        hourly[f"wind_speed_10m_{model}"] = [5.0] * len(times_h)
    return {
        "latitude": 45.0,
        "longitude": 0.0,
        "elevation": 120.0,
        "minutely_15": m15,
        "hourly": hourly,
    }


def test_parses_real_multimodel_fixture():
    payload = json.loads(FIXTURE.read_text())
    result = parse_openmeteo(
        payload,
        ["icon_seamless", "ecmwf_ifs025", "gfs_seamless"],
        [0.5, 0.35, 0.15],
        44.8125,
        20.4612,
    )
    assert result.elevation_m == pytest.approx(114.0)
    assert list(result.frame.columns) == CANONICAL_COLUMNS
    assert not result.frame["ghi"].isna().any()
    assert (result.frame["ghi"] >= 0).all()
    assert (result.frame["dni"] >= 0).all()
    # Fixture has native 15-minute radiation for all three models.
    assert len(result.frame) == len(payload["minutely_15"]["time"])
    assert list(result.model_ghi.columns) == [
        "icon_seamless",
        "ecmwf_ifs025",
        "gfs_seamless",
    ]
    assert result.model_ghi.index.equals(result.frame.index[: len(result.model_ghi)])


def test_hourly_join_and_closure():
    result = parse_openmeteo(_payload(), ["alpha", "beta"], [0.5, 0.5], 45.0, 0.0)
    assert list(result.model_ghi.columns) == ["alpha", "beta"]
    assert result.model_ghi["beta"].iloc[0] == pytest.approx(1000.0)
    frame = result.frame
    # Preceding-hour sums are split across the four quarter-hours of that hour.
    assert frame.loc["2026-06-21T07:00Z", "precip_mm"] == pytest.approx(2.0)
    assert frame.loc["2026-06-21T07:15Z", "precip_mm"] == pytest.approx(2.0)
    assert frame.loc["2026-06-21T06:00Z", "precip_mm"] == pytest.approx(0.0)
    # snowfall at 08:00 = 2 cm / 4 = 0.5 cm per quarter-hour.
    assert frame.loc["2026-06-21T08:00Z", "snowfall_cm"] == pytest.approx(0.5)
    # Instantaneous hourly values are interpolated.
    assert frame.loc["2026-06-21T06:30Z", "snow_depth_m"] == pytest.approx(0.15)
    # minutely_15 temperature wins over the hourly value.
    assert frame.loc["2026-06-21T06:30Z", "temp_2m_c"] == pytest.approx(10.0)
    # Closure: ghi == dhi + dni * max(0, cos(theta_z))
    from pvlib.solarposition import get_solarposition

    position = get_solarposition(frame.index, 45.0, 0.0, altitude=result.elevation_m or 0.0)
    cosz = np.cos(np.deg2rad(position["apparent_zenith"])).clip(lower=0)
    expected = frame["dhi"] + frame["dni"] * cosz
    assert np.allclose(frame["ghi"], expected, atol=1e-8)
    # Blended radiation with equal weights: alpha=1x, beta=2x => 1.5x.
    assert frame.loc["2026-06-21T06:00Z", "dni"] == pytest.approx(150.0)
    assert frame.loc["2026-06-21T06:00Z", "dhi"] == pytest.approx(75.0)


def test_weight_renormalization_when_model_is_null():
    payload = _payload()
    # beta has no radiation at the second timestamp.
    idx = 2
    for base in ("shortwave_radiation_instant", "direct_normal_irradiance_instant", "diffuse_radiation_instant"):
        payload["minutely_15"][f"{base}_beta"][idx] = None
    result = parse_openmeteo(payload, ["alpha", "beta"], [0.5, 0.5], 45.0, 0.0)
    frame = result.frame
    stamp = frame.index[idx]
    assert frame.loc[stamp, "dni"] == pytest.approx(100.0)
    assert frame.loc[stamp, "dhi"] == pytest.approx(50.0)


def test_prefix_only_payload_is_supported():
    payload = _payload()
    # Strip suffixes to mimic a single-model response without model-suffixed keys.
    stripped = {"time": payload["minutely_15"]["time"]}
    for key, values in payload["minutely_15"].items():
        if key != "time" and key.endswith("_alpha"):
            stripped[key.rsplit("_alpha", 1)[0]] = values
    payload["minutely_15"] = stripped
    hourly = {"time": payload["hourly"]["time"]}
    for key, values in payload["hourly"].items():
        if key != "time" and key.endswith("_alpha"):
            hourly[key.rsplit("_alpha", 1)[0]] = values
    payload["hourly"] = hourly
    result = parse_openmeteo(payload, ["alpha"], [1.0], 45.0, 0.0)
    assert result.frame["dni"].iloc[0] == pytest.approx(100.0)


def test_all_null_radiation_rows_are_dropped():
    payload = _payload()
    for model in ("alpha", "beta"):
        for base in ("shortwave_radiation_instant", "direct_normal_irradiance_instant", "diffuse_radiation_instant"):
            payload["minutely_15"][f"{base}_{model}"][0] = None
    result = parse_openmeteo(payload, ["alpha", "beta"], [0.5, 0.5], 45.0, 0.0)
    assert pd.Timestamp("2026-06-21T06:00Z") not in result.frame.index


def test_build_params():
    client = OpenMeteoClient(models=["icon_seamless"], weights=[1.0])
    params = client.build_params(44.8125, 20.4612)
    assert params["wind_speed_unit"] == "ms"
    assert params["timezone"] == "GMT"
    assert params["past_days"] == 2
    assert params["forecast_days"] == 7
    assert params["models"] == "icon_seamless"
