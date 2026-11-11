"""SQLite persistence: schema, migrations and thread-safe access."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS systems (
    system_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    lat          REAL NOT NULL,
    lon          REAL NOT NULL,
    elevation_m  REAL,
    timezone     TEXT NOT NULL,
    inverter_kw  REAL,
    system_loss  REAL NOT NULL DEFAULT 0.10,
    gamma_pdc    REAL NOT NULL DEFAULT -0.004,
    damping_morning REAL NOT NULL DEFAULT 0.0,
    damping_evening REAL NOT NULL DEFAULT 0.0,
    created_at   INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS inverters (
    system_id   TEXT NOT NULL REFERENCES systems(system_id) ON DELETE CASCADE,
    inverter_id TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    ac_kw       REAL NOT NULL CHECK (ac_kw > 0),
    PRIMARY KEY (system_id, inverter_id),
    UNIQUE (system_id, ordinal)
);

CREATE TABLE IF NOT EXISTS planes (
    plane_id   TEXT PRIMARY KEY,
    system_id  TEXT NOT NULL REFERENCES systems(system_id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    dec        REAL NOT NULL CHECK (dec BETWEEN 0 AND 90),
    az_fs      REAL NOT NULL CHECK (az_fs BETWEEN -180 AND 180),
    kwp        REAL NOT NULL CHECK (kwp > 0),
    inverter_id TEXT,
    UNIQUE (system_id, ordinal)
);

CREATE TABLE IF NOT EXISTS horizon_cache (
    lat_round     REAL NOT NULL,
    lon_round     REAL NOT NULL,
    azimuth_deg   REAL NOT NULL,
    elevation_deg REAL NOT NULL,
    source        TEXT NOT NULL,
    fetched_at    INTEGER NOT NULL,
    PRIMARY KEY (lat_round, lon_round, azimuth_deg)
);

CREATE TABLE IF NOT EXISTS system_horizon (
    system_id    TEXT NOT NULL REFERENCES systems(system_id) ON DELETE CASCADE,
    azimuth_deg  REAL NOT NULL,
    elevation_deg REAL NOT NULL,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (system_id, azimuth_deg)
);

CREATE TABLE IF NOT EXISTS weather_model_ghi (
    lat_round   REAL NOT NULL,
    lon_round   REAL NOT NULL,
    fetched_at  INTEGER NOT NULL,
    target_time INTEGER NOT NULL,
    model       TEXT NOT NULL,
    ghi         REAL,
    PRIMARY KEY (lat_round, lon_round, fetched_at, target_time, model)
);

CREATE TABLE IF NOT EXISTS model_weights (
    lat_round  REAL NOT NULL,
    lon_round  REAL NOT NULL,
    model      TEXT NOT NULL,
    weight     REAL NOT NULL,
    samples    INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (lat_round, lon_round, model)
);

CREATE TABLE IF NOT EXISTS weather_forecast (
    lat_round   REAL NOT NULL,
    lon_round   REAL NOT NULL,
    fetched_at  INTEGER NOT NULL,
    target_time INTEGER NOT NULL,
    ghi REAL, dni REAL, dhi REAL,
    temp_2m_c REAL, wind_10m_ms REAL,
    cloud_total REAL, cloud_low REAL, cloud_mid REAL, cloud_high REAL,
    precip_mm REAL, snowfall_cm REAL, snow_depth_m REAL,
    PRIMARY KEY (lat_round, lon_round, fetched_at, target_time)
);

CREATE TABLE IF NOT EXISTS forecast_runs (
    system_id    TEXT NOT NULL REFERENCES systems(system_id) ON DELETE CASCADE,
    fetched_at   INTEGER NOT NULL,
    period_start INTEGER NOT NULL,
    period_min   INTEGER NOT NULL,
    lead_hours   REAL NOT NULL,
    source       TEXT NOT NULL,
    energy_wh    REAL NOT NULL,
    mean_watts   REAL NOT NULL,
    PRIMARY KEY (system_id, fetched_at, period_start, source)
);
CREATE INDEX IF NOT EXISTS ix_runs_target ON forecast_runs(system_id, period_start, source);

CREATE TABLE IF NOT EXISTS actual_telemetry (
    system_id   TEXT NOT NULL REFERENCES systems(system_id) ON DELETE CASCADE,
    period_start INTEGER NOT NULL,
    period_min   INTEGER NOT NULL,
    energy_wh    REAL NOT NULL,
    mean_watts   REAL NOT NULL,
    curtailed    INTEGER NOT NULL DEFAULT 0,
    inverter_temp_c REAL,
    origin       TEXT NOT NULL,
    PRIMARY KEY (system_id, period_start, period_min)
);

CREATE TABLE IF NOT EXISTS system_calibration (
    system_id   TEXT PRIMARY KEY REFERENCES systems(system_id) ON DELETE CASCADE,
    factor_c    REAL NOT NULL DEFAULT 1.0,
    clear_days  INTEGER NOT NULL,
    samples     INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS learned_shading (
    system_id    TEXT NOT NULL REFERENCES systems(system_id) ON DELETE CASCADE,
    az_bucket    INTEGER NOT NULL,
    el_bucket    INTEGER NOT NULL,
    k_beam       REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    day_count    INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (system_id, az_bucket, el_bucket)
);

CREATE TABLE IF NOT EXISTS model_registry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    system_id     TEXT NOT NULL REFERENCES systems(system_id) ON DELETE CASCADE,
    model_type    TEXT NOT NULL,
    trained_at    INTEGER NOT NULL,
    artifact_path TEXT NOT NULL,
    train_rows    INTEGER NOT NULL,
    holdout_rows  INTEGER NOT NULL,
    holdout_nmae_physics REAL NOT NULL,
    holdout_nmae_hybrid  REAL NOT NULL,
    active        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS forecast_solar_raw (
    system_id  TEXT NOT NULL,
    plane_id   TEXT NOT NULL,
    fetched_at INTEGER NOT NULL,
    http_status INTEGER NOT NULL,
    body_json  TEXT NOT NULL,
    PRIMARY KEY (system_id, plane_id, fetched_at)
);
"""

WEATHER_COLUMNS = [
    "ghi",
    "dni",
    "dhi",
    "temp_2m_c",
    "wind_10m_ms",
    "cloud_total",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "precip_mm",
    "snowfall_cm",
    "snow_depth_m",
]


def plane_id_for(system_id: str, dec: float, az: float, kwp: float) -> str:
    key = f"{system_id}|{dec:.1f}|{az:.1f}|{kwp:.3f}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._local = threading.local()
        self._lock = threading.Lock()

    # -- connection -----------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    async def close(self) -> None:
        def _close() -> None:
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                conn.close()
                self._local.conn = None

        await asyncio.to_thread(_close)

    # -- schema ---------------------------------------------------------
    def _init_sync(self) -> None:
        conn = self._connect()
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.executescript(SCHEMA_SQL)
            if version < 2:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(planes)")}
                if "inverter_id" not in columns:
                    conn.execute("ALTER TABLE planes ADD COLUMN inverter_id TEXT")
                conn.execute(
                    "INSERT OR IGNORE INTO inverters (system_id,inverter_id,ordinal,ac_kw) "
                    "SELECT system_id,'main',1,inverter_kw FROM systems WHERE inverter_kw IS NOT NULL"
                )
                conn.execute(
                    "UPDATE planes SET inverter_id='main' WHERE inverter_id IS NULL "
                    "AND system_id IN (SELECT system_id FROM inverters WHERE inverter_id='main')"
                )
            if version < SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            conn.commit()
        finally:
            conn.close()

    async def initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._init_sync)

    # -- generic helpers ------------------------------------------------
    def _execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.conn.execute(sql, tuple(params))

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchall()

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        await asyncio.to_thread(self._execute, sql, params)

    async def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        rows = await asyncio.to_thread(self._query, sql, params)
        return [dict(row) for row in rows]

    # -- systems --------------------------------------------------------
    def _upsert_system_sync(
        self,
        system: dict[str, Any],
        planes: list[dict[str, Any]],
        inverters: list[dict[str, Any]] | None = None,
    ) -> bool:
        now = int(time.time())
        if inverters is None:
            if system.get("inverter_kw") is not None:
                inverters = [
                    {"inverter_id": "main", "ac_kw": system["inverter_kw"]}
                ]
                planes = [{**plane, "inverter_id": "main"} for plane in planes]
            else:
                inverters = []
                planes = [{**plane, "inverter_id": None} for plane in planes]
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            existing = conn.execute(
                "SELECT * FROM systems WHERE system_id=?", (system["system_id"],)
            ).fetchone()
            old_planes = conn.execute(
                "SELECT dec, az_fs, kwp, inverter_id FROM planes "
                "WHERE system_id=? ORDER BY ordinal",
                (system["system_id"],),
            ).fetchall()
            old_inverters = conn.execute(
                "SELECT inverter_id, ac_kw FROM inverters WHERE system_id=? ORDER BY ordinal",
                (system["system_id"],),
            ).fetchall()
            new_geometry = sorted(
                (p["dec"], p["az_fs"], p["kwp"], p.get("inverter_id") or "")
                for p in planes
            )
            old_geometry = sorted(
                (p["dec"], p["az_fs"], p["kwp"], p["inverter_id"] or "")
                for p in old_planes
            )
            new_inverters = sorted(
                (inv["inverter_id"], float(inv["ac_kw"])) for inv in inverters
            )
            old_inverter_geometry = sorted(
                (inv["inverter_id"], float(inv["ac_kw"])) for inv in old_inverters
            )
            geometry_changed = existing is not None and (
                new_geometry != old_geometry
                or new_inverters != old_inverter_geometry
            )
            conn.execute(
                """
                INSERT INTO systems (system_id,name,lat,lon,elevation_m,timezone,inverter_kw,
                    system_loss,gamma_pdc,damping_morning,damping_evening,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(system_id) DO UPDATE SET
                    name=excluded.name, lat=excluded.lat, lon=excluded.lon,
                    elevation_m=COALESCE(excluded.elevation_m, systems.elevation_m),
                    timezone=excluded.timezone, inverter_kw=excluded.inverter_kw,
                    system_loss=excluded.system_loss, gamma_pdc=excluded.gamma_pdc,
                    damping_morning=excluded.damping_morning,
                    damping_evening=excluded.damping_evening, updated_at=excluded.updated_at
                """,
                (
                    system["system_id"],
                    system["name"],
                    system["lat"],
                    system["lon"],
                    system.get("elevation_m"),
                    system["timezone"],
                    system.get("inverter_kw"),
                    system.get("system_loss", 0.10),
                    system.get("gamma_pdc", -0.004),
                    system.get("damping_morning", 0.0),
                    system.get("damping_evening", 0.0),
                    now,
                    now,
                ),
            )
            conn.execute("DELETE FROM planes WHERE system_id=?", (system["system_id"],))
            conn.execute("DELETE FROM inverters WHERE system_id=?", (system["system_id"],))
            for ordinal, inverter in enumerate(inverters, start=1):
                conn.execute(
                    "INSERT INTO inverters (system_id,inverter_id,ordinal,ac_kw) VALUES (?,?,?,?)",
                    (
                        system["system_id"],
                        inverter["inverter_id"],
                        ordinal,
                        inverter["ac_kw"],
                    ),
                )
            for ordinal, plane in enumerate(planes, start=1):
                conn.execute(
                    "INSERT INTO planes "
                    "(plane_id,system_id,ordinal,dec,az_fs,kwp,inverter_id) VALUES (?,?,?,?,?,?,?)",
                    (
                        plane.get("plane_id")
                        or plane_id_for(system["system_id"], plane["dec"], plane["az_fs"], plane["kwp"]),
                        system["system_id"],
                        ordinal,
                        plane["dec"],
                        plane["az_fs"],
                        plane["kwp"],
                        plane.get("inverter_id"),
                    ),
                )
            if geometry_changed:
                conn.execute("DELETE FROM system_calibration WHERE system_id=?", (system["system_id"],))
                conn.execute("DELETE FROM learned_shading WHERE system_id=?", (system["system_id"],))
                conn.execute("DELETE FROM model_registry WHERE system_id=?", (system["system_id"],))
                LOG.info("geometry changed: reset C, K and MOS for %s", system["system_id"])
            conn.execute("COMMIT")
            return geometry_changed
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def upsert_system(
        self,
        system: dict[str, Any],
        planes: list[dict[str, Any]],
        inverters: list[dict[str, Any]] | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._upsert_system_sync, system, planes, inverters
        )

    def _get_system_sync(self, system_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM systems WHERE system_id=?", (system_id,)).fetchone()
        if row is None:
            return None
        system = dict(row)
        system["planes"] = [
            dict(plane)
            for plane in self.conn.execute(
                "SELECT * FROM planes WHERE system_id=? ORDER BY ordinal", (system_id,)
            )
        ]
        system["inverters"] = [
            dict(inverter)
            for inverter in self.conn.execute(
                "SELECT * FROM inverters WHERE system_id=? ORDER BY ordinal", (system_id,)
            )
        ]
        return system

    async def get_system(self, system_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_system_sync, system_id)

    async def list_systems(self) -> list[dict[str, Any]]:
        rows = await self.query("SELECT * FROM systems ORDER BY system_id")
        result = []
        for row in rows:
            system = await self.get_system(row["system_id"])
            if system:
                result.append(system)
        return result

    async def delete_system(self, system_id: str) -> bool:
        def _delete() -> bool:
            cur = self.conn.execute("DELETE FROM systems WHERE system_id=?", (system_id,))
            return cur.rowcount > 0

        return await asyncio.to_thread(_delete)

    async def set_elevation(self, system_id: str, elevation_m: float) -> None:
        await self.execute(
            "UPDATE systems SET elevation_m=?, updated_at=? WHERE system_id=?",
            (elevation_m, int(time.time()), system_id),
        )

    # -- horizon --------------------------------------------------------
    def _get_horizon_sync(self, lat_r: float, lon_r: float) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM horizon_cache WHERE lat_round=? AND lon_round=? ORDER BY azimuth_deg",
                (lat_r, lon_r),
            )
        ]

    async def get_horizon(self, lat: float, lon: float) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_horizon_sync, round(lat, 3), round(lon, 3))

    async def put_horizon(
        self, lat: float, lon: float, azimuth: Iterable[float], elevation: Iterable[float], source: str
    ) -> None:
        lat_r, lon_r = round(lat, 3), round(lon, 3)
        now = int(time.time())
        rows = [
            (lat_r, lon_r, float(az), float(el), source, now)
            for az, el in zip(azimuth, elevation)
        ]

        def _write() -> None:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM horizon_cache WHERE lat_round=? AND lon_round=?", (lat_r, lon_r)
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO horizon_cache "
                    "(lat_round,lon_round,azimuth_deg,elevation_deg,source,fetched_at) VALUES (?,?,?,?,?,?)",
                    rows,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        await asyncio.to_thread(_write)

    async def get_system_horizon(self, system_id: str) -> list[dict[str, Any]]:
        return await self.query(
            "SELECT * FROM system_horizon WHERE system_id=? ORDER BY azimuth_deg",
            (system_id,),
        )

    async def put_system_horizon(
        self,
        system_id: str,
        azimuth_deg: list[float],
        elevation_deg: list[float],
        updated_at: int,
    ) -> None:
        rows = [
            (system_id, float(az), float(el), int(updated_at))
            for az, el in zip(azimuth_deg, elevation_deg)
        ]

        def _write() -> None:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM system_horizon WHERE system_id=?", (system_id,)
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO system_horizon "
                    "(system_id,azimuth_deg,elevation_deg,updated_at) VALUES (?,?,?,?)",
                    rows,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        await asyncio.to_thread(_write)

    async def delete_system_horizon(self, system_id: str) -> bool:
        def _delete() -> bool:
            cur = self.conn.execute(
                "DELETE FROM system_horizon WHERE system_id=?", (system_id,)
            )
            return cur.rowcount > 0

        return await asyncio.to_thread(_delete)

    # -- weather --------------------------------------------------------
    async def store_weather(
        self, lat: float, lon: float, fetched_at: int, frame, model_ghi=None
    ) -> None:
        import pandas as pd

        lat_r, lon_r = round(lat, 3), round(lon, 3)
        rows = [
            (
                lat_r,
                lon_r,
                int(fetched_at),
                int(ts.timestamp()),
                *[None if pd.isna(row.get(col)) else float(row.get(col)) for col in WEATHER_COLUMNS],
            )
            for ts, row in frame.iterrows()
        ]
        model_rows = []
        if model_ghi is not None:
            for ts, row in model_ghi.iterrows():
                for model, ghi in row.items():
                    model_rows.append(
                        (
                            lat_r,
                            lon_r,
                            int(fetched_at),
                            int(ts.timestamp()),
                            str(model),
                            None if pd.isna(ghi) else float(ghi),
                        )
                    )

        def _write() -> None:
            self.conn.executemany(
                f"INSERT OR REPLACE INTO weather_forecast "
                f"(lat_round,lon_round,fetched_at,target_time,{','.join(WEATHER_COLUMNS)}) "
                f"VALUES ({','.join(['?'] * (4 + len(WEATHER_COLUMNS)))})",
                rows,
            )
            if model_rows:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO weather_model_ghi "
                    "(lat_round,lon_round,fetched_at,target_time,model,ghi) "
                    "VALUES (?,?,?,?,?,?)",
                    model_rows,
                )

        await asyncio.to_thread(_write)

    async def latest_weather_fetched_at(self, lat: float, lon: float) -> int | None:
        rows = await self.query(
            "SELECT MAX(fetched_at) AS ts FROM weather_forecast WHERE lat_round=? AND lon_round=?",
            (round(lat, 3), round(lon, 3)),
        )
        return rows[0]["ts"] if rows else None

    async def latest_weather(self, lat: float, lon: float):
        import pandas as pd

        ts = await self.latest_weather_fetched_at(lat, lon)
        if ts is None:
            return None, None
        rows = await self.query(
            "SELECT target_time,{} FROM weather_forecast "
            "WHERE lat_round=? AND lon_round=? AND fetched_at=? ORDER BY target_time".format(
                ",".join(WEATHER_COLUMNS)
            ),
            (round(lat, 3), round(lon, 3), ts),
        )
        if not rows:
            return None, ts
        frame = pd.DataFrame(rows)
        frame.index = pd.to_datetime(frame.pop("target_time"), unit="s", utc=True)
        return frame[WEATHER_COLUMNS].astype("float64"), ts

    async def load_weather(self, lat: float, lon: float, fetched_at: int):
        import pandas as pd

        rows = await self.query(
            "SELECT target_time,{} FROM weather_forecast "
            "WHERE lat_round=? AND lon_round=? AND fetched_at=? ORDER BY target_time".format(
                ",".join(WEATHER_COLUMNS)
            ),
            (round(lat, 3), round(lon, 3), fetched_at),
        )
        if not rows:
            return None
        frame = pd.DataFrame(rows)
        frame.index = pd.to_datetime(frame.pop("target_time"), unit="s", utc=True)
        return frame[WEATHER_COLUMNS].astype("float64")

    async def weather_fetched_at_for_target(self, lat: float, lon: float, target_ts: int) -> int | None:
        rows = await self.query(
            "SELECT MIN(fetched_at) AS ts FROM weather_forecast "
            "WHERE lat_round=? AND lon_round=? AND target_time=?",
            (round(lat, 3), round(lon, 3), target_ts),
        )
        return rows[0]["ts"] if rows else None

    async def get_weights(self, lat: float, lon: float) -> dict[str, float] | None:
        rows = await self.query(
            "SELECT model,weight FROM model_weights WHERE lat_round=? AND lon_round=?",
            (round(lat, 3), round(lon, 3)),
        )
        if not rows:
            return None
        return {row["model"]: float(row["weight"]) for row in rows}

    async def put_weights(
        self,
        lat: float,
        lon: float,
        rows: list[dict[str, Any]],
        updated_at: int,
    ) -> None:
        lat_r, lon_r = round(lat, 3), round(lon, 3)
        payload = [
            (
                lat_r,
                lon_r,
                str(row["model"]),
                float(row["weight"]),
                int(row["samples"]),
                int(updated_at),
            )
            for row in rows
        ]

        def _write() -> None:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM model_weights WHERE lat_round=? AND lon_round=?",
                    (lat_r, lon_r),
                )
                conn.executemany(
                    "INSERT INTO model_weights "
                    "(lat_round,lon_round,model,weight,samples,updated_at) VALUES (?,?,?,?,?,?)",
                    payload,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        await asyncio.to_thread(_write)

    async def weights_updated_at(self, lat: float, lon: float) -> int | None:
        rows = await self.query(
            "SELECT MAX(updated_at) AS ts FROM model_weights "
            "WHERE lat_round=? AND lon_round=?",
            (round(lat, 3), round(lon, 3)),
        )
        return rows[0]["ts"] if rows else None

    async def model_ghi_latest(
        self, lat: float, lon: float, start_ts: int
    ) -> list[dict[str, Any]]:
        lat_r, lon_r = round(lat, 3), round(lon, 3)
        return await self.query(
            """
            SELECT m.target_time, m.model, m.ghi
            FROM weather_model_ghi m
            JOIN (
                SELECT target_time, model, MAX(fetched_at) AS fetched_at
                FROM weather_model_ghi
                WHERE lat_round=? AND lon_round=? AND target_time>=?
                GROUP BY target_time, model
            ) latest
              ON latest.target_time=m.target_time
             AND latest.model=m.model
             AND latest.fetched_at=m.fetched_at
            WHERE m.lat_round=? AND m.lon_round=?
            ORDER BY m.target_time, m.model
            """,
            (lat_r, lon_r, int(start_ts), lat_r, lon_r),
        )

    # -- forecast runs --------------------------------------------------
    async def insert_runs(self, rows: Iterable[dict[str, Any]]) -> None:
        payload = [
            (
                row["system_id"],
                int(row["fetched_at"]),
                int(row["period_start"]),
                int(row["period_min"]),
                float(row["lead_hours"]),
                row["source"],
                float(row["energy_wh"]),
                float(row["mean_watts"]),
            )
            for row in rows
        ]
        if not payload:
            return
        await asyncio.to_thread(
            self._executemany,
            "INSERT OR REPLACE INTO forecast_runs "
            "(system_id,fetched_at,period_start,period_min,lead_hours,source,energy_wh,mean_watts) "
            "VALUES (?,?,?,?,?,?,?,?)",
            payload,
        )

    async def runs(
        self, system_id: str, source: str, start_ts: int | None = None, end_ts: int | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM forecast_runs WHERE system_id=? AND source=?"
        params: list[Any] = [system_id, source]
        if start_ts is not None:
            sql += " AND period_start>=?"
            params.append(start_ts)
        if end_ts is not None:
            sql += " AND period_start<?"
            params.append(end_ts)
        sql += " ORDER BY period_start, fetched_at"
        return await self.query(sql, params)

    async def latest_run_fetched_at(self, system_id: str, source: str) -> int | None:
        rows = await self.query(
            "SELECT MAX(fetched_at) AS ts FROM forecast_runs WHERE system_id=? AND source=?",
            (system_id, source),
        )
        return rows[0]["ts"] if rows else None

    # -- telemetry ------------------------------------------------------
    def _upsert_telemetry_sync(self, system_id: str, items: list[dict[str, Any]]) -> int:
        conn = self.conn
        accepted = 0
        conn.execute("BEGIN IMMEDIATE")
        try:
            for item in items:
                existing = conn.execute(
                    "SELECT origin FROM actual_telemetry WHERE system_id=? AND period_start=? AND period_min=?",
                    (system_id, item["period_start"], item["period_min"]),
                ).fetchone()
                if existing and existing["origin"] == "push" and item["origin"] == "backfill":
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO actual_telemetry "
                    "(system_id,period_start,period_min,energy_wh,mean_watts,curtailed,inverter_temp_c,origin) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        system_id,
                        item["period_start"],
                        item["period_min"],
                        item["energy_wh"],
                        item["mean_watts"],
                        int(bool(item.get("curtailed", False))),
                        item.get("inverter_temp_c"),
                        item["origin"],
                    ),
                )
                accepted += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return accepted

    async def upsert_telemetry(self, system_id: str, items: list[dict[str, Any]]) -> int:
        return await asyncio.to_thread(self._upsert_telemetry_sync, system_id, items)

    async def telemetry(
        self, system_id: str, start_ts: int | None = None, end_ts: int | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM actual_telemetry WHERE system_id=?"
        params: list[Any] = [system_id]
        if start_ts is not None:
            sql += " AND period_start>=?"
            params.append(start_ts)
        if end_ts is not None:
            sql += " AND period_start<?"
            params.append(end_ts)
        sql += " ORDER BY period_start"
        return await self.query(sql, params)

    async def last_telemetry(self, system_id: str) -> dict[str, Any] | None:
        rows = await self.query(
            "SELECT * FROM actual_telemetry WHERE system_id=? ORDER BY period_start DESC LIMIT 1",
            (system_id,),
        )
        return rows[0] if rows else None

    # -- learning state -------------------------------------------------
    async def get_calibration(self, system_id: str) -> dict[str, Any] | None:
        rows = await self.query(
            "SELECT * FROM system_calibration WHERE system_id=?", (system_id,)
        )
        return rows[0] if rows else None

    async def put_calibration(
        self, system_id: str, factor_c: float, clear_days: int, samples: int, updated_at: int
    ) -> None:
        await self.execute(
            "INSERT OR REPLACE INTO system_calibration "
            "(system_id,factor_c,clear_days,samples,updated_at) VALUES (?,?,?,?,?)",
            (system_id, factor_c, clear_days, samples, updated_at),
        )

    async def get_shading(self, system_id: str) -> list[dict[str, Any]]:
        return await self.query(
            "SELECT * FROM learned_shading WHERE system_id=?", (system_id,)
        )

    def _replace_shading_sync(self, system_id: str, rows: list[dict[str, Any]]) -> None:
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM learned_shading WHERE system_id=?", (system_id,))
            conn.executemany(
                "INSERT INTO learned_shading "
                "(system_id,az_bucket,el_bucket,k_beam,sample_count,day_count,updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        system_id,
                        row["az_bucket"],
                        row["el_bucket"],
                        row["k_beam"],
                        row["sample_count"],
                        row["day_count"],
                        row["updated_at"],
                    )
                    for row in rows
                ],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def replace_shading(self, system_id: str, rows: list[dict[str, Any]]) -> None:
        await asyncio.to_thread(self._replace_shading_sync, system_id, rows)

    async def active_model(self, system_id: str) -> dict[str, Any] | None:
        rows = await self.query(
            "SELECT * FROM model_registry WHERE system_id=? AND active=1 "
            "ORDER BY trained_at DESC LIMIT 1",
            (system_id,),
        )
        return rows[0] if rows else None

    async def register_model(self, record: dict[str, Any]) -> int:
        def _write() -> int:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                if record.get("active"):
                    conn.execute(
                        "UPDATE model_registry SET active=0 WHERE system_id=? AND model_type=?",
                        (record["system_id"], record["model_type"]),
                    )
                cur = conn.execute(
                    "INSERT INTO model_registry (system_id,model_type,trained_at,artifact_path,"
                    "train_rows,holdout_rows,holdout_nmae_physics,holdout_nmae_hybrid,active) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        record["system_id"],
                        record["model_type"],
                        int(record["trained_at"]),
                        record["artifact_path"],
                        int(record["train_rows"]),
                        int(record["holdout_rows"]),
                        float(record["holdout_nmae_physics"]),
                        float(record["holdout_nmae_hybrid"]),
                        1 if record.get("active") else 0,
                    ),
                )
                conn.execute("COMMIT")
                return int(cur.lastrowid)
            except Exception:
                conn.execute("ROLLBACK")
                raise

        return await asyncio.to_thread(_write)

    async def deactivate_models(self, system_id: str, model_type: str = "mos_lgbm") -> None:
        await self.execute(
            "UPDATE model_registry SET active=0 WHERE system_id=? AND model_type=?",
            (system_id, model_type),
        )

    async def training_join(self, system_id: str) -> list[dict[str, Any]]:
        """Telemetry joined to physics runs and their weather snapshots."""
        return await self.query(
            """
            SELECT t.period_start, t.energy_wh, t.mean_watts, t.period_min,
                   r.fetched_at, r.energy_wh AS physics_wh, r.mean_watts AS physics_w,
                   (r.period_start-r.fetched_at)/3600.0 AS lead_hours,
                   w.ghi, w.dni, w.dhi, w.temp_2m_c, w.wind_10m_ms,
                   w.cloud_total, w.cloud_low, w.cloud_mid, w.cloud_high,
                   w.precip_mm, w.snowfall_cm, w.snow_depth_m
            FROM actual_telemetry t
            JOIN forecast_runs r
              ON r.system_id=t.system_id AND r.period_start=t.period_start AND r.source='physics'
            JOIN systems s ON s.system_id=t.system_id
            JOIN weather_forecast w
              ON w.lat_round=ROUND(s.lat,3) AND w.lon_round=ROUND(s.lon,3)
             AND w.fetched_at=r.fetched_at AND w.target_time=t.period_start
            WHERE t.system_id=? AND t.curtailed=0 AND t.period_min=15
            ORDER BY t.period_start, r.fetched_at
            """,
            (system_id,),
        )

    # -- forecast.solar shadow -----------------------------------------
    async def insert_forecast_solar_raw(
        self, system_id: str, plane_id: str, fetched_at: int, status: int, body: str
    ) -> None:
        await self.execute(
            "INSERT OR REPLACE INTO forecast_solar_raw "
            "(system_id,plane_id,fetched_at,http_status,body_json) VALUES (?,?,?,?,?)",
            (system_id, plane_id, fetched_at, status, body),
        )

    def _executemany(self, sql: str, payload) -> None:
        self.conn.executemany(sql, payload)

    async def insert_forecast_solar_runs(
        self, system_id: str, fetched_at: int, rows: Iterable[dict[str, Any]]
    ) -> None:
        payload = [
            (
                system_id,
                fetched_at,
                int(row["period_start"]),
                60,
                (row["period_start"] - fetched_at) / 3600.0,
                "forecast_solar",
                float(row["energy_wh"]),
                float(row["mean_watts"]),
            )
            for row in rows
        ]
        if payload:
            await asyncio.to_thread(
                self._executemany,
                "INSERT OR REPLACE INTO forecast_runs "
                "(system_id,fetched_at,period_start,period_min,lead_hours,source,energy_wh,mean_watts) "
                "VALUES (?,?,?,?,?,?,?,?)",
                payload,
            )

    # -- housekeeping ---------------------------------------------------
    async def housekeeping(self, now_ts: int) -> dict[str, int]:
        def _run() -> dict[str, int]:
            conn = self.conn
            day = 86400
            deleted = {}
            cur = conn.execute(
                "DELETE FROM weather_forecast WHERE fetched_at<?", (now_ts - 400 * day,)
            )
            deleted["weather_forecast"] = cur.rowcount
            cur = conn.execute(
                "DELETE FROM weather_model_ghi WHERE fetched_at<?", (now_ts - 400 * day,)
            )
            deleted["weather_model_ghi"] = cur.rowcount
            cur = conn.execute(
                "DELETE FROM forecast_runs WHERE fetched_at<?", (now_ts - 400 * day,)
            )
            deleted["forecast_runs"] = cur.rowcount
            cur = conn.execute(
                "DELETE FROM forecast_solar_raw WHERE fetched_at<?", (now_ts - 30 * day,)
            )
            deleted["forecast_solar_raw"] = cur.rowcount
            return deleted

        return await asyncio.to_thread(_run)

    async def vacuum(self) -> None:
        await asyncio.to_thread(lambda: self.conn.execute("VACUUM"))


def query_dataframe(rows: list[dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(rows)
