import sqlite3

import pytest

from app.core.database import Database


@pytest.mark.asyncio
async def test_v1_database_migrates_to_v2_idempotently(tmp_path):
    path = tmp_path / "v1.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE systems (
            system_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            elevation_m REAL,
            timezone TEXT NOT NULL,
            inverter_kw REAL,
            system_loss REAL NOT NULL DEFAULT 0.10,
            gamma_pdc REAL NOT NULL DEFAULT -0.004,
            damping_morning REAL NOT NULL DEFAULT 0.0,
            damping_evening REAL NOT NULL DEFAULT 0.0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE planes (
            plane_id TEXT PRIMARY KEY,
            system_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            dec REAL NOT NULL,
            az_fs REAL NOT NULL,
            kwp REAL NOT NULL,
            UNIQUE (system_id, ordinal)
        );
        INSERT INTO systems VALUES
            ('home','Home',44.8125,20.4612,NULL,'Europe/Belgrade',5.0,0.1,-0.004,0,0,1,1);
        INSERT INTO planes VALUES ('plane1','home',1,30,0,5.0);
        PRAGMA user_version=1;
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    await db.initialize()
    stored = await db.get_system("home")
    assert stored is not None
    assert stored["inverters"] == [
        {
            "system_id": "home",
            "inverter_id": "main",
            "ordinal": 1,
            "ac_kw": 5.0,
        }
    ]
    assert stored["planes"][0]["inverter_id"] == "main"
    assert (await db.query("PRAGMA user_version"))[0]["user_version"] == 2

    await db.initialize()
    stored = await db.get_system("home")
    assert len(stored["inverters"]) == 1
    assert stored["planes"][0]["inverter_id"] == "main"
