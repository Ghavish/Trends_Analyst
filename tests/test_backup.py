"""
Tests for backup.py and db.backup_database(). Temporary folders only.
"""

import os
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import backup
import config
import db

UTC = timezone.utc
NOW = datetime(2026, 11, 16, 9, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "TICKS_DIR", tmp_path / "data" / "ticks")
    monkeypatch.setattr(config, "BARS_1M_DIR", tmp_path / "data" / "bars_1m")
    target = tmp_path / "D_drive" / "backups"
    target.mkdir(parents=True)
    monkeypatch.setattr(config, "BACKUP_DIR", target)
    yield tmp_path


def write_recording(kind: str, inst: str, day: date, rows: int = 3) -> Path:
    root = config.TICKS_DIR if kind == "ticks" else config.BARS_1M_DIR
    path = root / inst / f"{day.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_utc": pd.date_range(str(day), periods=rows, freq="1min", tz="UTC"),
                  "bid": [1.08] * rows}).to_parquet(path, index=False)
    return path


def _touch_later(path: Path, seconds: int = 120):
    st = path.stat()
    os.utime(path, (st.st_atime + seconds, st.st_mtime + seconds))


# --------------------------------------------------------------------------
# Folder checks
# --------------------------------------------------------------------------
def test_no_folder_set(monkeypatch):
    monkeypatch.setattr(config, "BACKUP_DIR", None)
    with pytest.raises(backup.BackupError, match="No backup folder set"):
        backup.backup_dir()


def test_missing_drive_stops_and_creates_nothing(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "E_drive" / "backups"
    monkeypatch.setattr(config, "BACKUP_DIR", missing)
    assert backup.main([]) == 1
    assert "plugged in" in capsys.readouterr().out
    assert not missing.exists()


def test_quoted_path_with_tab_is_explained(monkeypatch):
    monkeypatch.setattr(config, "BACKUP_DIR", Path("D:\\TrendsAnalyst\\backups\today"))   # "\t" is a tab
    with pytest.raises(backup.BackupError, match="without quotes"):
        backup.backup_dir()


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def test_database_backup_is_complete_and_self_contained():
    db.set_kill_switch("before backup", "cli")
    res = backup.backup_database(config.BACKUP_DIR, NOW)
    assert res["integrity"] == "ok" and res["schema_version"] == len(db.MIGRATIONS)
    assert res["path"].name == "trends_analyst_2026-11-16_0930UTC.db"
    assert not Path(str(res["path"]) + "-wal").exists()           # one self-contained file
    with closing(sqlite3.connect(res["path"])) as conn:
        assert conn.execute("SELECT is_on FROM flags WHERE name='KILL_SWITCH'").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_database_backup_while_another_connection_writes():
    # WAL mode: an open write elsewhere must not stop or corrupt the backup.
    db.init_db()
    with closing(sqlite3.connect(config.DB_PATH)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO engine_status (key, value, updated_utc) VALUES ('x', '1', 'now')")
        res = backup.backup_database(config.BACKUP_DIR, NOW)          # sees the last committed state
        writer.rollback()
    assert res["integrity"] == "ok"


def test_only_newest_database_copies_kept(monkeypatch):
    monkeypatch.setattr(config, "BACKUP_KEEP_DB_COPIES", 3)
    for week in range(5):
        backup.backup_database(config.BACKUP_DIR, NOW + timedelta(weeks=week))
    names = sorted(p.name for p in (config.BACKUP_DIR / "database").iterdir())
    assert names == ["trends_analyst_2026-11-30_0930UTC.db", "trends_analyst_2026-12-07_0930UTC.db",
                     "trends_analyst_2026-12-14_0930UTC.db"]


# --------------------------------------------------------------------------
# IG recordings
# --------------------------------------------------------------------------
def test_recordings_copied_then_skipped():
    write_recording("ticks", "EURUSD", date(2026, 11, 10))
    write_recording("bars_1m", "EURUSD", date(2026, 11, 10))
    first = backup.backup_recordings(config.BACKUP_DIR, NOW)
    assert first["new"] == 2 and first["bytes"] > 0
    copy = config.BACKUP_DIR / "ig_recordings" / "ticks" / "EURUSD" / "2026-11-10.parquet"
    assert len(pd.read_parquet(copy)) == 3
    second = backup.backup_recordings(config.BACKUP_DIR, NOW)
    assert (second["new"], second["updated"], second["unchanged"]) == (0, 0, 2)


def test_todays_file_is_refreshed():
    path = write_recording("ticks", "EURUSD", NOW.date(), rows=3)
    backup.backup_recordings(config.BACKUP_DIR, NOW)
    write_recording("ticks", "EURUSD", NOW.date(), rows=10)       # the engine added prices
    _touch_later(path)
    res = backup.backup_recordings(config.BACKUP_DIR, NOW)
    assert res["updated"] == 1
    copy = config.BACKUP_DIR / "ig_recordings" / "ticks" / "EURUSD" / f"{NOW.date()}.parquet"
    assert len(pd.read_parquet(copy)) == 10


def test_changed_old_file_never_overwrites_backup():
    old_day = NOW.date() - timedelta(days=10)
    path = write_recording("ticks", "EURUSD", old_day, rows=3)
    backup.backup_recordings(config.BACKUP_DIR, NOW)
    write_recording("ticks", "EURUSD", old_day, rows=1)           # e.g. damaged on the laptop
    _touch_later(path)
    res = backup.backup_recordings(config.BACKUP_DIR, NOW)
    folder = config.BACKUP_DIR / "ig_recordings" / "ticks" / "EURUSD"
    assert len(pd.read_parquet(folder / f"{old_day}.parquet")) == 3        # original kept
    assert len(res["kept_both"]) == 1
    side = list(folder.glob(f"{old_day}.changed-*.parquet"))
    assert len(side) == 1 and len(pd.read_parquet(side[0])) == 1
    again = backup.backup_recordings(config.BACKUP_DIR, NOW + timedelta(weeks=1))
    assert again["kept_both"] == [] and again["unchanged"] == 1            # not saved twice


def test_nothing_deleted_from_backup():
    path = write_recording("ticks", "EURUSD", date(2026, 11, 10))
    backup.backup_recordings(config.BACKUP_DIR, NOW)
    path.unlink()                                                  # gone from the laptop
    backup.backup_recordings(config.BACKUP_DIR, NOW)
    assert (config.BACKUP_DIR / "ig_recordings" / "ticks" / "EURUSD" / "2026-11-10.parquet").exists()


def test_damaged_source_file_is_not_given_a_real_name():
    bad = config.TICKS_DIR / "EURUSD" / "2026-11-10.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not parquet")
    with pytest.raises(Exception):
        backup.backup_recordings(config.BACKUP_DIR, NOW)
    folder = config.BACKUP_DIR / "ig_recordings" / "ticks" / "EURUSD"
    assert not (folder / "2026-11-10.parquet").exists()
    assert not list(folder.glob("*.tmp"))


def test_full_run_and_list(capsys):
    write_recording("ticks", "GBPUSD", date(2026, 11, 12))
    assert backup.main([]) == 0
    out = capsys.readouterr().out
    assert "integrity ok" in out and "1 new" in out
    assert backup.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "Database copies (1)" in out and "IG ticks: 1 files" in out
