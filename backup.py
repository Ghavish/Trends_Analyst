"""
backup.py - copies what can't be downloaded again to another drive.

    python backup.py           # run it weekly (the engine may keep running)
    python backup.py --list    # show what is in the backup folder

What is backed up (into config.BACKUP_DIR, set in .env):
  * the SQLite database -> database/trends_analyst_<YYYY-MM-DD_HHMM>UTC.db
      made with SQLite's backup API (through db.py), checked for integrity;
      the newest BACKUP_KEEP_DB_COPIES copies are kept
  * the IG recordings   -> ig_recordings/ticks/... and ig_recordings/bars_1m/...
      the same layout as data/ticks and data/bars_1m. New files are copied;
      unchanged files are skipped; files of the last BACKUP_LIVE_DAYS days
      (still being written by the engine) are refreshed. Nothing is ever
      deleted from the backup. An older file that has changed on the laptop
      (it shouldn't) is saved NEXT TO the backup copy, never over it.

Not backed up: the Dukascopy history (it can be downloaded again; its ticks
are already on the other drive) and models (they can be retrained).
Every copy is written to a temporary file, checked (size, and that Parquet
files can be read), and only then given its real name.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

import config
import db
import timeutils

DB_BACKUP_PREFIX = "trends_analyst_"


class BackupError(Exception):
    """The backup can't run (no folder set, drive missing ...)."""


def backup_dir() -> Path:
    """The backup folder from .env; it must exist (stops if the drive is missing)."""
    path = config.BACKUP_DIR
    if path is None:
        raise BackupError("No backup folder set. Add a line to .env (no quotes), e.g. "
                          "BACKUP_DIR=D:\\TrendsAnalyst\\backups, and create that folder.")
    if any(ord(c) < 32 for c in str(path)):
        raise BackupError(f"BACKUP_DIR in .env contains a hidden control character ({str(path)!r}). "
                          "Write the path without quotes.")
    if not path.is_dir():
        raise BackupError(f"The backup folder {path} can't be found. Is the drive plugged in, with the "
                          f'same letter? If this is its first use, create it: New-Item -ItemType Directory "{path}"')
    return path


def same_drive_warning(path: Path) -> str | None:
    if Path(path).resolve().anchor.lower() == config.PROJECT_ROOT.resolve().anchor.lower():
        return (f"WARNING: the backup folder is on the same drive as the project ({Path(path).anchor}); "
                "it won't help if that drive fails.")
    return None


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def backup_database(dest_dir: Path, now: datetime | None = None) -> dict:
    now = timeutils.ensure_utc(now or timeutils.now_utc())
    dest = dest_dir / "database" / f"{DB_BACKUP_PREFIX}{now:%Y-%m-%d_%H%M}UTC.db"
    result = db.backup_database(dest)
    # Keep only the newest copies (names sort by date).
    copies = sorted((dest_dir / "database").glob(f"{DB_BACKUP_PREFIX}*UTC.db"))
    removed = []
    for old in copies[:-config.BACKUP_KEEP_DB_COPIES]:
        old.unlink()
        removed.append(old.name)
    return {**result, "path": dest, "removed": removed}


# --------------------------------------------------------------------------
# IG recordings (Parquet files)
# --------------------------------------------------------------------------
def _file_day(path: Path) -> date | None:
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d").date()
    except ValueError:
        return None


def _copy_checked(src: Path, dest: Path) -> int:
    """Copy via a temporary file; check size and that Parquet can read it; then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copy2(src, tmp)                         # copy2 keeps the modified time
    size = src.stat().st_size
    try:
        if tmp.stat().st_size != size:
            raise BackupError(f"copy of {src} came out a different size")
        pq.read_metadata(tmp)                      # raises if the file isn't valid Parquet
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return size


def _unchanged(src: Path, dest: Path) -> bool:
    a, b = src.stat(), dest.stat()
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


def backup_recordings(dest_dir: Path, now: datetime | None = None) -> dict:
    now = timeutils.ensure_utc(now or timeutils.now_utc())
    live_from = now.date() - timedelta(days=config.BACKUP_LIVE_DAYS - 1)
    res = {"new": 0, "updated": 0, "unchanged": 0, "kept_both": [], "bytes": 0}
    for name, src_root in (("ticks", config.TICKS_DIR), ("bars_1m", config.BARS_1M_DIR)):
        if not Path(src_root).exists():
            continue
        for src in sorted(Path(src_root).rglob("*.parquet")):
            dest = dest_dir / "ig_recordings" / name / src.relative_to(src_root)
            if not dest.exists():
                res["bytes"] += _copy_checked(src, dest)
                res["new"] += 1
            elif _unchanged(src, dest):
                res["unchanged"] += 1
            else:
                day = _file_day(src)
                if day is not None and day >= live_from:
                    res["bytes"] += _copy_checked(src, dest)    # still being written: refresh
                    res["updated"] += 1
                elif any(_unchanged(src, s) for s in dest.parent.glob(f"{dest.stem}.changed-*{dest.suffix}")):
                    res["unchanged"] += 1                         # this version was already saved beside it
                else:
                    # A finished day changed: keep the backup, save the new version beside it.
                    side = dest.with_name(f"{dest.stem}.changed-{now:%Y%m%dT%H%M}UTC{dest.suffix}")
                    res["bytes"] += _copy_checked(src, side)
                    res["kept_both"].append(str(side.relative_to(dest_dir)))
    return res


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def run_backup(now: datetime | None = None) -> list[str]:
    target = backup_dir()
    now = timeutils.ensure_utc(now or timeutils.now_utc())
    lines = [f"Backup to {target} at {timeutils.fmt_dual(now, with_date=True)}"]
    warning = same_drive_warning(target)
    if warning:
        lines.append(warning)
    d = backup_database(target, now)
    lines.append(f"Database: {d['path'].name}, {d['bytes'] / 1e6:.1f} MB, integrity {d['integrity']}"
                 + (f"; removed {len(d['removed'])} old copy(ies)" if d["removed"] else ""))
    r = backup_recordings(target, now)
    lines.append(f"IG recordings: {r['new']} new, {r['updated']} refreshed, {r['unchanged']} unchanged "
                 f"({r['bytes'] / 1e6:.1f} MB copied)")
    for side in r["kept_both"]:
        lines.append(f"  NOTE: a finished day changed on the laptop; both versions kept: {side}")
    return lines


def list_backup() -> list[str]:
    target = backup_dir()
    lines = [f"Backup folder {target}"]
    copies = sorted((target / "database").glob(f"{DB_BACKUP_PREFIX}*UTC.db"))
    lines.append(f"Database copies ({len(copies)}):")
    lines += [f"  {c.name}  {c.stat().st_size / 1e6:.1f} MB" for c in copies] or ["  (none)"]
    for name in ("ticks", "bars_1m"):
        root = target / "ig_recordings" / name
        files = list(root.rglob("*.parquet")) if root.exists() else []
        size = sum(f.stat().st_size for f in files)
        lines.append(f"IG {name}: {len(files)} files, {size / 1e6:.1f} MB")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up the IG recordings and the database.")
    parser.add_argument("--list", action="store_true", help="show what is in the backup folder")
    args = parser.parse_args(argv)
    try:
        print("\n".join(list_backup() if args.list else run_backup()))
        return 0
    except (BackupError, db.DBError) as exc:
        print(f"STOPPED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
