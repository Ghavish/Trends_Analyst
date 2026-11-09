"""
db.py - the ONLY file that talks to SQLite (Section 10.2, design rule).

The engine (signals.py) and the dashboard (app.py) are separate programs that
share one SQLite file (config.DB_PATH). The rest of the project only calls the
functions below; if the database ever changes (Section 13.9) only this file
needs rewriting.

Who may call what:
  * dashboard (app.py): the get_/list_ read functions, trading_block_reasons(),
    and request_job() - nothing else (it only reads and writes requests).
  * killswitch.py (command line): set_kill_switch / clear_kill_switch /
    clear_halt, so the kill switch works even if the dashboard is down.
  * engine: everything.

Tables:
  flags             KILL_SWITCH (manual) and HALT (automatic) - each on/off
  flag_events       every activation / deactivation: time, reason, source
  jobs              requests from the dashboard; the engine updates progress/status
  variants          model variants and their live role: ACTIVE / SHADOW / OFF
  variant_role_log  every role change (Section 10.6 "Make active")
  predictions       every signal (active and shadow), with its outcome
  gaps              periods with missing / possibly missing prices (step 5)
  feed_status       latest price and market status per instrument (for the dashboard)
  market_status_log every change of IG's market status (DLG_FLAG)
  engine_status     engine heartbeat and stream status (is the engine running?)
  market_info       contract details from IG (pip size, decimals, min deal ...)
  history_days      which instrument-days of history are fully downloaded (resume)
  final_check_access every read of the final-check period (Section 7.9)

Price bars are NOT here: they are stored in Parquet files (fetch_data.py).

All times are stored as UTC text in ISO 8601, e.g. 2026-11-09T11:40:42.909+00:00.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

import config
import timeutils

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
KILL_SWITCH = "KILL_SWITCH"
HALT = "HALT"

SOURCES = ("cli", "dashboard", "engine")
ROLES = ("ACTIVE", "SHADOW", "OFF")

# Jobs the dashboard may request. The engine processes them as features are
# built. (No "close all positions" job until orders exist in Phase 3.)
JOB_KILL_SWITCH_ON = "KILL_SWITCH_ON"
JOB_KILL_SWITCH_OFF = "KILL_SWITCH_OFF"
JOB_CLEAR_HALT = "CLEAR_HALT"
JOB_SET_VARIANT_ROLE = "SET_VARIANT_ROLE"
JOB_RETRAIN = "RETRAIN"
JOB_RUN_COMPARISON = "RUN_COMPARISON"
JOB_REFRESH_DATA = "REFRESH_DATA"
JOB_KINDS = (JOB_KILL_SWITCH_ON, JOB_KILL_SWITCH_OFF, JOB_CLEAR_HALT, JOB_SET_VARIANT_ROLE,
             JOB_RETRAIN, JOB_RUN_COMPARISON, JOB_REFRESH_DATA)
JOB_STATUSES = ("PENDING", "RUNNING", "DONE", "FAILED", "CANCELLED")


class DBError(Exception):
    """Invalid request to the database layer (bad value, unknown name...)."""


# --------------------------------------------------------------------------
# Schema and migrations
# --------------------------------------------------------------------------
# Each entry upgrades the schema by one version. Never edit an old entry once
# it has run on your laptop; add a new one instead (PRAGMA user_version
# records how far a database file has been upgraded).
MIGRATIONS = [
    # ---- version 1 ----
    """
    CREATE TABLE flags (
        name         TEXT PRIMARY KEY CHECK (name IN ('KILL_SWITCH', 'HALT')),
        is_on        INTEGER NOT NULL DEFAULT 0 CHECK (is_on IN (0, 1)),
        reason       TEXT,
        source       TEXT,
        changed_utc  TEXT,
        expires_utc  TEXT            -- HALT only: when it ends by itself (NULL = manual clear)
    );
    INSERT INTO flags (name, is_on) VALUES ('KILL_SWITCH', 0), ('HALT', 0);

    CREATE TABLE flag_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        time_utc     TEXT NOT NULL,
        flag         TEXT NOT NULL,
        action       TEXT NOT NULL CHECK (action IN ('ON', 'OFF')),
        reason       TEXT NOT NULL,
        source       TEXT NOT NULL,
        expires_utc  TEXT
    );

    CREATE TABLE jobs (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        kind           TEXT NOT NULL,
        params         TEXT NOT NULL DEFAULT '{}',   -- JSON
        status         TEXT NOT NULL DEFAULT 'PENDING'
                       CHECK (status IN ('PENDING', 'RUNNING', 'DONE', 'FAILED', 'CANCELLED')),
        progress       REAL NOT NULL DEFAULT 0,      -- 0..1
        message        TEXT,
        result         TEXT,                          -- JSON
        requested_by   TEXT NOT NULL,
        requested_utc  TEXT NOT NULL,
        started_utc    TEXT,
        finished_utc   TEXT
    );
    CREATE INDEX jobs_by_status ON jobs (status, id);

    CREATE TABLE variants (
        name              TEXT PRIMARY KEY,
        feature_set       TEXT NOT NULL,
        description       TEXT,
        model_file        TEXT,
        model_version     TEXT,
        role              TEXT NOT NULL DEFAULT 'OFF' CHECK (role IN ('ACTIVE', 'SHADOW', 'OFF')),
        created_utc       TEXT NOT NULL,
        role_changed_utc  TEXT
    );
    -- At most one ACTIVE variant, enforced by the database itself.
    CREATE UNIQUE INDEX variants_one_active ON variants (role) WHERE role = 'ACTIVE';

    CREATE TABLE variant_role_log (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        time_utc  TEXT NOT NULL,
        variant   TEXT NOT NULL,
        old_role  TEXT,
        new_role  TEXT NOT NULL,
        source    TEXT NOT NULL,
        reason    TEXT
    );

    CREATE TABLE predictions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        created_utc      TEXT NOT NULL,
        variant          TEXT NOT NULL,
        model_version    TEXT,
        is_shadow        INTEGER NOT NULL CHECK (is_shadow IN (0, 1)),
        instrument       TEXT NOT NULL,           -- key in config.INSTRUMENTS, e.g. EURUSD
        epic             TEXT NOT NULL,
        bar_time_utc     TEXT NOT NULL,           -- close time of the completed bar used
        direction        TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
        entry            REAL NOT NULL,
        stop_loss        REAL NOT NULL,
        take_profit      REAL NOT NULL,
        atr              REAL,
        probability      REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
        expected_value_r REAL,
        spread           REAL,
        position_size    REAL,
        risk_pct         REAL,
        close_by_utc     TEXT,                    -- time limit (end of session)
        pattern          TEXT,                    -- candlestick pattern note for the trade card
        blocked_reason   TEXT,                    -- set if filters/limits stopped it being a trade card
        status           TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED')),
        outcome          TEXT CHECK (outcome IN ('TP', 'SL', 'TIMEOUT')),
        exit_price       REAL,
        exit_time_utc    TEXT,
        result_r         REAL
    );
    CREATE INDEX predictions_open ON predictions (status);
    CREATE INDEX predictions_by_variant ON predictions (variant, bar_time_utc);
    """,
    # ---- version 2: price recording (Phase 1, step 5) ----
    """
    CREATE TABLE gaps (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        instrument   TEXT NOT NULL,
        kind         TEXT NOT NULL CHECK (kind IN ('OUTAGE', 'QUIET', 'ENGINE_OFF')),
        start_utc    TEXT NOT NULL,
        end_utc      TEXT,                 -- NULL while the gap is still open
        detail       TEXT
    );
    CREATE INDEX gaps_by_start ON gaps (start_utc);
    CREATE INDEX gaps_open ON gaps (end_utc);

    CREATE TABLE feed_status (
        instrument         TEXT PRIMARY KEY,
        epic               TEXT NOT NULL,
        last_tick_utc      TEXT,
        bid                REAL,
        offer              REAL,
        market_status      TEXT,           -- IG DLG_FLAG, e.g. DEAL / CLOSED
        market_status_utc  TEXT,
        ticks_today        INTEGER,
        updated_utc        TEXT NOT NULL
    );

    CREATE TABLE market_status_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        time_utc    TEXT NOT NULL,
        instrument  TEXT NOT NULL,
        status      TEXT NOT NULL
    );

    CREATE TABLE engine_status (
        key          TEXT PRIMARY KEY,     -- e.g. heartbeat, stream_status, state
        value        TEXT,
        updated_utc  TEXT NOT NULL
    );
    """,
    # ---- version 3: contract details from IG (read by the engine at start) ----
    """
    CREATE TABLE market_info (
        instrument         TEXT PRIMARY KEY,
        epic               TEXT NOT NULL,
        name               TEXT,
        pip_size           REAL,           -- from IG's onePipMeans
        one_pip_means      TEXT,
        decimals           INTEGER,        -- IG's decimalPlacesFactor
        value_of_one_pip   TEXT,
        contract_size      REAL,
        currency           TEXT,
        min_deal_size      REAL,
        min_stop_distance  REAL,
        updated_utc        TEXT NOT NULL
    );
    """,
    # ---- version 4: historical downloads and the final-check log (Phase 2, step 1) ----
    """
    -- One row per instrument-day that has been downloaded completely (every
    -- hour fetched without error). Used to skip finished days when a download
    -- is restarted. A day with no ticks at all (e.g. a holiday) is also DONE.
    CREATE TABLE history_days (
        source            TEXT NOT NULL,          -- 'dukascopy'
        instrument        TEXT NOT NULL,
        day               TEXT NOT NULL,          -- UTC date, YYYY-MM-DD
        tick_count        INTEGER NOT NULL,
        hours_requested   INTEGER NOT NULL,
        hours_empty       INTEGER NOT NULL,       -- requested hours that returned no ticks
        hours_skipped     INTEGER NOT NULL,       -- weekend hours not requested
        requests          INTEGER NOT NULL,       -- HTTP requests made (incl. retries we saw)
        seconds           REAL NOT NULL,          -- time taken, including throttling
        file_bytes        INTEGER NOT NULL,       -- size of the tick file (0 if none)
        feed_version      INTEGER,
        completed_utc     TEXT NOT NULL,
        PRIMARY KEY (source, instrument, day)
    );

    -- Every time data from the final-check period is read (Section 7.9: it
    -- should be looked at once, at the end).
    CREATE TABLE final_check_access (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        time_utc     TEXT NOT NULL,
        instruments  TEXT NOT NULL,
        reason       TEXT NOT NULL
    );
    """,
]

_initialised: set[str] = set()
_init_lock = threading.Lock()


def _ts(dt: datetime | None) -> str | None:
    """Datetime -> stored text (UTC). Naive datetimes are rejected."""
    if dt is None:
        return None
    return timeutils.ensure_utc(dt).isoformat(timespec="milliseconds")


def _dt(text: str | None) -> datetime | None:
    """Stored text -> aware UTC datetime."""
    return None if text is None else datetime.fromisoformat(text)


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10)      # wait up to 10 s if the other program is writing
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_db(path: Path | None = None) -> Path:
    """
    Create the database file (and data/ folder) if needed and bring the schema
    up to date. Safe to call many times; called automatically on first use.
    """
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_open(path)) as conn:
        # WAL mode lets the dashboard read while the engine writes.
        conn.execute("PRAGMA journal_mode = WAL")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for number, script in enumerate(MIGRATIONS, start=1):
            if number > version:
                # One transaction per version: if anything fails, closing the
                # connection rolls it back and the version number is unchanged.
                conn.executescript(f"BEGIN; {script}; PRAGMA user_version = {number}; COMMIT;")
    return path


@contextmanager
def _conn():
    """A connection to config.DB_PATH (schema checked once per process), closed afterwards."""
    path = Path(config.DB_PATH)
    key = str(path.resolve())
    if key not in _initialised:
        with _init_lock:
            if key not in _initialised:
                init_db(path)
                _initialised.add(key)
    conn = _open(path)
    try:
        yield conn
    finally:
        conn.close()


def schema_version() -> int:
    with _conn() as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


def _check_source(source: str) -> None:
    if source not in SOURCES:
        raise DBError(f"source must be one of {SOURCES}, not {source!r}")


def _check_reason(reason: str) -> str:
    reason = (reason or "").strip()
    if not reason:
        raise DBError("a reason is required (it is written to the activation log)")
    return reason


# --------------------------------------------------------------------------
# Kill switch and halt
# --------------------------------------------------------------------------
def _set_flag(name: str, on: bool, reason: str, source: str,
              expires_utc: datetime | None = None, now: datetime | None = None) -> None:
    _check_source(source)
    reason = _check_reason(reason)
    now_s = _ts(now or timeutils.now_utc())
    exp_s = _ts(expires_utc) if on else None
    with _conn() as conn, conn:
        conn.execute(
            "UPDATE flags SET is_on = ?, reason = ?, source = ?, changed_utc = ?, expires_utc = ? "
            "WHERE name = ?",
            (int(on), reason, source, now_s, exp_s, name),
        )
        conn.execute(
            "INSERT INTO flag_events (time_utc, flag, action, reason, source, expires_utc) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now_s, name, "ON" if on else "OFF", reason, source, exp_s),
        )


def set_kill_switch(reason: str, source: str, now: datetime | None = None) -> None:
    """Turn the kill switch ON. It stays on until clear_kill_switch() is called."""
    _set_flag(KILL_SWITCH, True, reason, source, now=now)


def clear_kill_switch(reason: str, source: str, now: datetime | None = None) -> None:
    """Turn the kill switch OFF (manual action only: CLI or dashboard job)."""
    _set_flag(KILL_SWITCH, False, reason, source, now=now)


def set_halt(reason: str, source: str = "engine", expires_utc: datetime | None = None,
             now: datetime | None = None) -> None:
    """
    Automatic halt by the engine. expires_utc=None means it stays until cleared
    manually (API errors / disconnections). For the daily loss limit pass the
    next trading-day start (timeutils.next_trading_day_start_utc).
    """
    _set_flag(HALT, True, reason, source, expires_utc=expires_utc, now=now)


def clear_halt(reason: str, source: str, now: datetime | None = None) -> None:
    _set_flag(HALT, False, reason, source, now=now)


def expire_halt(now: datetime | None = None) -> bool:
    """
    Clear a HALT whose expiry time has passed (e.g. daily loss limit at the
    17:00 New York reset), logging it. Returns True if it was cleared.
    """
    now = now or timeutils.now_utc()
    flag = get_flags()[HALT]
    if flag["is_on"] and flag["expires_utc"] is not None and flag["expires_utc"] <= now:
        clear_halt("halt period ended (trading day reset)", "engine", now=now)
        return True
    return False


def get_flags() -> dict:
    """{'KILL_SWITCH': {...}, 'HALT': {...}} with is_on, reason, source, times."""
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM flags").fetchall()
    return {
        r["name"]: {
            "is_on": bool(r["is_on"]),
            "reason": r["reason"],
            "source": r["source"],
            "changed_utc": _dt(r["changed_utc"]),
            "expires_utc": _dt(r["expires_utc"]),
        }
        for r in rows
    }


def trading_block_reasons(now: datetime | None = None) -> list[str]:
    """
    Why trading (new signals / orders) must not happen right now; an empty list
    means it may. The engine checks this every bar and before every order.

    A kill request still waiting in the jobs table counts too, so a dashboard
    kill takes effect even before the engine has processed the job.
    """
    now = now or timeutils.now_utc()
    flags = get_flags()
    reasons = []
    kill = flags[KILL_SWITCH]
    if kill["is_on"]:
        reasons.append(f"kill switch ON ({kill['source']}: {kill['reason']})")
    with _conn() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = ? AND status IN ('PENDING', 'RUNNING')",
            (JOB_KILL_SWITCH_ON,),
        ).fetchone()[0]
    if pending:
        reasons.append("kill switch requested from the dashboard (waiting for the engine)")
    halt = flags[HALT]
    if halt["is_on"] and (halt["expires_utc"] is None or halt["expires_utc"] > now):
        until = f", until {timeutils.fmt_dual(halt['expires_utc'], with_date=True)}" if halt["expires_utc"] else ""
        reasons.append(f"HALT ({halt['source']}: {halt['reason']}{until})")
    return reasons


def list_flag_events(limit: int = 50) -> list[dict]:
    """Newest first."""
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM flag_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [
        {**dict(r), "time_utc": _dt(r["time_utc"]), "expires_utc": _dt(r["expires_utc"])}
        for r in rows
    ]


# --------------------------------------------------------------------------
# Jobs (dashboard -> engine)
# --------------------------------------------------------------------------
def _job_dict(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    d["params"] = json.loads(d["params"] or "{}")
    d["result"] = json.loads(d["result"]) if d["result"] else None
    for k in ("requested_utc", "started_utc", "finished_utc"):
        d[k] = _dt(d[k])
    return d


def request_job(kind: str, params: dict | None = None, requested_by: str = "dashboard",
                now: datetime | None = None) -> int:
    """Queue a request for the engine. Returns the job id."""
    if kind not in JOB_KINDS:
        raise DBError(f"unknown job kind {kind!r}; allowed: {JOB_KINDS}")
    _check_source(requested_by)
    with _conn() as conn, conn:
        cur = conn.execute(
            "INSERT INTO jobs (kind, params, requested_by, requested_utc) VALUES (?, ?, ?, ?)",
            (kind, json.dumps(params or {}), requested_by, _ts(now or timeutils.now_utc())),
        )
        return cur.lastrowid


def claim_next_job(kinds: tuple[str, ...] | None = None, now: datetime | None = None) -> dict | None:
    """
    Engine only: take the oldest PENDING job (optionally of given kinds) and
    mark it RUNNING, atomically, so it can never be processed twice.
    """
    with _conn() as conn:
        conn.execute("BEGIN IMMEDIATE")      # lock for writing before choosing the job
        try:
            sql = "SELECT id FROM jobs WHERE status = 'PENDING'"
            args: list = []
            if kinds:
                sql += f" AND kind IN ({','.join('?' * len(kinds))})"
                args += list(kinds)
            row = conn.execute(sql + " ORDER BY id LIMIT 1", args).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE jobs SET status = 'RUNNING', started_utc = ? WHERE id = ?",
                (_ts(now or timeutils.now_utc()), row["id"]),
            )
            job = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return _job_dict(job)


def update_job_progress(job_id: int, progress: float, message: str | None = None) -> None:
    progress = min(max(float(progress), 0.0), 1.0)
    with _conn() as conn, conn:
        conn.execute("UPDATE jobs SET progress = ?, message = COALESCE(?, message) WHERE id = ?",
                     (progress, message, job_id))


def finish_job(job_id: int, ok: bool, message: str | None = None, result: dict | None = None,
               now: datetime | None = None) -> None:
    with _conn() as conn, conn:
        conn.execute(
            "UPDATE jobs SET status = ?, progress = CASE WHEN ? THEN 1 ELSE progress END, "
            "message = ?, result = ?, finished_utc = ? WHERE id = ?",
            ("DONE" if ok else "FAILED", int(ok), message,
             json.dumps(result) if result is not None else None,
             _ts(now or timeutils.now_utc()), job_id),
        )


def get_job(job_id: int) -> dict | None:
    with _conn() as conn:
        return _job_dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


def list_jobs(limit: int = 50, status: str | None = None) -> list[dict]:
    """Newest first."""
    if status is not None and status not in JOB_STATUSES:
        raise DBError(f"status must be one of {JOB_STATUSES}")
    sql, args = "SELECT * FROM jobs", []
    if status:
        sql += " WHERE status = ?"
        args.append(status)
    with _conn() as conn:
        rows = conn.execute(sql + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
    return [_job_dict(r) for r in rows]


# --------------------------------------------------------------------------
# Model variants (Sections 7.10 and 10.6)
# --------------------------------------------------------------------------
def _variant_dict(r):
    if r is None:
        return None
    d = dict(r)
    d["created_utc"] = _dt(d["created_utc"])
    d["role_changed_utc"] = _dt(d["role_changed_utc"])
    return d


def add_variant(name: str, feature_set: str, description: str | None = None,
                model_file: str | None = None, model_version: str | None = None,
                now: datetime | None = None) -> None:
    """Register a variant (starts as OFF)."""
    try:
        with _conn() as conn, conn:
            conn.execute(
                "INSERT INTO variants (name, feature_set, description, model_file, model_version, "
                "role, created_utc) VALUES (?, ?, ?, ?, ?, 'OFF', ?)",
                (name, feature_set, description, model_file, model_version,
                 _ts(now or timeutils.now_utc())),
            )
    except sqlite3.IntegrityError as exc:
        raise DBError(f"variant {name!r} already exists") from exc


def set_variant_role(name: str, role: str, source: str, reason: str | None = None,
                     now: datetime | None = None) -> None:
    """
    Change a variant's live role. Making one ACTIVE moves the previous ACTIVE
    one to SHADOW (Section 10.6). Every change is logged.
    """
    if role not in ROLES:
        raise DBError(f"role must be one of {ROLES}")
    _check_source(source)
    now_s = _ts(now or timeutils.now_utc())
    with _conn() as conn, conn:
        row = conn.execute("SELECT role FROM variants WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise DBError(f"unknown variant {name!r}")
        old = row["role"]
        if old == role:
            return
        if role == "ACTIVE":
            prev = conn.execute("SELECT name FROM variants WHERE role = 'ACTIVE'").fetchone()
            if prev is not None:
                conn.execute("UPDATE variants SET role = 'SHADOW', role_changed_utc = ? WHERE name = ?",
                             (now_s, prev["name"]))
                conn.execute(
                    "INSERT INTO variant_role_log (time_utc, variant, old_role, new_role, source, reason) "
                    "VALUES (?, ?, 'ACTIVE', 'SHADOW', ?, ?)",
                    (now_s, prev["name"], source, f"replaced by {name}"),
                )
        conn.execute("UPDATE variants SET role = ?, role_changed_utc = ? WHERE name = ?",
                     (role, now_s, name))
        conn.execute(
            "INSERT INTO variant_role_log (time_utc, variant, old_role, new_role, source, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now_s, name, old, role, source, reason),
        )


def get_variants() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM variants ORDER BY name").fetchall()
    return [_variant_dict(r) for r in rows]


def get_active_variant() -> dict | None:
    with _conn() as conn:
        return _variant_dict(conn.execute("SELECT * FROM variants WHERE role = 'ACTIVE'").fetchone())


def list_variant_role_log(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM variant_role_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "time_utc": _dt(r["time_utc"])} for r in rows]


# --------------------------------------------------------------------------
# Predictions (every signal, active and shadow, with its outcome)
# --------------------------------------------------------------------------
_PRED_TIME_FIELDS = ("created_utc", "bar_time_utc", "close_by_utc", "exit_time_utc")
_PRED_REQUIRED = ("variant", "is_shadow", "instrument", "epic", "bar_time_utc", "direction",
                  "entry", "stop_loss", "take_profit", "probability")
_PRED_OPTIONAL = ("model_version", "atr", "expected_value_r", "spread", "position_size",
                  "risk_pct", "close_by_utc", "pattern", "blocked_reason")


def _pred_dict(r):
    d = dict(r)
    d["is_shadow"] = bool(d["is_shadow"])
    for k in _PRED_TIME_FIELDS:
        d[k] = _dt(d[k])
    return d


def save_prediction(pred: dict, now: datetime | None = None) -> int:
    """Store a new signal (status OPEN). Returns its id."""
    missing = [k for k in _PRED_REQUIRED if pred.get(k) is None]
    if missing:
        raise DBError(f"prediction is missing: {', '.join(missing)}")
    unknown = set(pred) - set(_PRED_REQUIRED) - set(_PRED_OPTIONAL)
    if unknown:
        raise DBError(f"unknown prediction fields: {', '.join(sorted(unknown))}")
    if pred["instrument"] not in config.INSTRUMENTS:
        raise DBError(f"unknown instrument {pred['instrument']!r}")
    row = {k: pred.get(k) for k in _PRED_REQUIRED + _PRED_OPTIONAL}
    row["is_shadow"] = int(bool(row["is_shadow"]))
    for k in ("bar_time_utc", "close_by_utc"):
        row[k] = _ts(row[k])
    row["created_utc"] = _ts(now or timeutils.now_utc())
    cols = ", ".join(row)
    try:
        with _conn() as conn, conn:
            cur = conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({', '.join('?' * len(row))})",
                               list(row.values()))
            return cur.lastrowid
    except sqlite3.IntegrityError as exc:
        raise DBError(f"invalid prediction: {exc}") from exc


def close_prediction(pred_id: int, outcome: str, exit_price: float, exit_time_utc: datetime,
                     result_r: float) -> None:
    """Record how a signal ended: TP, SL or TIMEOUT."""
    if outcome not in ("TP", "SL", "TIMEOUT"):
        raise DBError("outcome must be TP, SL or TIMEOUT")
    with _conn() as conn, conn:
        cur = conn.execute(
            "UPDATE predictions SET status = 'CLOSED', outcome = ?, exit_price = ?, "
            "exit_time_utc = ?, result_r = ? WHERE id = ? AND status = 'OPEN'",
            (outcome, exit_price, _ts(exit_time_utc), result_r, pred_id),
        )
        if cur.rowcount != 1:
            raise DBError(f"prediction {pred_id} not found or already closed")


def get_open_predictions(include_shadow: bool = True) -> list[dict]:
    sql = "SELECT * FROM predictions WHERE status = 'OPEN'"
    if not include_shadow:
        sql += " AND is_shadow = 0"
    with _conn() as conn:
        rows = conn.execute(sql + " ORDER BY id").fetchall()
    return [_pred_dict(r) for r in rows]


def list_predictions(since_utc: datetime | None = None, variant: str | None = None,
                     include_shadow: bool = True, limit: int = 500) -> list[dict]:
    """Newest first."""
    sql, args = "SELECT * FROM predictions WHERE 1 = 1", []
    if since_utc is not None:
        sql += " AND bar_time_utc >= ?"
        args.append(_ts(since_utc))
    if variant is not None:
        sql += " AND variant = ?"
        args.append(variant)
    if not include_shadow:
        sql += " AND is_shadow = 0"
    with _conn() as conn:
        rows = conn.execute(sql + " ORDER BY bar_time_utc DESC, id DESC LIMIT ?", args + [limit]).fetchall()
    return [_pred_dict(r) for r in rows]


# --------------------------------------------------------------------------
# Price feed: gaps, latest prices, market status, engine heartbeat
# --------------------------------------------------------------------------
GAP_KINDS = ("OUTAGE", "QUIET", "ENGINE_OFF")


def _gap_dict(r):
    d = dict(r)
    d["start_utc"], d["end_utc"] = _dt(d["start_utc"]), _dt(d["end_utc"])
    return d


def open_gap(instrument: str, kind: str, start_utc: datetime, detail: str | None = None) -> int:
    """Start a gap record (end unknown yet). Returns its id."""
    if kind not in GAP_KINDS:
        raise DBError(f"gap kind must be one of {GAP_KINDS}")
    with _conn() as conn, conn:
        return conn.execute(
            "INSERT INTO gaps (instrument, kind, start_utc, detail) VALUES (?, ?, ?, ?)",
            (instrument, kind, _ts(start_utc), detail),
        ).lastrowid


def close_gap(gap_id: int, end_utc: datetime, detail: str | None = None) -> None:
    with _conn() as conn, conn:
        conn.execute(
            "UPDATE gaps SET end_utc = ?, detail = COALESCE(?, detail) WHERE id = ? AND end_utc IS NULL",
            (_ts(end_utc), detail, gap_id),
        )


def add_closed_gap(instrument: str, kind: str, start_utc: datetime, end_utc: datetime,
                   detail: str | None = None) -> int:
    gap_id = open_gap(instrument, kind, start_utc, detail)
    close_gap(gap_id, end_utc)
    return gap_id


def list_gaps(since_utc: datetime | None = None, open_only: bool = False, limit: int = 200) -> list[dict]:
    """Newest first."""
    sql, args = "SELECT * FROM gaps WHERE 1 = 1", []
    if since_utc is not None:
        sql += " AND (end_utc IS NULL OR end_utc >= ?)"
        args.append(_ts(since_utc))
    if open_only:
        sql += " AND end_utc IS NULL"
    with _conn() as conn:
        rows = conn.execute(sql + " ORDER BY start_utc DESC, id DESC LIMIT ?", args + [limit]).fetchall()
    return [_gap_dict(r) for r in rows]


def close_all_open_gaps(end_utc: datetime, detail: str) -> int:
    """Used when the engine stops: no gap stays open forever."""
    with _conn() as conn, conn:
        return conn.execute(
            "UPDATE gaps SET end_utc = ?, detail = COALESCE(detail || '; ', '') || ? WHERE end_utc IS NULL",
            (_ts(end_utc), detail),
        ).rowcount


def upsert_feed_status(rows: list[dict], now: datetime | None = None) -> None:
    """rows: instrument, epic, last_tick_utc, bid, offer, market_status, market_status_utc, ticks_today."""
    now_s = _ts(now or timeutils.now_utc())
    with _conn() as conn, conn:
        for r in rows:
            conn.execute(
                "INSERT INTO feed_status (instrument, epic, last_tick_utc, bid, offer, market_status, "
                "market_status_utc, ticks_today, updated_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (instrument) DO UPDATE SET epic = excluded.epic, "
                "last_tick_utc = excluded.last_tick_utc, bid = excluded.bid, offer = excluded.offer, "
                "market_status = excluded.market_status, market_status_utc = excluded.market_status_utc, "
                "ticks_today = excluded.ticks_today, updated_utc = excluded.updated_utc",
                (r["instrument"], r["epic"], _ts(r.get("last_tick_utc")), r.get("bid"), r.get("offer"),
                 r.get("market_status"), _ts(r.get("market_status_utc")), r.get("ticks_today"), now_s),
            )


def get_feed_status() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM feed_status ORDER BY instrument").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("last_tick_utc", "market_status_utc", "updated_utc"):
            d[k] = _dt(d[k])
        out.append(d)
    return out


def log_market_status(instrument: str, status: str, time_utc: datetime) -> None:
    with _conn() as conn, conn:
        conn.execute("INSERT INTO market_status_log (time_utc, instrument, status) VALUES (?, ?, ?)",
                     (_ts(time_utc), instrument, status))


def list_market_status_log(limit: int = 100) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM market_status_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "time_utc": _dt(r["time_utc"])} for r in rows]


def set_engine_status(values: dict, now: datetime | None = None) -> None:
    """Store key/value pairs (values as text); 'heartbeat' is set to now as well."""
    now_s = _ts(now or timeutils.now_utc())
    values = {**values, "heartbeat": now_s}
    with _conn() as conn, conn:
        for key, value in values.items():
            conn.execute(
                "INSERT INTO engine_status (key, value, updated_utc) VALUES (?, ?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_utc = excluded.updated_utc",
                (key, None if value is None else str(value), now_s),
            )


def get_engine_status() -> dict:
    """{key: value}; 'heartbeat' is returned as a datetime."""
    with _conn() as conn:
        rows = conn.execute("SELECT key, value FROM engine_status").fetchall()
    d = {r["key"]: r["value"] for r in rows}
    if d.get("heartbeat"):
        d["heartbeat"] = _dt(d["heartbeat"])
    return d


def engine_is_running(now: datetime | None = None) -> bool:
    """True if the engine's heartbeat is recent and it has not reported stopping."""
    status = get_engine_status()
    hb = status.get("heartbeat")
    if hb is None or status.get("state") == "stopped":
        return False
    age = ((now or timeutils.now_utc()) - hb).total_seconds()
    return age <= config.ENGINE_HEARTBEAT_STALE_SECONDS


# --------------------------------------------------------------------------
# Contract details from IG (pip size, decimals, minimum deal ...)
# --------------------------------------------------------------------------
def upsert_market_info(instrument: str, market: dict, now: datetime | None = None) -> None:
    """market: the dict returned by broker.Broker.get_market()."""
    def rule_value(r):
        return r.get("value") if r else None

    currency = next((c["code"] for c in market.get("currencies") or [] if c.get("is_default")), None)
    decimals = market.get("decimal_places_factor")
    with _conn() as conn, conn:
        conn.execute(
            "INSERT INTO market_info (instrument, epic, name, pip_size, one_pip_means, decimals, "
            "value_of_one_pip, contract_size, currency, min_deal_size, min_stop_distance, updated_utc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (instrument) DO UPDATE SET "
            "epic = excluded.epic, name = excluded.name, pip_size = excluded.pip_size, "
            "one_pip_means = excluded.one_pip_means, decimals = excluded.decimals, "
            "value_of_one_pip = excluded.value_of_one_pip, contract_size = excluded.contract_size, "
            "currency = excluded.currency, min_deal_size = excluded.min_deal_size, "
            "min_stop_distance = excluded.min_stop_distance, updated_utc = excluded.updated_utc",
            (instrument, market.get("epic"), market.get("name"), market.get("pip_size"),
             market.get("one_pip_means"), int(decimals) if isinstance(decimals, (int, float)) else None,
             market.get("value_of_one_pip"),
             market.get("contract_size") if isinstance(market.get("contract_size"), (int, float)) else None,
             currency, rule_value(market.get("min_deal_size")),
             rule_value(market.get("min_stop_or_limit_distance")),
             _ts(now or timeutils.now_utc())),
        )


def get_market_info() -> dict[str, dict]:
    """{instrument: {...}}"""
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM market_info").fetchall()
    return {r["instrument"]: {**dict(r), "updated_utc": _dt(r["updated_utc"])} for r in rows}


# --------------------------------------------------------------------------
# Historical downloads (fetch_data.py) and the final-check log
# --------------------------------------------------------------------------
def mark_history_day_done(source: str, instrument: str, day, stats: dict,
                          now: datetime | None = None) -> None:
    """
    Record that one instrument-day has been downloaded completely.
    stats: tick_count, hours_requested, hours_empty, hours_skipped, requests,
    seconds, file_bytes, feed_version. Re-downloading a day replaces its row.
    """
    with _conn() as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO history_days (source, instrument, day, tick_count, "
            "hours_requested, hours_empty, hours_skipped, requests, seconds, file_bytes, "
            "feed_version, completed_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (source, instrument, day.isoformat(), int(stats["tick_count"]),
             int(stats["hours_requested"]), int(stats["hours_empty"]), int(stats["hours_skipped"]),
             int(stats["requests"]), float(stats["seconds"]), int(stats["file_bytes"]),
             stats.get("feed_version"), _ts(now or timeutils.now_utc())),
        )


def get_history_days(source: str, instrument: str | None = None,
                     start_day=None, end_day=None) -> list[dict]:
    """Downloaded days, oldest first. start_day included, end_day not included."""
    sql = "SELECT * FROM history_days WHERE source = ?"
    args: list = [source]
    if instrument is not None:
        sql += " AND instrument = ?"
        args.append(instrument)
    if start_day is not None:
        sql += " AND day >= ?"
        args.append(start_day.isoformat())
    if end_day is not None:
        sql += " AND day < ?"
        args.append(end_day.isoformat())
    sql += " ORDER BY instrument, day"
    with _conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["day"] = datetime.strptime(d["day"], "%Y-%m-%d").date()
        d["completed_utc"] = _dt(d["completed_utc"])
        out.append(d)
    return out


def log_final_check_access(instruments: list[str], reason: str,
                           now: datetime | None = None) -> None:
    reason = _check_reason(reason)
    with _conn() as conn, conn:
        conn.execute("INSERT INTO final_check_access (time_utc, instruments, reason) VALUES (?, ?, ?)",
                     (_ts(now or timeutils.now_utc()), ",".join(instruments), reason))


def list_final_check_access() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM final_check_access ORDER BY id").fetchall()
    return [{**dict(r), "time_utc": _dt(r["time_utc"])} for r in rows]
