"""
fetch_data.py - records the live IG feed (Phase 1) and downloads history
from Dukascopy (Phase 2).

Recording (run by the engine, signals.py):
  * every price update -> data/ticks/<INSTRUMENT>/<YYYY-MM-DD>.parquet
      columns: time_utc (IG's time), bid, offer, received_utc (laptop), is_snapshot
  * 1-minute bars      -> data/bars_1m/<INSTRUMENT>/<YYYY-MM-DD>.parquet
      columns: time_utc (bar START), bid_open/high/low/close,
               offer_open/high/low/close, tick_count, spread_mean, spread_max,
               partial (True if prices were not being received for part of
               the minute: an OUTAGE / ENGINE_OFF gap, or recording started
               mid-minute - such a bar's high/low may be incomplete)
  * files are split at UTC midnight and rewritten every RECORDER_FLUSH_SECONDS
    (write to a temporary file, then swap it in, so a crash never leaves a
    half-written file; at most one flush interval is lost)
  * gaps, latest prices and market status go to SQLite through db.py

No-lookahead rule for bars: a bar labelled 10:00 covers [10:00:00, 10:01:00)
and is only complete at 10:01. Only complete bars are written (plus
BAR_CLOSE_GRACE_SECONDS for late prices). A minute without any price update
gets no bar - nothing is invented. Bars can always be rebuilt from the ticks.

Duplicates: after a reconnection IG re-sends the last price as a "snapshot".
A snapshot identical (time, bid, offer) to a price already stored is dropped.

Gap kinds logged (table gaps):
  OUTAGE      the stream was not connected (all instruments)
  QUIET       no price for QUIET_GAP_MINUTES[instrument] while IG reported the
              market open (possibly just a lull)
  ENGINE_OFF  the engine was not running (from its last heartbeat to restart)

History from Dukascopy (command line, see the bottom of this file):
  * ticks  -> data/dukascopy/ticks/<INSTRUMENT>/<YYYY-MM-DD>.parquet (one UTC day)
  * 1-minute bars -> data/dukascopy/bars_1m/<INSTRUMENT>/<YYYY-MM>.parquet
    (one UTC month; same columns as the IG bars above, partial always False)
  * downloaded one hour at a time, at most one request per second; a day is
    recorded as done in the database (history_days) only when every hour of
    it was fetched without error, so an interrupted download resumes where
    it stopped and never re-downloads a finished day
  * fetch_dukascopy_hour() is the ONLY function that uses the dukascopy-python
    library; if it breaks, only that function needs replacing
  * load_history_1m() is how the model pipeline reads history. It refuses the
    final-check period (config.FINAL_CHECK_*) unless the purpose is the final
    evaluation, and the indices until config.HISTORY_APPROVED_FOR_MODEL allows them.

    python fetch_data.py pilot       # one week, all five instruments, then a report
    python fetch_data.py download    # everything from 2019 (resumable; Ctrl+C is safe)
    python fetch_data.py status      # what has been downloaded
    python fetch_data.py move-ticks  # move ticks to the folder set in .env (DUKASCOPY_TICKS_DIR)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config
import db
import timeutils

log = logging.getLogger("fetch_data")

TICK_COLUMNS = ["time_utc", "bid", "offer", "received_utc", "is_snapshot"]
BAR_COLUMNS = ["time_utc",
               "bid_open", "bid_high", "bid_low", "bid_close",
               "offer_open", "offer_high", "offer_low", "offer_close",
               "tick_count", "spread_mean", "spread_max", "partial"]

# Gap kinds during which prices were definitely NOT being received. A bar
# overlapping one of these is marked partial. (QUIET gaps don't count: the
# stream was connected, the market may simply have been calm.)
NOT_RECEIVING_GAP_KINDS = ("OUTAGE", "ENGINE_OFF")


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
def tick_file(instrument: str, day: date) -> Path:
    return config.TICKS_DIR / instrument / f"{day.isoformat()}.parquet"


def bar_file(instrument: str, day: date) -> Path:
    return config.BARS_1M_DIR / instrument / f"{day.isoformat()}.parquet"


def write_parquet_atomic(df: pd.DataFrame, path: Path, attempts: int = 5,
                         compression: str = "snappy") -> None:
    """Write to <name>.tmp then swap it in, so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression=compression)
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows: another program (e.g. the dashboard) may have the file
            # open for a moment. Wait and retry.
            if attempt == attempts - 1:
                raise
            time.sleep(0.2)


def read_ticks(instrument: str, day: date) -> pd.DataFrame:
    path = tick_file(instrument, day)
    if not path.exists():
        return pd.DataFrame(columns=TICK_COLUMNS)
    return pd.read_parquet(path)


def read_bars(instrument: str, day: date) -> pd.DataFrame:
    path = bar_file(instrument, day)
    if not path.exists():
        return pd.DataFrame(columns=BAR_COLUMNS)
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Bars from ticks
# --------------------------------------------------------------------------
def bar_end_utc(bar_start: datetime, minutes: int = 1) -> datetime:
    """A bar labelled by its start is complete at start + its length."""
    return bar_start + timedelta(minutes=minutes)


def ticks_to_bars(ticks: pd.DataFrame, now: datetime | None = None,
                  not_receiving: list[tuple[datetime, datetime | None]] | None = None,
                  session_start: datetime | None = None) -> pd.DataFrame:
    """
    1-minute bid/offer bars from ticks, labelled by bar START (UTC).
    If `now` is given, only bars that are complete (end + grace <= now) are
    returned - never a bar that is still forming.
    `not_receiving`: (start, end) periods when no prices were received (end
    None = still ongoing); `session_start`: when recording started. Bars
    overlapping either are marked partial=True.
    """
    if ticks.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = ticks.dropna(subset=["bid", "offer"]).sort_values("time_utc", kind="stable")
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = df.assign(spread=df["offer"] - df["bid"],
                   minute=pd.to_datetime(df["time_utc"], utc=True).dt.floor("1min"))
    g = df.groupby("minute", sort=True)
    bars = pd.DataFrame({
        "bid_open": g["bid"].first(), "bid_high": g["bid"].max(),
        "bid_low": g["bid"].min(), "bid_close": g["bid"].last(),
        "offer_open": g["offer"].first(), "offer_high": g["offer"].max(),
        "offer_low": g["offer"].min(), "offer_close": g["offer"].last(),
        "tick_count": g.size(),
        "spread_mean": g["spread"].mean(), "spread_max": g["spread"].max(),
    })
    bars.index.name = "time_utc"
    bars = bars.reset_index()
    bar_start = bars["time_utc"]
    bar_end = bar_start + pd.Timedelta(minutes=1)
    partial = pd.Series(False, index=bars.index)
    far_future = pd.Timestamp.max.tz_localize("UTC")
    for start, end in not_receiving or []:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end is not None else far_future
        partial |= (bar_start < e) & (bar_end > s)
    if session_start is not None:
        s = pd.Timestamp(session_start)
        partial |= (bar_start < s) & (bar_end > s)
    bars["partial"] = partial
    if now is not None:
        cutoff = pd.Timestamp(timeutils.ensure_utc(now)) - pd.Timedelta(
            minutes=1, seconds=config.BAR_CLOSE_GRACE_SECONDS)
        bars = bars[bars["time_utc"] <= cutoff]
    return bars[BAR_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------
# The recorder
# --------------------------------------------------------------------------
@dataclass
class _DayBuffer:
    rows: list = field(default_factory=list)       # tuples in TICK_COLUMNS order
    keys: set = field(default_factory=set)         # (time ms, bid, offer) for duplicate checks
    dirty: bool = False                             # ticks not yet written
    bars_written: object = None                     # fingerprint of the bars last written


def _tick_key(time_utc: datetime, bid, offer) -> tuple:
    return (int(time_utc.timestamp() * 1000), bid, offer)


class Recorder:
    """
    Collects price updates in memory and, on maintenance(), writes the files
    and the gap / feed-status records. Thread-safe: on_tick, on_market_status
    and set_streaming may be called from the Lightstreamer thread; they only
    touch memory. maintenance() (engine scheduler) does all disk/DB writes.
    """

    def __init__(self, instruments: dict[str, str] | None = None, clock=timeutils.now_utc):
        # instrument -> epic (default: all five from config.py)
        self._inst_epic = instruments or {k: v["epic"] for k, v in config.INSTRUMENTS.items()}
        self._epic_inst = {epic: inst for inst, epic in self._inst_epic.items()}
        self._clock = clock
        self._lock = threading.Lock()
        self._days: dict[tuple[str, date], _DayBuffer] = {}
        self._last_tick: dict[str, tuple] = {}                 # inst -> (time, bid, offer)
        self._market: dict[str, tuple[str, datetime]] = {}     # inst -> (status, since)
        self._status_events: list[tuple[str, str, datetime]] = []
        self._stream_events: list[tuple[bool, datetime]] = []
        self._streaming = False
        self._streaming_since: datetime | None = None
        self._outage_gaps: dict[str, int] = {}                 # inst -> open gap id
        self._quiet_gaps: dict[str, tuple[int, datetime]] = {} # inst -> (gap id, start)
        self._first_tick_after_quiet: dict[str, datetime] = {}
        self.duplicates_dropped = 0
        self.write_errors = 0
        self.started_utc: datetime | None = None

    # ---------------- start / stop ----------------
    def start(self, last_heartbeat: datetime | None = None) -> None:
        """
        Load today's ticks already on disk (so a restart continues the same
        files) and log an ENGINE_OFF gap since the previous run's heartbeat.
        """
        now = self._clock()
        self.started_utc = now
        today = now.date()
        with self._lock:
            for inst in self._inst_epic:
                self._buffer(inst, today)
        if last_heartbeat is not None and (now - last_heartbeat).total_seconds() > 60:
            for inst in self._inst_epic:
                db.add_closed_gap(inst, "ENGINE_OFF", last_heartbeat, now,
                                  "engine was not running (last heartbeat to restart)")
            log.info("Logged ENGINE_OFF gap from %s", timeutils.fmt_dual(last_heartbeat, with_date=True))

    def close(self) -> None:
        """Final write and close any open gaps (the engine is stopping)."""
        now = self._clock()
        self.maintenance(now)
        db.close_all_open_gaps(now, "engine stopped")

    # ---------------- inputs (may run on the Lightstreamer thread) ----------------
    def _buffer(self, inst: str, day: date) -> _DayBuffer:
        """The in-memory buffer for one instrument-day, loaded from disk the first time."""
        buf = self._days.get((inst, day))
        if buf is None:
            buf = _DayBuffer()
            existing = read_ticks(inst, day)
            for row in existing[TICK_COLUMNS].itertuples(index=False, name=None):
                t = pd.Timestamp(row[0]).to_pydatetime()
                rec = (t, row[1], row[2], pd.Timestamp(row[3]).to_pydatetime(), bool(row[4]))
                buf.rows.append(rec)
                buf.keys.add(_tick_key(t, row[1], row[2]))
            if buf.rows:
                last = max(buf.rows, key=lambda r: r[0])
                prev = self._last_tick.get(inst)
                if prev is None or last[0] > prev[0]:
                    self._last_tick[inst] = (last[0], last[1], last[2])
            self._days[(inst, day)] = buf
        return buf

    def on_tick(self, tick) -> None:
        """Called for every price update (broker.Tick)."""
        inst = self._epic_inst.get(tick.epic)
        if inst is None:
            return
        key = _tick_key(tick.time_utc, tick.bid, tick.offer)
        with self._lock:
            buf = self._buffer(inst, tick.time_utc.date())
            if tick.is_snapshot and key in buf.keys:
                self.duplicates_dropped += 1          # IG re-sent a price we already have
                return
            buf.rows.append((tick.time_utc, tick.bid, tick.offer, tick.received_utc, tick.is_snapshot))
            buf.keys.add(key)
            buf.dirty = True
            prev = self._last_tick.get(inst)
            if prev is None or tick.time_utc >= prev[0]:
                self._last_tick[inst] = (tick.time_utc, tick.bid, tick.offer)
            if inst in self._quiet_gaps and inst not in self._first_tick_after_quiet \
                    and tick.time_utc > self._quiet_gaps[inst][1]:
                self._first_tick_after_quiet[inst] = tick.time_utc

    def on_market_status(self, epic: str, status: str, time_utc: datetime) -> None:
        """IG market status change (DLG_FLAG) for an epic."""
        inst = self._epic_inst.get(epic)
        if inst is None:
            return
        with self._lock:
            self._market[inst] = (status, time_utc)
            self._status_events.append((inst, status, time_utc))

    def set_streaming(self, streaming: bool, when: datetime | None = None) -> None:
        """Called by the engine whenever the stream starts or stops streaming."""
        with self._lock:
            if streaming != self._streaming:
                when = when or self._clock()
                self._streaming = streaming
                self._streaming_since = when if streaming else None
                self._stream_events.append((streaming, when))

    # ---------------- maintenance (engine scheduler, every minute) ----------------
    def maintenance(self, now: datetime | None = None) -> dict:
        """Write files, gaps, market-status log and feed status. Returns a summary."""
        now = now or self._clock()
        with self._lock:
            stream_events, self._stream_events = self._stream_events, []
            status_events, self._status_events = self._status_events, []
        self._handle_stream_events(stream_events)
        self._handle_status_events(status_events)
        self._check_quiet(now)
        written = self._write_files(now)
        self.write_feed_status(now)
        return {"files_written": written, "duplicates_dropped": self.duplicates_dropped,
                "write_errors": self.write_errors,
                "open_gaps": len(self._outage_gaps) + len(self._quiet_gaps)}

    def _handle_stream_events(self, events) -> None:
        for streaming, when in events:
            if not streaming and not self._outage_gaps:
                for inst in self._inst_epic:
                    self._outage_gaps[inst] = db.open_gap(inst, "OUTAGE", when, "price stream not connected")
            elif streaming and self._outage_gaps:
                for gap_id in self._outage_gaps.values():
                    db.close_gap(gap_id, when)
                self._outage_gaps.clear()

    def _handle_status_events(self, events) -> None:
        for inst, status, when in events:
            db.log_market_status(inst, status, when)
            if status not in config.MARKET_OPEN_DLG_FLAGS and inst in self._quiet_gaps:
                gap_id, _ = self._quiet_gaps.pop(inst)
                db.close_gap(gap_id, when, f"market status became {status}")

    def _check_quiet(self, now: datetime) -> None:
        with self._lock:
            market = dict(self._market)
            last_tick = dict(self._last_tick)
            first_after = dict(self._first_tick_after_quiet)
            streaming = self._streaming
            streaming_since = self._streaming_since
        for inst in self._inst_epic:
            # Close a quiet gap once a price has arrived.
            if inst in self._quiet_gaps and inst in first_after:
                gap_id, _ = self._quiet_gaps.pop(inst)
                db.close_gap(gap_id, first_after[inst])
                with self._lock:
                    self._first_tick_after_quiet.pop(inst, None)
                continue
            if inst in self._quiet_gaps or not streaming:
                continue                      # already open, or covered by an OUTAGE gap
            status, status_since = market.get(inst, (None, None))
            if status not in config.MARKET_OPEN_DLG_FLAGS:
                continue                      # closed or unknown: silence is expected / can't judge
            # Silence is measured from the latest of: market opening, engine start,
            # stream (re)connection and the last price received.
            since_candidates = [t for t in (status_since, self.started_utc, streaming_since)
                                if t is not None]
            if inst in last_tick:
                since_candidates.append(last_tick[inst][0])
            since = max(since_candidates) if since_candidates else now
            limit = timedelta(minutes=config.QUIET_GAP_MINUTES.get(inst, 5))
            if now - since > limit:
                gap_id = db.open_gap(inst, "QUIET", since,
                                     f"no price for over {limit.seconds // 60} min while market {status}")
                self._quiet_gaps[inst] = (gap_id, since)
                log.warning("QUIET gap: %s has had no price since %s", inst,
                            timeutils.fmt_dual(since, with_date=True))

    def _write_files(self, now: datetime) -> int:
        """Write ticks (if new) and complete bars for each in-memory day; drop old days."""
        written = 0
        with self._lock:
            items = [(k, b, list(b.rows), b.dirty) for k, b in self._days.items()]
            for _, b, _, _ in items:
                b.dirty = False
        day_start = datetime(now.year, now.month, now.day, tzinfo=now.tzinfo) - timedelta(days=1)
        gaps = [g for g in db.list_gaps(since_utc=day_start, limit=1000)
                if g["kind"] in NOT_RECEIVING_GAP_KINDS]
        for (inst, day), buf, rows, dirty in items:
            try:
                ticks = pd.DataFrame(rows, columns=TICK_COLUMNS).sort_values("time_utc", kind="stable")
                if dirty:
                    write_parquet_atomic(ticks, tick_file(inst, day))
                    written += 1
                periods = [(g["start_utc"], g["end_utc"]) for g in gaps if g["instrument"] == inst]
                bars = ticks_to_bars(ticks, now=now, not_receiving=periods,
                                     session_start=self.started_utc)
                if len(bars) == 0 and not bar_file(inst, day).exists():
                    continue                        # no complete bar yet: don't create an empty file
                fingerprint = (len(bars), int(bars["partial"].sum()), bars["tick_count"].sum())
                if fingerprint != buf.bars_written:
                    write_parquet_atomic(bars, bar_file(inst, day))
                    buf.bars_written = fingerprint
                    written += 1
            except Exception as exc:
                self.write_errors += 1
                with self._lock:
                    buf.dirty = True                # try again next time
                log.error("Writing %s %s failed (will retry): %s", inst, day, exc)
        # Keep only today and yesterday in memory (late prices around midnight).
        keep_from = now.date() - timedelta(days=1)
        with self._lock:
            for key in [k for k, b in self._days.items() if k[1] < keep_from and not b.dirty]:
                del self._days[key]
        return written

    def write_feed_status(self, now: datetime | None = None) -> None:
        """Latest price / market status per instrument -> db (for the dashboard)."""
        now = now or self._clock()
        today = now.date()
        with self._lock:
            rows = []
            for inst, epic in self._inst_epic.items():
                t = self._last_tick.get(inst)
                status = self._market.get(inst)
                buf = self._days.get((inst, today))
                rows.append({
                    "instrument": inst, "epic": epic,
                    "last_tick_utc": t[0] if t else None,
                    "bid": t[1] if t else None, "offer": t[2] if t else None,
                    "market_status": status[0] if status else None,
                    "market_status_utc": status[1] if status else None,
                    "ticks_today": len(buf.rows) if buf else 0,
                })
        db.upsert_feed_status(rows, now=now)


# ==========================================================================
# Historical data from Dukascopy (Phase 2, step 1)
# ==========================================================================
DUKASCOPY = "dukascopy"                    # source name in the history_days table
DUKASCOPY_LIBRARY_VERSION = "6.0.0"        # the version this code was written and checked against
HISTORY_TICK_COLUMNS = ["time_utc", "bid", "offer", "bid_volume", "offer_volume"]

# Why history is being read (load_history_1m):
PURPOSE_DEVELOPMENT = "development"            # features, labels, training, walk-forward, backtests
PURPOSE_SOURCE_COMPARISON = "source_comparison"  # Dukascopy vs IG checks (any instrument)
PURPOSE_FINAL_EVALUATION = "final_evaluation"  # the one final check (Section 7.9)
PURPOSES = (PURPOSE_DEVELOPMENT, PURPOSE_SOURCE_COMPARISON, PURPOSE_FINAL_EVALUATION)

# The library logs every hour at INFO level; keep only its warnings and errors.
_LIB_LOG = logging.getLogger("dukascopy_python_lib")
_LIB_LOG.setLevel(logging.WARNING)


class DukascopyError(Exception):
    """A download failed. blocked=True means Dukascopy refused us: stop, don't retry."""

    def __init__(self, message: str, blocked: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.blocked = blocked
        self.status_code = status_code


class HistoryAccessError(Exception):
    """A read of history that the final-check / approval rules don't allow."""


class TicksFolderError(Exception):
    """The tick folder can't be used (drive unplugged, other letter, bad setting)."""


# --------------------------------------------------------------------------
# Files and calendar
# --------------------------------------------------------------------------
def history_tick_file(instrument: str, day: date) -> Path:
    return config.DUKASCOPY_TICKS_DIR / instrument / f"{day.isoformat()}.parquet"


def history_bar_file(instrument: str, year: int, month: int) -> Path:
    return config.DUKASCOPY_BARS_1M_DIR / instrument / f"{year:04d}-{month:02d}.parquet"


def ticks_folder_problem() -> str | None:
    """Why the tick folder can't be used right now, or None if it can."""
    path = config.DUKASCOPY_TICKS_DIR
    if any(ord(c) < 32 for c in str(path)):
        # e.g. "D:\TrendsAnalyst\dukascopy\ticks" in double quotes: .env turns \t into a tab.
        return (f"DUKASCOPY_TICKS_DIR in .env contains a hidden control character ({str(path)!r}). "
                "Write the path without quotes, e.g. DUKASCOPY_TICKS_DIR=D:\\TrendsAnalyst\\dukascopy\\ticks")
    if config.DUKASCOPY_TICKS_DIR_FROM_ENV and not path.is_dir():
        return (f"The tick folder {path} (DUKASCOPY_TICKS_DIR in .env) can't be found. Is the drive "
                "plugged in, with the same letter? If this is its first use, create the folder: "
                f'New-Item -ItemType Directory "{path}"')
    return None


def require_ticks_folder() -> None:
    """Stop (TicksFolderError) rather than read or write ticks in the wrong place."""
    problem = ticks_folder_problem()
    if problem:
        raise TicksFolderError(problem)


def is_skipped_hour(hour_start: datetime) -> bool:
    """True for an hour inside the weekly closed window (config.DUKASCOPY_SKIP_*)."""
    t = timeutils.ensure_utc(hour_start)
    return config.DUKASCOPY_SKIP_FROM <= (t.weekday(), t.hour) < config.DUKASCOPY_SKIP_UNTIL


def default_history_end(now: datetime | None = None) -> datetime:
    """Midnight UTC, DUKASCOPY_LAG_DAYS before today: the newest day may still be incomplete."""
    now = timeutils.ensure_utc(now or timeutils.now_utc())
    today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return today - timedelta(days=config.DUKASCOPY_LAG_DAYS)


def _days(start: datetime, end: datetime) -> list[date]:
    """UTC days d with start <= d < end (start/end are midnights or are rounded up to one)."""
    start, end = timeutils.ensure_utc(start), timeutils.ensure_utc(end)
    d = start.date() if start.time() == datetime.min.time() else start.date() + timedelta(days=1)
    out = []
    while datetime(d.year, d.month, d.day, tzinfo=timezone.utc) < end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _hour_starts(day: date) -> list[datetime]:
    return [datetime(day.year, day.month, day.day, h, tzinfo=timezone.utc) for h in range(24)]


# --------------------------------------------------------------------------
# Polite HTTP: throttle, time-out and retries for every request the library makes
# --------------------------------------------------------------------------
class _Throttle:
    """Makes sure requests are at least config.DUKASCOPY_MIN_SECONDS_BETWEEN_REQUESTS apart."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            delay = self._next_allowed - self._clock()
            if delay > 0:
                self._sleep(delay)
            self._next_allowed = self._clock() + config.DUKASCOPY_MIN_SECONDS_BETWEEN_REQUESTS


_THROTTLE = _Throttle()


@dataclass
class _RequestCounter:
    count: int = 0


class _PoliteSession(requests.Session):
    """A requests.Session that waits for the throttle and always has a time-out."""

    def __init__(self, counter: _RequestCounter | None = None):
        super().__init__()
        self._counter = counter
        retry = Retry(
            total=config.DUKASCOPY_HTTP_RETRIES,
            backoff_factor=config.DUKASCOPY_RETRY_BACKOFF_SECONDS,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.mount("https://", adapter)
        self.mount("http://", adapter)

    def request(self, method, url, *args, **kwargs):
        _THROTTLE.wait()
        kwargs.setdefault("timeout", config.DUKASCOPY_REQUEST_TIMEOUT_SECONDS)
        if self._counter is not None:
            self._counter.count += 1
        return super().request(method, url, *args, **kwargs)


# --------------------------------------------------------------------------
# The one place that uses the dukascopy-python library
# --------------------------------------------------------------------------
_library_checked = False


def _library():
    """Import dukascopy_python (only when downloading) and check its version."""
    global _library_checked
    import dukascopy_python
    if not _library_checked:
        from importlib.metadata import version
        installed = version("dukascopy-python")
        if installed != DUKASCOPY_LIBRARY_VERSION:
            raise RuntimeError(
                f"dukascopy-python {installed} is installed, but this code was checked against "
                f"{DUKASCOPY_LIBRARY_VERSION}. Run: python -m pip install -r requirements.txt")
        _library_checked = True
    return dukascopy_python


def instrument_meta(instrument: str) -> dict:
    """Dukascopy's own details for an instrument: priceScale (decimals), pipValue ..."""
    from dukascopy_python.instruments import INSTRUMENT_CODE_TO_DATA
    code = config.DUKASCOPY_SYMBOLS[instrument]
    meta = INSTRUMENT_CODE_TO_DATA.get(code)
    if meta is None:
        raise KeyError(f"{code} is not in dukascopy-python's instrument table")
    return meta


def _classify_error(exc: Exception) -> DukascopyError:
    """Turn whatever the library or requests raised into a plain-language DukascopyError."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(exc, requests.exceptions.RetryError):
        return DukascopyError("Dukascopy kept answering with errors (rate limit or server "
                              f"problem) after {config.DUKASCOPY_HTTP_RETRIES} retries: {exc}")
    if status == 403:
        return DukascopyError("HTTP 403: Dukascopy refused the request (possibly bot protection)",
                              blocked=True, status_code=403)
    if status is not None:
        return DukascopyError(f"HTTP {status}: {exc}", status_code=status)
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return DukascopyError(f"network problem: {type(exc).__name__}: {exc}")
    if isinstance(exc, json.JSONDecodeError):
        return DukascopyError("the reply was not price data (possibly a bot-protection page)",
                              blocked=True)
    if isinstance(exc, (KeyError, TypeError, IndexError)):
        return DukascopyError(f"the reply was in an unexpected format: {type(exc).__name__}: {exc}")
    return DukascopyError(f"{type(exc).__name__}: {exc}")


def fetch_dukascopy_hour(instrument: str, hour_start: datetime, feed_version: int | None = None,
                         counter: _RequestCounter | None = None) -> pd.DataFrame:
    """
    All ticks (bid and offer) for one UTC hour [hour_start, hour_start + 1 h).
    Returns HISTORY_TICK_COLUMNS, oldest first; empty if Dukascopy has no ticks.
    Raises DukascopyError if the request fails.
    """
    hour_start = timeutils.ensure_utc(hour_start)
    end = hour_start + timedelta(hours=1)
    decimals = int(instrument_meta(instrument)["priceScale"])
    duka = _library()
    try:
        ticks = duka.fetch_ticks(
            config.DUKASCOPY_SYMBOLS[instrument], hour_start, end,
            version=feed_version or config.DUKASCOPY_FEED_VERSION,
            create_session=lambda: _PoliteSession(counter),
            logger=_LIB_LOG,
        )
        ticks = list(ticks)
    except Exception as exc:                       # anything from the library or the network
        raise _classify_error(exc) from exc
    return _ticks_to_frame(ticks, decimals, hour_start, end)


def _ticks_to_frame(ticks: list, decimals: int, start: datetime, end: datetime) -> pd.DataFrame:
    """Library Tick objects -> DataFrame. Prices rounded to Dukascopy's decimals."""
    rows = [(t.timestamp, t.bid, t.ask, t.bid_volume, t.ask_volume) for t in ticks]
    df = pd.DataFrame(rows, columns=HISTORY_TICK_COLUMNS)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    for col in ("bid", "offer", "bid_volume", "offer_volume"):
        df[col] = df[col].astype("float64")
    if df.empty:
        return df
    # The library rebuilds prices by adding small steps, which leaves float
    # noise such as 1.0849999999; Dukascopy's own precision is `decimals`.
    df["bid"] = df["bid"].round(decimals)
    df["offer"] = df["offer"].round(decimals)
    outside = (df["time_utc"] < pd.Timestamp(start)) | (df["time_utc"] >= pd.Timestamp(end))
    if outside.any():
        raise DukascopyError(f"{int(outside.sum())} ticks fall outside the requested hour "
                             f"{start.isoformat()} - refusing to store them")
    return df.sort_values("time_utc", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# Downloading days, building monthly bars
# --------------------------------------------------------------------------
def download_day(instrument: str, day: date, skip_closed: bool = True,
                 feed_version: int | None = None) -> dict:
    """
    Download every hour of one UTC day, write the tick file and record the
    day as done. If any hour fails, nothing is written or recorded (the whole
    day is tried again next time) and DukascopyError is raised.
    """
    require_ticks_folder()
    started = time.monotonic()
    counter = _RequestCounter()
    frames, requested, empty, skipped = [], 0, 0, 0
    for hour_start in _hour_starts(day):
        if skip_closed and is_skipped_hour(hour_start):
            skipped += 1
            continue
        try:
            df = fetch_dukascopy_hour(instrument, hour_start, feed_version, counter)
        except DukascopyError as exc:
            raise DukascopyError(f"{instrument} {hour_start:%Y-%m-%d %H}:00 UTC: {exc}",
                                 blocked=exc.blocked, status_code=exc.status_code) from exc
        requested += 1
        if df.empty:
            empty += 1
        else:
            frames.append(df)
    path = history_tick_file(instrument, day)
    file_bytes = 0
    tick_count = sum(len(f) for f in frames)
    if frames:
        ticks = pd.concat(frames, ignore_index=True)
        write_parquet_atomic(ticks, path, compression="zstd")
        file_bytes = path.stat().st_size
    elif path.exists():
        path.unlink()                                 # an old file for a day that is now empty
    stats = {"tick_count": tick_count, "hours_requested": requested, "hours_empty": empty,
             "hours_skipped": skipped, "requests": counter.count,
             "seconds": round(time.monotonic() - started, 2), "file_bytes": file_bytes,
             "feed_version": feed_version or config.DUKASCOPY_FEED_VERSION}
    db.mark_history_day_done(DUKASCOPY, instrument, day, stats)
    return stats


def read_history_ticks(instrument: str, day: date) -> pd.DataFrame:
    path = history_tick_file(instrument, day)
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_TICK_COLUMNS)
    return pd.read_parquet(path)


def build_month_bars(instrument: str, year: int, month: int) -> int:
    """
    (Re)build the month's 1-minute bar file from its downloaded tick files.
    Days are processed one at a time (a minute never spans two UTC days), so
    a busy month never has to fit in memory as ticks. Returns the bar count.
    """
    require_ticks_folder()
    first = date(year, month, 1)
    after = date(year + (month == 12), month % 12 + 1, 1)
    parts = []
    for row in db.get_history_days(DUKASCOPY, instrument, first, after):
        if row["tick_count"] == 0:
            continue
        path = history_tick_file(instrument, row["day"])
        if not path.exists():
            # Never write a month file that silently lacks a downloaded day.
            raise FileNotFoundError(
                f"{path} is missing although {row['day']} was downloaded "
                f"({row['tick_count']} ticks); bars for {year}-{month:02d} were not rebuilt")
        bars = ticks_to_bars(pd.read_parquet(path))
        if len(bars):
            parts.append(bars)
    out = history_bar_file(instrument, year, month)
    if not parts:
        return 0
    bars = pd.concat(parts, ignore_index=True).sort_values("time_utc", kind="stable")
    write_parquet_atomic(bars.reset_index(drop=True), out, compression="zstd")
    return len(bars)


def _fmt_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


def download_history(instruments: list[str] | None = None, start: datetime | None = None,
                     end: datetime | None = None, skip_closed: bool = True,
                     feed_version: int | None = None) -> dict:
    """
    Download every not-yet-done day in [start, end) for each instrument, in
    order, then (re)build the monthly bar files of the months touched.
    Stops early if Dukascopy blocks us or DUKASCOPY_STOP_AFTER_FAILED_DAYS
    days in a row fail. Safe to interrupt (Ctrl+C) and run again.
    """
    require_ticks_folder()
    instruments = instruments or list(config.INSTRUMENTS)
    start = timeutils.ensure_utc(start or config.HISTORY_START_UTC)
    latest = default_history_end()
    end = timeutils.ensure_utc(end or latest)
    if end > latest:
        log.warning("Not downloading after %s: Dukascopy may still be completing those days "
                    "(DUKASCOPY_LAG_DAYS = %d).", latest.date(), config.DUKASCOPY_LAG_DAYS)
        end = latest
    all_days = _days(start, end)
    summary: dict = {"stopped": None, "instruments": {}}
    failures_in_a_row = 0
    for inst in instruments:
        done = {r["day"] for r in db.get_history_days(DUKASCOPY, inst, all_days[0], all_days[-1]
                                                       + timedelta(days=1))} if all_days else set()
        todo = [d for d in all_days if d not in done]
        res = {"already_done": len(all_days) - len(todo), "downloaded": 0, "failed": 0,
               "ticks": 0, "bars_by_month": {}}
        summary["instruments"][inst] = res
        log.info("%s: %d days in range, %d already done, %d to download",
                 inst, len(all_days), res["already_done"], len(todo))
        touched: set[tuple[int, int]] = set()
        spent = 0.0
        try:
            for i, d in enumerate(todo, start=1):
                # Build the previous month's bars as soon as a new month starts.
                for ym in sorted(m for m in touched if m != (d.year, d.month)):
                    res["bars_by_month"][f"{ym[0]}-{ym[1]:02d}"] = build_month_bars(inst, *ym)
                    touched.discard(ym)
                try:
                    stats = download_day(inst, d, skip_closed, feed_version)
                except DukascopyError as exc:
                    res["failed"] += 1
                    failures_in_a_row += 1
                    log.error("Download failed (day will be retried next run): %s", exc)
                    if exc.blocked:
                        summary["stopped"] = f"Dukascopy refused the requests: {exc}"
                    elif failures_in_a_row >= config.DUKASCOPY_STOP_AFTER_FAILED_DAYS:
                        summary["stopped"] = (f"{failures_in_a_row} days in a row failed; last error: {exc}")
                    if summary["stopped"]:
                        break
                    continue
                failures_in_a_row = 0
                touched.add((d.year, d.month))
                res["downloaded"] += 1
                res["ticks"] += stats["tick_count"]
                spent += stats["seconds"]
                left = spent / i * (len(todo) - i)
                log.info("%s %s: %s ticks, %.1f MB, %d requests, %.0f s | %d/%d days | about %s left for %s",
                         inst, d, f"{stats['tick_count']:,}", stats["file_bytes"] / 1e6,
                         stats["requests"], stats["seconds"], i, len(todo), _fmt_duration(left), inst)
        finally:
            for ym in sorted(touched):
                res["bars_by_month"][f"{ym[0]}-{ym[1]:02d}"] = build_month_bars(inst, *ym)
        if summary["stopped"]:
            log.error("STOPPED: %s", summary["stopped"])
            break
    return summary


# --------------------------------------------------------------------------
# Reading history for the model - with the final-check lock
# --------------------------------------------------------------------------
def development_end_utc() -> datetime:
    """Development data must end before this: the final check minus the gap."""
    return config.FINAL_CHECK_START_UTC - timedelta(days=config.FINAL_CHECK_GAP_DAYS)


def check_history_access(instrument: str, start: datetime, end: datetime, purpose: str) -> None:
    """Raise HistoryAccessError if this read breaks the final-check or approval rules."""
    start, end = timeutils.ensure_utc(start), timeutils.ensure_utc(end)
    if purpose not in PURPOSES:
        raise HistoryAccessError(f"purpose must be one of {PURPOSES}, not {purpose!r}")
    if instrument not in config.DUKASCOPY_SYMBOLS:
        raise HistoryAccessError(f"unknown instrument {instrument!r}")
    if start >= end:
        raise HistoryAccessError(f"start {start} must be before end {end}")
    locked_from = development_end_utc()               # final check + the gap before it
    locked_to = config.FINAL_CHECK_END_UTC
    approved = config.HISTORY_APPROVED_FOR_MODEL.get(instrument, False)
    if purpose == PURPOSE_DEVELOPMENT:
        if not approved:
            raise HistoryAccessError(
                f"{instrument} history is not approved for the model yet "
                "(config.HISTORY_APPROVED_FOR_MODEL; the indices wait for the IG comparison)")
        if end > locked_from:
            raise HistoryAccessError(
                f"development data must end by {locked_from:%Y-%m-%d %H:%M} UTC: the final-check "
                f"period starts {config.FINAL_CHECK_START_UTC:%Y-%m-%d} (with a "
                f"{config.FINAL_CHECK_GAP_DAYS}-day gap) and later data comes after it")
    elif purpose == PURPOSE_SOURCE_COMPARISON:
        if start < locked_to and end > locked_from:
            raise HistoryAccessError(
                f"the final-check period {locked_from:%Y-%m-%d} to {locked_to:%Y-%m-%d} "
                "can't be used for source comparisons")
    else:  # final evaluation
        if not approved:
            raise HistoryAccessError(f"{instrument} history is not approved for the model")
        if start < config.FINAL_CHECK_START_UTC or end > config.FINAL_CHECK_END_UTC:
            raise HistoryAccessError(
                f"the final evaluation reads only the final-check period "
                f"{config.FINAL_CHECK_START_UTC:%Y-%m-%d} to {config.FINAL_CHECK_END_UTC:%Y-%m-%d}")


def load_history_1m(instrument: str, start: datetime, end: datetime, purpose: str,
                    reason: str | None = None) -> pd.DataFrame:
    """
    Dukascopy 1-minute bars with start <= time_utc < end (bar START times).
    purpose: PURPOSE_DEVELOPMENT, PURPOSE_SOURCE_COMPARISON or
    PURPOSE_FINAL_EVALUATION (needs a reason; every such read is logged and
    earlier reads are reported, because the final check should happen once).
    """
    check_history_access(instrument, start, end, purpose)
    start, end = timeutils.ensure_utc(start), timeutils.ensure_utc(end)
    if purpose == PURPOSE_FINAL_EVALUATION:
        if not (reason or "").strip():
            raise HistoryAccessError("a final-evaluation read needs a reason (it is logged)")
        earlier = db.list_final_check_access()
        if earlier:
            log.warning("The final-check period has already been read %d time(s), first on %s. "
                        "Results tuned after seeing it are no longer an honest test.",
                        len(earlier), timeutils.fmt_dual(earlier[0]["time_utc"], with_date=True))
        db.log_final_check_access([instrument], reason)
    frames = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        path = history_bar_file(instrument, y, m)
        if path.exists():
            frames.append(pd.read_parquet(path))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if not frames:
        return pd.DataFrame(columns=BAR_COLUMNS)
    bars = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(bars["time_utc"], utc=True)
    keep = (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))
    return bars[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Pilot report and status (command line)
# --------------------------------------------------------------------------
def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*.parquet")) if path.exists() else 0


def pilot_report(instruments: list[str], start: datetime, end: datetime) -> list[str]:
    """Plain-text findings from the pilot week, and estimates for the full download."""
    require_ticks_folder()
    lines: list[str] = []
    days = _days(start, end)
    full_start, full_end = config.HISTORY_START_UTC, default_history_end()
    full_days = _days(full_start, full_end)
    full_hours = sum(1 for d in full_days for h in _hour_starts(d) if not is_skipped_hour(h))
    total_seconds = total_bytes = 0.0
    ig_info = db.get_market_info()               # IG pip sizes, saved by the engine (may be empty)
    for inst in instruments:
        rows = {r["day"]: r for r in db.get_history_days(DUKASCOPY, inst, days[0], days[-1] + timedelta(days=1))}
        lines.append(f"{inst} ({config.DUKASCOPY_SYMBOLS[inst]})")
        if len(rows) < len(days):
            missing = [d.isoformat() for d in days if d not in rows]
            lines.append(f"  NOT COMPLETE: {len(missing)} day(s) not downloaded: {', '.join(missing)}")
        if not rows:
            lines.append("")
            continue
        req = sum(r["requests"] for r in rows.values())
        secs = sum(r["seconds"] for r in rows.values())
        hours_req = sum(r["hours_requested"] for r in rows.values())
        ticks = sum(r["tick_count"] for r in rows.values())
        size = sum(r["file_bytes"] for r in rows.values())
        lines.append(f"  {len(rows)} days, {ticks:,} ticks, {size / 1e6:.1f} MB of tick files, "
                     f"{req} requests in {_fmt_duration(secs)} ({secs / max(req, 1):.2f} s per request)")
        # Look at the ticks themselves.
        frames = [read_history_ticks(inst, d).assign(day=d) for d in rows if rows[d]["tick_count"]]
        ticks_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HISTORY_TICK_COLUMNS)
        if len(ticks_df):
            t = pd.to_datetime(ticks_df["time_utc"], utc=True)
            hour = t.dt.floor("1h")
            # Same rule as is_skipped_hour(), for all ticks at once: (weekday, hour) as one number.
            key = t.dt.day_of_week * 24 + t.dt.hour
            lo = config.DUKASCOPY_SKIP_FROM[0] * 24 + config.DUKASCOPY_SKIP_FROM[1]
            hi = config.DUKASCOPY_SKIP_UNTIL[0] * 24 + config.DUKASCOPY_SKIP_UNTIL[1]
            in_skip = int(((key >= lo) & (key < hi)).sum())
            lines.append(f"  ticks inside the skipped weekend window: {in_skip}"
                         + ("  <-- the skip rule would lose these; tell Claude" if in_skip else "  (skip rule is safe here)"))
            spread = ticks_df["offer"] - ticks_df["bid"]
            # In price units, and in IG pips when the engine has stored IG's pip size.
            pip = (ig_info.get(inst) or {}).get("pip_size")
            unit, div = ("IG pips", pip) if pip else ("price units", 1.0)
            lines.append(f"  spread ({unit}): median {spread.median() / div:.4g}, 99th percentile "
                         f"{spread.quantile(0.99) / div:.4g}, max {spread.max() / div:.4g}; "
                         f"offer below bid: {int((spread < 0).sum())}, offer = bid: {int((spread == 0).sum())}")
            by_day = ticks_df.groupby("day")["time_utc"].agg(["min", "max", "count"])
            for d, r in by_day.iterrows():
                lines.append(f"    {d:%a %Y-%m-%d}: {int(r['count']):>9,} ticks, first "
                             f"{timeutils.fmt_dual(pd.Timestamp(r['min']).to_pydatetime(), seconds=True)}, "
                             f"last {timeutils.fmt_dual(pd.Timestamp(r['max']).to_pydatetime(), seconds=True)}")
            weekday = t.dt.day_of_week < 5
            busy_hours = set(hour[weekday])
            silent = [h for d in days for h in _hour_starts(d)
                      if h.weekday() < 5 and pd.Timestamp(h) not in busy_hours]
            if silent:
                lines.append("  Mon-Fri hours with no ticks (UTC): "
                             + ", ".join(f"{h:%a %H}h" for h in silent[:40])
                             + (" ..." if len(silent) > 40 else ""))
        # Estimates for the whole range, from this week's speed and size.
        sec_per_hour = secs / max(hours_req, 1)
        bytes_per_day = size / len(rows)
        est_s, est_b = full_hours * sec_per_hour, len(full_days) * bytes_per_day
        total_seconds += est_s
        total_bytes += est_b
        lines.append(f"  estimate {full_start:%Y-%m-%d} to {full_end:%Y-%m-%d}: about "
                     f"{_fmt_duration(est_s)} and {est_b / 1e9:.1f} GB of tick files")
        lines.append("")
    lines.append(f"ALL: about {_fmt_duration(total_seconds)} of downloading and "
                 f"{total_bytes / 1e9:.1f} GB of tick files (rough: busier years have more ticks).")
    lines.append("     1-minute bar files are much smaller; they are measured after the download.")
    return lines


def missing_tick_files(instrument: str) -> list[date]:
    """Downloaded days (with ticks) whose tick file can't be found."""
    return [r["day"] for r in db.get_history_days(DUKASCOPY, instrument)
            if r["tick_count"] > 0 and not history_tick_file(instrument, r["day"]).exists()]


def history_status() -> list[str]:
    lines = []
    problem = ticks_folder_problem()
    where = "from .env" if config.DUKASCOPY_TICKS_DIR_FROM_ENV else "default"
    lines.append(f"Tick files: {config.DUKASCOPY_TICKS_DIR} ({where})"
                 + (f"  NOT REACHABLE: {problem}" if problem else ""))
    lines.append(f"Bar files:  {config.DUKASCOPY_BARS_1M_DIR}")
    end = default_history_end()
    all_days = _days(config.HISTORY_START_UTC, end)
    for inst in config.INSTRUMENTS:
        rows = db.get_history_days(DUKASCOPY, inst)
        in_range = {r["day"] for r in rows if config.HISTORY_START_UTC.date() <= r["day"] < end.date()}
        ticks = sum(r["tick_count"] for r in rows)
        approved = "approved for the model" if config.HISTORY_APPROVED_FOR_MODEL.get(inst) else "NOT approved for the model yet"
        tick_mb = "?" if problem else f"{_dir_bytes(config.DUKASCOPY_TICKS_DIR / inst) / 1e6:,.0f}"
        lines.append(f"{inst:<7} {len(in_range):>5}/{len(all_days)} days since "
                     f"{config.HISTORY_START_UTC:%Y-%m-%d}, {ticks:,} ticks, "
                     f"ticks {tick_mb} MB, "
                     f"bars {_dir_bytes(config.DUKASCOPY_BARS_1M_DIR / inst) / 1e6:,.1f} MB  ({approved})")
        if not problem:
            missing = missing_tick_files(inst)
            if missing:
                lines.append(f"        WARNING: {len(missing)} downloaded day(s) have no tick file in the "
                             f"tick folder, e.g. {missing[0]} (wrong folder or drive?)")
    lines.append(f"Downloads stop at {end:%Y-%m-%d} (today minus {config.DUKASCOPY_LAG_DAYS} days).")
    lines.append(f"Final-check period: {config.FINAL_CHECK_START_UTC:%Y-%m-%d} to "
                 f"{config.FINAL_CHECK_END_UTC:%Y-%m-%d} (not included); development data ends "
                 f"{development_end_utc():%Y-%m-%d}.")
    reads = db.list_final_check_access()
    lines.append(f"Final-check period read {len(reads)} time(s)"
                 + (f", last on {timeutils.fmt_dual(reads[-1]['time_utc'], with_date=True)}" if reads else "."))
    return lines


def move_ticks(source: Path | None = None) -> dict:
    """
    Move tick files from `source` (default: data/dukascopy/ticks) into the
    folder set in .env. Each file is copied, checked by size, then deleted
    from the source. Nothing is overwritten: if any file already exists in
    the target, nothing is moved.
    """
    source = Path(source or config.DUKASCOPY_TICKS_DIR_DEFAULT)
    target = config.DUKASCOPY_TICKS_DIR
    if not config.DUKASCOPY_TICKS_DIR_FROM_ENV:
        raise TicksFolderError("Set DUKASCOPY_TICKS_DIR in .env first (the folder to move the ticks to).")
    require_ticks_folder()
    if source.resolve() == target.resolve() or not source.exists():
        return {"moved": 0, "bytes": 0}
    files = sorted(source.rglob("*.parquet"))
    clashes = [f for f in files if (target / f.relative_to(source)).exists()]
    if clashes:
        raise TicksFolderError(f"{len(clashes)} file(s) already exist in {target}, e.g. "
                               f"{clashes[0].relative_to(source)}; nothing was moved")
    moved = total = 0
    for f in files:
        dest = target / f.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copy2(f, tmp)
        size = f.stat().st_size
        if tmp.stat().st_size != size:
            tmp.unlink()
            raise TicksFolderError(f"copy of {f} came out a different size; stopped after {moved} files")
        os.replace(tmp, dest)
        f.unlink()
        moved += 1
        total += size
    for d in sorted((p for p in source.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()                                    # empty instrument folders
    if source.is_dir() and not any(source.iterdir()):
        source.rmdir()
    return {"moved": moved, "bytes": total}


def _parse_day(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dukascopy history: pilot, download, status.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_pilot = sub.add_parser("pilot", help="every hour of one week for all five instruments, then a report")
    p_pilot.add_argument("--feed", type=int, choices=(1, 2), default=None,
                         help="dukascopy-python data feed (default: config.DUKASCOPY_FEED_VERSION)")
    p_pilot.add_argument("--report-only", action="store_true", help="don't download, just report")
    p_dl = sub.add_parser("download", help="download history (resumable; Ctrl+C is safe)")
    p_dl.add_argument("--instruments", nargs="+", choices=list(config.INSTRUMENTS), default=None)
    p_dl.add_argument("--from", dest="start", type=_parse_day, default=None,
                      help="first UTC day, YYYY-MM-DD (default: config.HISTORY_START_UTC)")
    p_dl.add_argument("--to", dest="end", type=_parse_day, default=None,
                      help="day AFTER the last one, YYYY-MM-DD (default: today minus the lag)")
    p_dl.add_argument("--feed", type=int, choices=(1, 2), default=None)
    sub.add_parser("status", help="what has been downloaded, and the final-check settings")
    sub.add_parser("move-ticks", help="move tick files from data/dukascopy/ticks to DUKASCOPY_TICKS_DIR")
    args = parser.parse_args(argv)

    timeutils.configure_logging(log_file=config.LOGS_DIR / "history.log")
    try:
        return _run(args)
    except TicksFolderError as exc:
        print(f"\nSTOPPED: {exc}")
        return 1


def _run(args) -> int:
    if args.command == "status":
        print("\n".join(history_status()))
        return 0
    if args.command == "move-ticks":
        res = move_ticks()
        print(f"Moved {res['moved']} tick files ({res['bytes'] / 1e6:.1f} MB) to {config.DUKASCOPY_TICKS_DIR}.\n")
        print("\n".join(history_status()))
        return 0
    if args.command == "pilot":
        instruments = list(config.INSTRUMENTS)
        result = {"stopped": None}
        if not args.report_only:
            print(f"Pilot: every hour from {config.PILOT_START_UTC:%Y-%m-%d} to "
                  f"{config.PILOT_END_UTC:%Y-%m-%d} (not included) for {', '.join(instruments)}.")
            try:
                result = download_history(instruments, config.PILOT_START_UTC, config.PILOT_END_UTC,
                                          skip_closed=False, feed_version=args.feed)
            except KeyboardInterrupt:
                print("\nStopped by you. Run the same command to continue the pilot.")
                return 1
        print("\n" + "\n".join(pilot_report(instruments, config.PILOT_START_UTC, config.PILOT_END_UTC)))
        if result["stopped"]:
            print(f"\nSTOPPED: {result['stopped']}")
            return 1
        return 0
    try:
        result = download_history(args.instruments, args.start, args.end, feed_version=args.feed)
    except KeyboardInterrupt:
        print("\nStopped by you. Finished days are kept; run the same command to continue.")
        return 1
    print("\n".join(history_status()))
    if result["stopped"]:
        print(f"\nSTOPPED: {result['stopped']}\nFinished days are kept; run the same command again later.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
