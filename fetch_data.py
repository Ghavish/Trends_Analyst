"""
fetch_data.py - records the live IG feed (Phase 1) and, later, downloads
history from Dukascopy (Phase 2).

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
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

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


def write_parquet_atomic(df: pd.DataFrame, path: Path, attempts: int = 5) -> None:
    """Write to <name>.tmp then swap it in, so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
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
