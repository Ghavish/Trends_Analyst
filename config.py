"""
config.py - non-secret project settings in one place.

Credentials (IG username, password, API key) are NOT here. They live in the
.env file, which Git ignores, and only broker.py reads them.

Everything in this file is safe to commit.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------
# PROJECT_ROOT is the folder this file sits in (the root of Trends_Analyst).
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"      # price + spread data (Parquet) and the SQLite file
MODELS_DIR = PROJECT_ROOT / "models"  # saved model files (later phases)
LOGS_DIR = PROJECT_ROOT / "logs"      # log files

# The single SQLite database shared by the engine (signals.py) and the
# dashboard (app.py). Only db.py opens it.
DB_PATH = DATA_DIR / "trends_analyst.db"

# --------------------------------------------------------------------------
# Account safety
# --------------------------------------------------------------------------
# The project works on the IG DEMO account only. broker.py refuses to start
# if IG_ACC_TYPE in .env is anything other than this value. There is no
# live-trading path in the code.
ALLOWED_ACC_TYPE = "DEMO"

# IG's demo REST server. broker.py checks the connection really points here.
IG_DEMO_BASE_URL = "https://demo-api.ig.com/gateway/deal"

# Your IG account type (you confirmed CFD). broker.py stops if the login
# reports a different type (e.g. SPREADBET), because epics and sizing differ.
EXPECTED_ACCOUNT_TYPE = "CFD"

# Our own cap on IG REST requests per minute (non-trading). IG Labs' FAQ
# (checked 9 Nov 2026) states 30 per account and 60 per app. We stay well
# below that; live prices come from streaming, which doesn't count.
# test_connection.py shows your API key's actual allowances.
IG_REST_REQUESTS_PER_MINUTE = 20

# --------------------------------------------------------------------------
# Time zones (IANA names, understood by Python's zoneinfo and by pandas)
# --------------------------------------------------------------------------
# Storage is always UTC. These are used for session features (market-local
# time) and for display (Mauritius time next to UTC).
TZ_UTC = "UTC"
TZ_MAURITIUS = "Indian/Mauritius"   # UTC+4 all year, no daylight saving
TZ_TOKYO = "Asia/Tokyo"
TZ_LONDON = "Europe/London"
TZ_FRANKFURT = "Europe/Berlin"      # IANA has no "Europe/Frankfurt"; Germany uses Europe/Berlin
TZ_NEW_YORK = "America/New_York"

# The trading day (for the daily loss limit and daily summaries) resets at
# the forex daily rollover: 17:00 New York time (Section 6.4).
TRADING_DAY_RESET_TZ = TZ_NEW_YORK
TRADING_DAY_RESET_HOUR = 17

# --------------------------------------------------------------------------
# Instruments (Section 4.3)
# --------------------------------------------------------------------------
# "epic" is IG's identifier for each market. Chosen on 9 Nov 2026 from the
# demo account's search results (test_connection.py --market): the Mini
# forex contracts and the $1 Cash index contracts. Same spreads as the larger
# contracts, smaller minimum deal sizes (finer position sizing at 1% risk),
# and both indices priced in USD (the account currency).
#
# "group" marks correlated instruments: no two open trades may share a group
# (EUR/USD + GBP/USD, US 500 + Germany 40). USD/JPY has its own group.
#
# "search" is the text sent to IG's market search to find the epic.
INSTRUMENTS = {
    "EURUSD": {"name": "EUR/USD",    "search": "EUR/USD",    "epic": "CS.D.EURUSD.MINI.IP", "asset_class": "forex", "group": "USD_MAJORS"},
    "GBPUSD": {"name": "GBP/USD",    "search": "GBP/USD",    "epic": "CS.D.GBPUSD.MINI.IP", "asset_class": "forex", "group": "USD_MAJORS"},
    "USDJPY": {"name": "USD/JPY",    "search": "USD/JPY",    "epic": "CS.D.USDJPY.MINI.IP", "asset_class": "forex", "group": "USDJPY"},
    "US500":  {"name": "US 500",     "search": "US 500",     "epic": "IX.D.SPTRD.IFMM.IP", "asset_class": "index", "group": "EQUITY_INDICES"},
    "GER40":  {"name": "Germany 40", "search": "Germany 40", "epic": "IX.D.DAX.IFMS.IP", "asset_class": "index", "group": "EQUITY_INDICES"},
}

# --------------------------------------------------------------------------
# Hard risk limits (Section 12.2, values chosen by you)
# --------------------------------------------------------------------------
# These are applied in the engine's filter step (Section 8, step 4) as soon
# as signals exist, so the signals-only phase follows the same rules.
# broker.py adds a final check before any order (Phase 3).

RISK_PER_TRADE_PCT = 1.0          # % of account balance risked per trade (loss if SL is hit)
MAX_OPEN_TRADES = 2               # across all instruments
MAX_OPEN_TRADES_PER_INSTRUMENT = 1
# No new trade if an open trade exists in the same correlated "group" above.
BLOCK_SAME_CORRELATED_GROUP = True

DAILY_LOSS_LIMIT_PCT = 3.0        # % of balance, realised + open P/L; engine stops for the day

# Bug guard: a fixed per-instrument maximum size in lots, set so that an SL
# hit can't lose more than about this % of the balance.
MAX_LOSS_AT_SL_PCT = 2.0
# The lot values depend on your balance, each market's contract size / value
# per point (read from IG in step 2) and typical ATR-based stop distances
# (Phase 2), so they are NOT filled in yet. While a value is None, the
# engine will refuse to size a trade for that instrument.
MAX_POSITION_LOTS = {
    "EURUSD": None,
    "GBPUSD": None,
    "USDJPY": None,
    "US500": None,
    "GER40": None,
}

# All orders must be sent with SL and TP attached at IG, so positions stay
# protected even if the engine stops (enforced in broker.py in Phase 3).
REQUIRE_SL_AND_TP_ON_ORDERS = True

# --------------------------------------------------------------------------
# Automatic halt (you chose these on 9 Nov 2026)
# --------------------------------------------------------------------------
# The engine sets a halt flag (stops trading until you clear it manually)
# when any of these is reached. Price recording keeps trying to reconnect.
HALT_AFTER_FAILED_RECONNECTS = 5       # failed stream reconnection attempts in a row
HALT_AFTER_DISCONNECTS = 10            # stream disconnections ...
HALT_DISCONNECT_WINDOW_MINUTES = 60    # ... within this many minutes
HALT_AFTER_API_ERRORS = 5              # failed IG REST requests in a row (token
                                       # expiries are fixed by re-login, not counted)

# --------------------------------------------------------------------------
# Live price stream (broker.PriceStream)
# --------------------------------------------------------------------------
# The Lightstreamer library first tries to recover a dropped connection by
# itself. If it is still not streaming after this many seconds, or IG closes
# the session, we log in again (fresh tokens) and open a new connection.
STREAM_RECOVERY_GRACE_SECONDS = 60
# A (re)connection that hasn't started streaming after this long has failed.
STREAM_CONNECT_TIMEOUT_SECONDS = 30
# Waits before successive reconnection attempts (the last value repeats).
STREAM_RECONNECT_DELAYS_SECONDS = (5, 10, 20, 40, 60)

# --------------------------------------------------------------------------
# Price recording (fetch_data.py, run by the engine signals.py)
# --------------------------------------------------------------------------
TICKS_DIR = DATA_DIR / "ticks"        # raw price updates: ticks/<INSTRUMENT>/<YYYY-MM-DD>.parquet
BARS_1M_DIR = DATA_DIR / "bars_1m"    # 1-minute bars:    bars_1m/<INSTRUMENT>/<YYYY-MM-DD>.parquet
# Files are split at UTC midnight. (The trading day for the daily loss limit
# is separate: it resets at 17:00 New York, see TRADING_DAY_RESET_*.)

# How often the day's files are rewritten (at most this much is lost in a crash).
RECORDER_FLUSH_SECONDS = 60

# A 1-minute bar labelled 10:00 covers [10:00:00, 10:01:00) and is only
# complete at 10:01. It is written to file only after this extra grace period,
# so a price that arrives a moment late still lands in the right bar.
BAR_CLOSE_GRACE_SECONDS = 2

# "Quiet" gap: no price update for this many minutes while IG reports the
# market as open. Logged as a possible gap (could be a genuine lull).
QUIET_GAP_MINUTES = {
    "EURUSD": 5,
    "GBPUSD": 5,
    "USDJPY": 5,
    "US500": 5,
    "GER40": 5,
}

# Market status comes from IG's PRICE subscription field DLG_FLAG. IG's
# reference lists the values CLOSED, CALL, DEAL, EDIT, CLOSINGSONLY,
# DEALNOEDIT, AUCTION, AUCTIONNOEDIT, SUSPEND but does not define them.
# Our interpretation: only these mean "open, prices should be moving".
# Every status change is logged, so this can be checked against real data.
MARKET_OPEN_DLG_FLAGS = ("DEAL", "DEALNOEDIT")

# --------------------------------------------------------------------------
# Historical data from Dukascopy (Phase 2, step 1; fetch_data.py)
# --------------------------------------------------------------------------
# Downloaded with the dukascopy-python library (pinned in requirements.txt).
# Only fetch_data.py uses that library. It calls Dukascopy's web-chart data
# endpoints, which are NOT an official, documented API: they can change or
# be blocked at any time.
DUKASCOPY_DIR = DATA_DIR / "dukascopy"
# Ticks, one file per instrument per UTC day: <ticks folder>/<INSTRUMENT>/<YYYY-MM-DD>.parquet
#   columns: time_utc, bid, offer, bid_volume, offer_volume
# The ticks are large (~14 GB from 2019), so they may live on another drive:
# put e.g.  DUKASCOPY_TICKS_DIR=D:\TrendsAnalyst\dukascopy\ticks  in .env
# (a per-laptop setting, so it stays out of Git). That folder must already
# exist: if the drive is unplugged or has another letter, downloads stop
# instead of quietly writing somewhere else. Without the setting, ticks go to
# data/dukascopy/ticks (created automatically).
DUKASCOPY_TICKS_DIR_DEFAULT = DUKASCOPY_DIR / "ticks"


def _setting_from_env_file(name: str) -> str | None:
    """One non-secret setting from the environment or .env (nothing else is loaded)."""
    value = os.environ.get(name)
    if value is None:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            from dotenv import dotenv_values
            value = dotenv_values(env_file).get(name)
    value = (value or "").strip().strip('"').strip("'")
    return value or None


_ticks_setting = _setting_from_env_file("DUKASCOPY_TICKS_DIR")
# True when the tick folder was chosen in .env (then it must exist already).
DUKASCOPY_TICKS_DIR_FROM_ENV = _ticks_setting is not None
DUKASCOPY_TICKS_DIR = (PROJECT_ROOT / _ticks_setting) if _ticks_setting else DUKASCOPY_TICKS_DIR_DEFAULT
# 1-minute bid/offer bars built from those ticks, one file per UTC month:
#   bars_1m/<INSTRUMENT>/<YYYY-MM>.parquet  (same columns as the IG bars).
# They stay on the laptop, so training and backtests don't need the other drive.
DUKASCOPY_BARS_1M_DIR = DUKASCOPY_DIR / "bars_1m"

# Dukascopy's code for each instrument, as listed in dukascopy-python 6.0.0's
# instrument table (dukascopy_python.instruments). Dukascopy files both
# indices under its "IDX_CASH" group; whether they match IG's Cash prices is
# checked by comparing them with the IG bars you record (not yet proven).
DUKASCOPY_SYMBOLS = {
    "EURUSD": "EUR-USD",
    "GBPUSD": "GBP-USD",
    "USDJPY": "USD-JPY",
    "US500": "USA500.IDX-USD",     # "US 500 Index"
    "GER40": "DEU.IDX-EUR",        # "Germany 40 Index" (quoted in EUR at Dukascopy)
}

# First day downloaded (you chose 2019; the pilot shows whether 2015 is worth adding).
HISTORY_START_UTC = datetime(2019, 1, 1, tzinfo=timezone.utc)

# Dukascopy may publish recent hours late, and an hour that isn't published
# yet looks the same as an hour with no trading. So days newer than this are
# not downloaded (otherwise they could be stored as complete while missing data).
DUKASCOPY_LAG_DAYS = 3

# Which of dukascopy-python's two data feeds to use: 2 = jetta.dukascopy.com
# (one request per hour of ticks, bid/ask labelled by name), 1 = freeserv
# chart feed. Both are undocumented; 2 is tried first, 1 is the fallback.
DUKASCOPY_FEED_VERSION = 2

# Being polite to Dukascopy's servers (no published limits exist):
DUKASCOPY_MIN_SECONDS_BETWEEN_REQUESTS = 1.0  # at most one request per second
DUKASCOPY_REQUEST_TIMEOUT_SECONDS = 30        # give up on a request after this long
DUKASCOPY_HTTP_RETRIES = 4                    # retries on 429 / 5xx / network errors (inside requests),
DUKASCOPY_RETRY_BACKOFF_SECONDS = 2.0         # waiting 0, 4, 8, 16 s (or as long as the server asks)
DUKASCOPY_STOP_AFTER_FAILED_DAYS = 3          # stop the run after this many days in a row fail
                                              # (a day fails at its first hour that fails after retries)

# Hours that are never requested because these markets are closed: from
# Saturday 00:00 UTC up to Sunday 20:00 UTC. (Forex reopens at 17:00 New York
# = 21:00 or 22:00 UTC on Sunday; Dukascopy lists its index CFDs as opening
# on Sunday evening too.) The pilot downloads every hour of one week, so you
# can check that no ticks fall inside this window before it is relied on.
DUKASCOPY_SKIP_FROM = (5, 0)    # (weekday, hour) UTC; Monday = 0, so 5 = Saturday
DUKASCOPY_SKIP_UNTIL = (6, 20)  # Sunday 20:00 UTC (not included in the skip)

# The pilot: every hour of one week for all five instruments, to check that
# downloads work and to measure time and disk space before the bulk download.
# Chosen inside the development period (not the final-check period below).
PILOT_START_UTC = datetime(2026, 3, 2, tzinfo=timezone.utc)   # Monday
PILOT_END_UTC = datetime(2026, 3, 9, tzinfo=timezone.utc)     # next Monday (not included)

# --------------------------------------------------------------------------
# Final-check period (Section 7.9) - looked at ONCE, at the very end
# --------------------------------------------------------------------------
# The most recent 6 complete months before this step (decided 9 Nov 2026).
# These dates are FIXED: don't move them later to make room for more data.
# fetch_data.load_history_1m() refuses to return any bar in this period
# (or in the gap day before it) unless the purpose is the final evaluation,
# and every final-evaluation read is logged in the database.
FINAL_CHECK_START_UTC = datetime(2026, 5, 1, tzinfo=timezone.utc)
FINAL_CHECK_END_UTC = datetime(2026, 11, 1, tzinfo=timezone.utc)   # not included
# Gap between development data and the final check, so that no trade
# outcome leaks across the boundary (Section 7.9, "at least a one-day gap").
FINAL_CHECK_GAP_DAYS = 1

# Instruments whose Dukascopy history may be used to build the model.
# The indices stay out until the comparison with IG's own prices is done
# (you decided this on 9 Nov 2026). They are still downloaded.
HISTORY_APPROVED_FOR_MODEL = {
    "EURUSD": True,
    "GBPUSD": True,
    "USDJPY": True,
    "US500": False,
    "GER40": False,
}

# --------------------------------------------------------------------------
# Engine (signals.py)
# --------------------------------------------------------------------------
ENGINE_JOB_POLL_SECONDS = 5          # how often dashboard requests are picked up
ENGINE_HALT_CHECK_SECONDS = 60       # how often the automatic-halt rules are checked
FEED_STATUS_SECONDS = 5              # how often latest prices are written for the dashboard
# The engine writes a heartbeat; if it is older than this, the engine is
# considered not running (used by the dashboard and test_connection.py).
ENGINE_HEARTBEAT_STALE_SECONDS = 180
