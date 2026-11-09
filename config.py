"""
config.py - non-secret project settings in one place.

Credentials (IG username, password, API key) are NOT here. They live in the
.env file, which Git ignores, and only broker.py reads them.

Everything in this file is safe to commit.
"""

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
# "epic" is IG's identifier for each market. They are left as None until you
# look them up on your demo account (step 2 adds a lookup helper). Code that
# needs an epic will stop with a clear message while it is still None.
#
# "group" marks correlated instruments: no two open trades may share a group
# (EUR/USD + GBP/USD, US 500 + Germany 40). USD/JPY has its own group.
INSTRUMENTS = {
    "EURUSD": {"name": "EUR/USD",    "epic": None, "asset_class": "forex", "group": "USD_MAJORS"},
    "GBPUSD": {"name": "GBP/USD",    "epic": None, "asset_class": "forex", "group": "USD_MAJORS"},
    "USDJPY": {"name": "USD/JPY",    "epic": None, "asset_class": "forex", "group": "USDJPY"},
    "US500":  {"name": "US 500",     "epic": None, "asset_class": "index", "group": "EQUITY_INDICES"},
    "GER40":  {"name": "Germany 40", "epic": None, "asset_class": "index", "group": "EQUITY_INDICES"},
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
