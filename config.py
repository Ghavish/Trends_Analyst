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
