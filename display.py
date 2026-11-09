"""
display.py - how things are shown to you: plain words instead of IG / code
terms, and prices and spreads formatted per instrument.

Used by the dashboard (app.py) and later by Telegram alerts, so the wording
is the same everywhere. Display only: nothing here is used for calculations.
"""

from __future__ import annotations

from datetime import datetime

import config
import timeutils

# --------------------------------------------------------------------------
# Plain wording
# --------------------------------------------------------------------------
def stream_status_text(raw: str | None) -> str:
    """Lightstreamer status -> plain words."""
    if not raw:
        return "Unknown"
    if raw.startswith("CONNECTED:") and raw != "CONNECTED:STREAM-SENSING":
        return "Connected"
    if raw in ("CONNECTING", "CONNECTED:STREAM-SENSING", "starting"):
        return "Connecting"
    if raw == "STALLED":
        return "Waiting for data"
    if raw.startswith("DISCONNECTED:"):
        return "Reconnecting"
    if raw in ("DISCONNECTED", "stopped"):
        return "Disconnected"
    return raw


_MARKET_TEXT = {
    "CLOSED": "Closed",
    "SUSPEND": "Suspended",
    "AUCTION": "Auction",
    "AUCTIONNOEDIT": "Auction",
}


def market_is_open(code: str | None) -> bool:
    return code in config.MARKET_OPEN_DLG_FLAGS


def market_status_text(code: str | None) -> str:
    """IG's DLG_FLAG -> plain words ('Open', 'Closed', ...)."""
    if not code:
        return "Unknown"
    if market_is_open(code):
        return "Open"
    return _MARKET_TEXT.get(code, f"Limited ({code})")


GAP_KIND_TEXT = {
    "OUTAGE": "Connection to IG lost",
    "QUIET": "No prices while market open",
    "ENGINE_OFF": "Engine not running",
}


def gap_kind_text(kind: str) -> str:
    return GAP_KIND_TEXT.get(kind, kind)


def instrument_name(key: str) -> str:
    return config.INSTRUMENTS.get(key, {}).get("name", key)


# --------------------------------------------------------------------------
# Numbers and times
# --------------------------------------------------------------------------
def spread_unit(instrument: str) -> str:
    return "pips" if config.INSTRUMENTS.get(instrument, {}).get("asset_class") == "forex" else "points"


def format_price(value, decimals: int | None) -> str:
    """Price with the instrument's number of decimals (from IG), e.g. 1.11940."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}" if decimals is not None else f"{value:g}"


def format_spread(bid, offer, pip_size: float | None, unit: str) -> str:
    """Offer minus bid in pips (forex) / points (indices), e.g. '0.9 pips'."""
    if bid is None or offer is None:
        return "-"
    if pip_size:
        return f"{(offer - bid) / pip_size:.1f} {unit}"
    return f"{offer - bid:.6g}"          # pip size unknown: raw price difference


def age_text(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 120:
        return f"{max(seconds, 0):.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def time_text(ts: datetime | None, now: datetime | None = None, seconds: bool = True) -> str:
    """UTC with MUT alongside; the date is added when it isn't today (UTC)."""
    if ts is None:
        return "-"
    now = now or timeutils.now_utc()
    return timeutils.fmt_dual(ts, with_date=ts.date() != now.date(), seconds=seconds)


# --------------------------------------------------------------------------
# The one-line overall status at the top of the dashboard
# --------------------------------------------------------------------------
def overall_status(engine_running: bool, stream_raw: str | None, feed: list[dict],
                   open_gaps: list[dict], market_status_error: str | None,
                   now: datetime | None = None) -> tuple[str, str]:
    """
    Returns (level, message): level is 'ok', 'warning' or 'error'.
    feed: rows from db.get_feed_status(); open_gaps: db.list_gaps(open_only=True).
    """
    now = now or timeutils.now_utc()
    if not engine_running:
        return "error", ("Not recording: the engine is not running. "
                         "Start it with `python signals.py`.")
    stream = stream_status_text(stream_raw)
    if stream != "Connected":
        return "warning", (f"Engine running, but the connection to IG is: {stream.lower()}. "
                           "It reconnects by itself.")
    # A QUIET gap is closed by the engine only once a minute; ignore it as soon
    # as a newer price has arrived, so this line agrees with the price table.
    last_tick = {f["instrument"]: f.get("last_tick_utc") for f in feed}
    quiet = sorted({instrument_name(g["instrument"]) for g in open_gaps
                    if g["kind"] == "QUIET"
                    and not (last_tick.get(g["instrument"]) and last_tick[g["instrument"]] > g["start_utc"])})
    if quiet:
        return "warning", (f"No prices for {', '.join(quiet)} for several minutes although "
                           "IG says the market is open. Recording continues.")
    if market_status_error:
        return "warning", "Prices are recorded, but IG's open/closed status is unavailable."
    open_count = sum(1 for f in feed if market_is_open(f.get("market_status")))
    latest = max((f["last_tick_utc"] for f in feed if f.get("last_tick_utc")), default=None)
    latest_text = f" Latest price {age_text((now - latest).total_seconds())} ago." if latest else ""
    return "ok", (f"All OK: recording {len(config.INSTRUMENTS)} markets "
                  f"({open_count} open, {len(config.INSTRUMENTS) - open_count} closed or unknown)."
                  f"{latest_text}")
