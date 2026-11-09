"""
timeutils.py - one place for every time-zone conversion.

Rules (Sections 6.3 and 6.4 of the plan):
  * Everything is STORED in UTC.
  * Session features convert UTC to the market's local time, so "London open"
    always means 08:00 in London, whatever the season.
  * Everything SHOWN to you has Mauritius time (UTC+4) next to UTC.
  * The trading day resets at the forex rollover, 17:00 New York time.

Python's zoneinfo knows each country's daylight-saving rules. On Windows it
needs the "tzdata" package (listed in requirements.txt), because Windows has
no built-in IANA time-zone database.
"""

import logging
from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo

import config

UTC = timezone.utc
MUT = ZoneInfo(config.TZ_MAURITIUS)


# --------------------------------------------------------------------------
# Basic conversions
# --------------------------------------------------------------------------
def now_utc() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(ts: datetime) -> datetime:
    """
    Return `ts` as an aware UTC datetime.

    A naive datetime (no time zone attached) is rejected rather than guessed,
    because guessing the zone is exactly the bug Section 6.3 warns about.
    """
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(
            f"Naive datetime {ts!r}: attach a time zone before converting "
            "(e.g. datetime(..., tzinfo=timezone.utc))."
        )
    return ts.astimezone(UTC)


def from_epoch_ms(ms: int | float | str) -> datetime:
    """Convert milliseconds since 1970-01-01 UTC (as IG's stream sends) to UTC datetime."""
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC)


def to_market_local(ts: datetime, tz_name: str) -> datetime:
    """Convert a UTC time to a market's local time, e.g. config.TZ_LONDON."""
    return ensure_utc(ts).astimezone(ZoneInfo(tz_name))


def to_mut(ts: datetime) -> datetime:
    """Convert a UTC time to Mauritius time (display only)."""
    return ensure_utc(ts).astimezone(MUT)


def local_to_utc(day: date, local_time: time, tz_name: str) -> datetime:
    """
    Turn a wall-clock time in a given zone into UTC,
    e.g. (2026-03-16, 09:30, New York) -> 13:30 UTC.
    """
    local = datetime.combine(day, local_time, tzinfo=ZoneInfo(tz_name))
    return local.astimezone(UTC)


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
def fmt_dual(ts: datetime, with_date: bool = False, seconds: bool = False) -> str:
    """
    Format a time as UTC with Mauritius time alongside, in the trade-card
    style of Section 7.5:  "08:45 UTC (12:45 MUT)".

    with_date=True adds the date to each side (the dates can differ around
    midnight):  "2026-11-09 22:30 UTC (2026-11-10 02:30 MUT)".
    """
    utc = ensure_utc(ts)
    mut = to_mut(utc)
    fmt = "%H:%M:%S" if seconds else "%H:%M"
    if with_date:
        fmt = "%Y-%m-%d " + fmt
    return f"{utc.strftime(fmt)} UTC ({mut.strftime(fmt)} MUT)"


# --------------------------------------------------------------------------
# Trading day (forex rollover at 17:00 New York)
# --------------------------------------------------------------------------
def trading_day_start_utc(ts: datetime) -> datetime:
    """
    UTC time at which the trading day containing `ts` started: the most
    recent 17:00 New York time at or before `ts`.

    17:00 New York is 22:00 UTC in northern winter and 21:00 UTC in northern
    summer (Section 6.4); zoneinfo handles the switch.
    """
    ny = ZoneInfo(config.TRADING_DAY_RESET_TZ)
    local = ensure_utc(ts).astimezone(ny)
    reset = time(config.TRADING_DAY_RESET_HOUR, 0)
    start_day = local.date() if local.time() >= reset else local.date() - timedelta(days=1)
    return local_to_utc(start_day, reset, config.TRADING_DAY_RESET_TZ)


def trading_day_label(ts: datetime) -> date:
    """
    Name of the trading day containing `ts`: the New York calendar date on
    which that day ENDS (the usual forex convention). For example, anything
    from Monday 17:00 NY to Tuesday 16:59:59 NY is "Tuesday".
    """
    start = trading_day_start_utc(ts)
    ny = ZoneInfo(config.TRADING_DAY_RESET_TZ)
    return start.astimezone(ny).date() + timedelta(days=1)


# --------------------------------------------------------------------------
# Log messages with UTC and Mauritius time
# --------------------------------------------------------------------------
class DualTimeFormatter(logging.Formatter):
    """Log formatter whose timestamp is 'YYYY-MM-DD HH:MM:SS UTC (... MUT)'."""

    def formatTime(self, record, datefmt=None):
        ts = datetime.fromtimestamp(record.created, tz=UTC)
        return fmt_dual(ts, with_date=True, seconds=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Send log messages to the console with UTC + MUT timestamps."""
    handler = logging.StreamHandler()
    handler.setFormatter(DualTimeFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("trading_ig").setLevel(logging.WARNING)   # never log usernames
