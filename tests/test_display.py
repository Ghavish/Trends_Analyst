"""Tests for display.py: plain wording and per-instrument number formats."""

from datetime import datetime, timedelta, timezone

import broker
import display

UTC = timezone.utc
NOW = datetime(2026, 11, 9, 13, 17, 20, tzinfo=UTC)


def test_stream_status_in_plain_words():
    assert display.stream_status_text("CONNECTED:WS-STREAMING") == "Connected"
    assert display.stream_status_text("CONNECTED:HTTP-STREAMING") == "Connected"
    assert display.stream_status_text("CONNECTED:STREAM-SENSING") == "Connecting"
    assert display.stream_status_text("DISCONNECTED:WILL-RETRY") == "Reconnecting"
    assert display.stream_status_text("STALLED") == "Waiting for data"
    assert display.stream_status_text("DISCONNECTED") == "Disconnected"
    assert display.stream_status_text(None) == "Unknown"


def test_market_status_in_plain_words():
    assert display.market_status_text("DEAL") == "Open"
    assert display.market_status_text("CLOSED") == "Closed"
    assert display.market_status_text("SUSPEND") == "Suspended"
    assert display.market_status_text("EDIT") == "Limited (EDIT)"
    assert display.market_status_text(None) == "Unknown"


def test_prices_keep_igs_decimals():
    assert display.format_price(1.1194, 5) == "1.11940"
    assert display.format_price(157.9, 3) == "157.900"
    assert display.format_price(7431.46, None) == "7431.46"
    assert display.format_price(None, 5) == "-"


def test_spread_in_pips_or_points():
    assert display.format_spread(1.1194, 1.11949, 0.0001, "pips") == "0.9 pips"
    assert display.format_spread(157.906, 157.916, 0.01, "pips") == "1.0 pips"
    assert display.format_spread(26093.3, 26094.7, 1.0, "points") == "1.4 points"
    assert display.format_spread(1.1, 1.10009, None, "pips") == "9e-05"   # pip size unknown
    assert display.spread_unit("EURUSD") == "pips" and display.spread_unit("GER40") == "points"


def test_ig_pip_size_parsing():
    assert broker._pip_size("0.0001 USD/EUR") == 0.0001
    assert broker._pip_size("1 Index Point") == 1.0
    assert broker._pip_size("Index Point") is None
    assert broker._pip_size(None) is None


def test_time_text_adds_date_only_when_not_today():
    assert display.time_text(NOW - timedelta(seconds=5), NOW) == "13:17:15 UTC (17:17:15 MUT)"
    assert display.time_text(NOW - timedelta(days=1), NOW) == "2026-11-08 13:17:20 UTC (2026-11-08 17:17:20 MUT)"


def feed(open_codes):
    return [{"instrument": k, "market_status": c, "last_tick_utc": NOW - timedelta(seconds=3)}
            for k, c in open_codes.items()]


def test_overall_status_ok():
    level, msg = display.overall_status(True, "CONNECTED:WS-STREAMING",
                                        feed({"EURUSD": "DEAL", "GER40": "CLOSED"}), [], None, NOW)
    assert level == "ok" and msg.startswith("All OK: recording 5 markets (1 open, 4 closed or unknown)")
    assert "Latest price 3 s ago" in msg


def test_overall_status_problems_in_order_of_importance():
    assert display.overall_status(False, "CONNECTED:WS-STREAMING", [], [], None, NOW)[0] == "error"
    level, msg = display.overall_status(True, "DISCONNECTED:WILL-RETRY", [], [], None, NOW)
    assert level == "warning" and "reconnecting" in msg
    gaps = [{"instrument": "GER40", "kind": "QUIET", "start_utc": NOW - timedelta(minutes=6)}]
    level, msg = display.overall_status(True, "CONNECTED:WS-STREAMING", [], gaps, None, NOW)
    assert level == "warning" and "Germany 40" in msg
    # A price arrived after the quiet gap began: no warning, even before the engine closes the gap.
    fresh = [{"instrument": "GER40", "market_status": "DEAL", "last_tick_utc": NOW - timedelta(seconds=2)}]
    assert display.overall_status(True, "CONNECTED:WS-STREAMING", fresh, gaps, None, NOW)[0] == "ok"
    level, msg = display.overall_status(True, "CONNECTED:WS-STREAMING", [], [], "refused", NOW)
    assert level == "warning" and "open/closed status is unavailable" in msg
