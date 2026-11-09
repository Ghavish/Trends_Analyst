"""
Tests for the dashboard (app.py), run headless with Streamlit's AppTest on a
temporary database. They check that it shows the data and that its buttons
only write requests to the jobs table (the engine does the actual work).
"""

from datetime import timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
import db
import timeutils

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "TICKS_DIR", tmp_path / "data" / "ticks")
    monkeypatch.setattr(config, "BARS_1M_DIR", tmp_path / "data" / "bars_1m")
    yield tmp_path


def run_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + \
            [e.value for e in at.error] + [w.value for w in at.warning] + \
            [s.value for s in at.success] + [i.value for i in at.info] + \
            [x.label for x in at.expander]
    return "\n".join(map(str, parts))


def test_without_database_asks_to_start_engine():
    at = run_app()
    assert "Start the engine" in all_text(at)
    assert not Path(config.DB_PATH).exists()          # the dashboard did not create it


def seed_running_engine(now):
    db.set_engine_status({"state": "running", "stream_status": "CONNECTED:WS-STREAMING",
                          "connections_opened": 1}, now=now)
    db.upsert_feed_status([{
        "instrument": "EURUSD", "epic": config.INSTRUMENTS["EURUSD"]["epic"],
        "last_tick_utc": now - timedelta(seconds=3), "bid": 1.1194, "offer": 1.11949,
        "market_status": "DEAL", "market_status_utc": now, "ticks_today": 1234,
    }], now=now)
    db.upsert_market_info("EURUSD", {
        "epic": config.INSTRUMENTS["EURUSD"]["epic"], "name": "EUR/USD Mini", "pip_size": 0.0001,
        "one_pip_means": "0.0001 USD/EUR", "decimal_places_factor": 5,
        "currencies": [{"code": "USD", "is_default": True}],
        "min_deal_size": {"value": 0.1, "unit": "POINTS"},
    }, now=now)


def test_shows_status_prices_and_gaps_in_plain_words():
    now = timeutils.now_utc()
    seed_running_engine(now)
    db.add_closed_gap("GER40", "OUTAGE", now - timedelta(minutes=10), now - timedelta(minutes=9), "test gap")
    at = run_app()
    text = all_text(at)
    assert "All OK: recording 5 markets (1 open" in text
    assert "Trading: allowed" in text
    prices = at.dataframe[0].value
    eur = prices[prices["Market"] == "EUR/USD"].iloc[0]
    assert eur["Status"] == "Open"                          # not IG's code "DEAL"
    assert eur["Bid"] == "1.11940"                          # IG's 5 decimals, trailing zero kept
    assert eur["Spread"] == "0.9 pips"
    assert "UTC" in eur["Last price (UTC / MUT)"] and "MUT" in eur["Last price (UTC / MUT)"]
    assert "Gaps in the last 24 h: 1" in text
    gaps = at.dataframe[1].value
    assert list(gaps["What happened"]) == ["Connection to IG lost"] and list(gaps["Market"]) == ["Germany 40"]


def test_engine_not_running_warning():
    db.init_db()
    at = run_app()
    assert "Not recording: the engine is not running" in all_text(at)


def test_kill_button_only_writes_a_job():
    db.init_db()
    at = run_app()
    button = next(b for b in at.button if b.label == "Stop trading (kill switch ON)")
    button.click().run()
    jobs = db.list_jobs()
    assert [(j["kind"], j["requested_by"]) for j in jobs] == [(db.JOB_KILL_SWITCH_ON, "dashboard")]
    assert not db.get_flags()[db.KILL_SWITCH]["is_on"]     # the engine sets the flag, not the dashboard
    assert db.trading_block_reasons()                       # but trading is already blocked


def test_kill_off_needs_reason_and_confirmation():
    db.init_db()
    at = run_app()
    next(b for b in at.button if b.label == "Resume trading (kill switch OFF)").click().run()
    assert db.list_jobs() == []
    assert "Give a reason" in all_text(at)
