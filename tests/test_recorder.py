"""
Tests for price recording (fetch_data.py), the market-status subscription
(broker.PriceStream) and the engine's job/halt handling (signals.py).
Temporary folders and database only: your real data/ is never touched.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

import broker
import config
import db
import fetch_data
import signals
from tests.test_stream import FakeBroker, FakeClient, FakeSub, FakeUpdate, Clock

UTC = timezone.utc
EUR = config.INSTRUMENTS["EURUSD"]["epic"]
DAX = config.INSTRUMENTS["GER40"]["epic"]
T = datetime(2026, 11, 9, 10, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "TICKS_DIR", tmp_path / "data" / "ticks")
    monkeypatch.setattr(config, "BARS_1M_DIR", tmp_path / "data" / "bars_1m")
    yield tmp_path


def tick(epic, t, bid, offer, snapshot=False, received=None):
    return broker.Tick(epic=epic, time_utc=t, bid=bid, offer=offer,
                       received_utc=received or t + timedelta(milliseconds=150), is_snapshot=snapshot)


def ticks_df(rows):
    return pd.DataFrame(
        [(t, b, o, t, False) for t, b, o in rows], columns=fetch_data.TICK_COLUMNS)


def new_recorder(now=T):
    return fetch_data.Recorder(clock=lambda: now)


# --------------------------------------------------------------------------
# Bars: correct values, labelled by start, never an unfinished bar
# --------------------------------------------------------------------------
def test_bars_ohlc_spread_and_count():
    df = ticks_df([
        (T + timedelta(seconds=1), 1.10000, 1.10010),
        (T + timedelta(seconds=20), 1.10050, 1.10058),
        (T + timedelta(seconds=40), 1.09990, 1.10000),
        (T + timedelta(seconds=59), 1.10020, 1.10030),
        (T + timedelta(minutes=1, seconds=5), 1.10100, 1.10110),
    ])
    bars = fetch_data.ticks_to_bars(df)
    b = bars.iloc[0]
    assert b["time_utc"] == pd.Timestamp(T)                         # labelled by bar START
    assert (b["bid_open"], b["bid_high"], b["bid_low"], b["bid_close"]) == (1.10000, 1.10050, 1.09990, 1.10020)
    assert (b["offer_open"], b["offer_high"], b["offer_low"], b["offer_close"]) == (1.10010, 1.10058, 1.10000, 1.10030)
    assert b["tick_count"] == 4
    assert b["spread_max"] == pytest.approx(0.00010)
    assert b["spread_mean"] == pytest.approx((0.0001 + 0.00008 + 0.0001 + 0.0001) / 4)
    assert len(bars) == 2


@pytest.mark.parametrize("now_offset_s,expected_bars", [
    (59, 0),                                       # 10:00 bar still forming
    (60, 0),                                       # just ended, inside the grace period
    (60 + config.BAR_CLOSE_GRACE_SECONDS, 1),      # complete -> written
])
def test_unfinished_bar_never_returned(now_offset_s, expected_bars):
    df = ticks_df([(T + timedelta(seconds=5), 1.1, 1.1001), (T + timedelta(seconds=50), 1.2, 1.2001)])
    bars = fetch_data.ticks_to_bars(df, now=T + timedelta(seconds=now_offset_s))
    assert len(bars) == expected_bars


def test_minute_without_prices_gets_no_bar():
    df = ticks_df([(T, 1.1, 1.1001), (T + timedelta(minutes=3), 1.2, 1.2001)])
    bars = fetch_data.ticks_to_bars(df)
    assert list(bars["time_utc"]) == [pd.Timestamp(T), pd.Timestamp(T + timedelta(minutes=3))]


def test_bar_end():
    assert fetch_data.bar_end_utc(T) == T + timedelta(minutes=1)
    assert fetch_data.bar_end_utc(T, 5) == T + timedelta(minutes=5)


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
def test_ticks_and_complete_bars_written_with_utc_times():
    rec = new_recorder()
    rec.start()
    rec.on_tick(tick(EUR, T + timedelta(seconds=1), 1.1, 1.1001, snapshot=True))
    rec.on_tick(tick(EUR, T + timedelta(seconds=61), 1.2, 1.2001))
    rec.maintenance(now=T + timedelta(seconds=60 + config.BAR_CLOSE_GRACE_SECONDS, milliseconds=500))
    ticks = fetch_data.read_ticks("EURUSD", T.date())
    assert list(ticks.columns) == fetch_data.TICK_COLUMNS
    assert len(ticks) == 2 and bool(ticks["is_snapshot"].iloc[0])
    assert str(ticks["time_utc"].dt.tz) == "UTC" and str(ticks["received_utc"].dt.tz) == "UTC"
    bars = fetch_data.read_bars("EURUSD", T.date())
    assert list(bars["time_utc"]) == [pd.Timestamp(T)]                # 10:01 bar not complete yet
    rec.maintenance(now=T + timedelta(minutes=2, seconds=5))
    assert len(fetch_data.read_bars("EURUSD", T.date())) == 2


def test_files_split_at_utc_midnight():
    rec = new_recorder()
    rec.start()
    rec.on_tick(tick(EUR, datetime(2026, 11, 9, 23, 59, 59, tzinfo=UTC), 1.1, 1.1001))
    rec.on_tick(tick(EUR, datetime(2026, 11, 10, 0, 0, 1, tzinfo=UTC), 1.2, 1.2001))
    rec.maintenance(now=datetime(2026, 11, 10, 0, 0, 2, tzinfo=UTC))
    assert len(fetch_data.read_ticks("EURUSD", date(2026, 11, 9))) == 1
    assert len(fetch_data.read_ticks("EURUSD", date(2026, 11, 10))) == 1


def test_snapshot_duplicate_dropped_but_new_snapshot_kept():
    rec = new_recorder()
    rec.start()
    t1 = T + timedelta(seconds=10)
    rec.on_tick(tick(EUR, t1, 1.1, 1.1001))
    rec.on_tick(tick(EUR, t1, 1.1, 1.1001, snapshot=True))           # re-sent after reconnection
    rec.on_tick(tick(EUR, t1 + timedelta(seconds=1), 1.1, 1.1002, snapshot=True))   # a new price
    rec.maintenance(now=T + timedelta(seconds=30))
    assert len(fetch_data.read_ticks("EURUSD", T.date())) == 2
    assert rec.duplicates_dropped == 1


def test_restart_continues_same_file_and_drops_resent_snapshot():
    rec = new_recorder()
    rec.start()
    t1 = T + timedelta(seconds=10)
    rec.on_tick(tick(EUR, t1, 1.1, 1.1001))
    rec.close()
    rec2 = new_recorder(now=T + timedelta(minutes=5))
    rec2.start()
    rec2.on_tick(tick(EUR, t1, 1.1, 1.1001, snapshot=True))          # IG's snapshot on the new connection
    rec2.on_tick(tick(EUR, T + timedelta(minutes=5), 1.2, 1.2001))
    rec2.maintenance(now=T + timedelta(minutes=5, seconds=1))
    assert len(fetch_data.read_ticks("EURUSD", T.date())) == 2
    assert rec2.duplicates_dropped == 1


def test_no_temp_files_left_behind(tmp_storage):
    rec = new_recorder()
    rec.start()
    rec.on_tick(tick(EUR, T, 1.1, 1.1001))
    rec.maintenance(now=T + timedelta(seconds=5))
    assert not list(tmp_storage.rglob("*.tmp"))


# --------------------------------------------------------------------------
# Gaps
# --------------------------------------------------------------------------
def test_outage_gap_for_every_instrument():
    rec = new_recorder()
    rec.start()
    rec.set_streaming(True, T)
    rec.set_streaming(False, T + timedelta(seconds=30))
    rec.maintenance(now=T + timedelta(seconds=40))
    open_gaps = db.list_gaps(open_only=True)
    assert {g["instrument"] for g in open_gaps} == set(config.INSTRUMENTS)
    assert all(g["kind"] == "OUTAGE" and g["start_utc"] == T + timedelta(seconds=30) for g in open_gaps)
    rec.set_streaming(True, T + timedelta(seconds=36))
    rec.maintenance(now=T + timedelta(seconds=60))
    assert db.list_gaps(open_only=True) == []
    assert all(g["end_utc"] == T + timedelta(seconds=36) for g in db.list_gaps())


def test_quiet_gap_opens_and_closes():
    rec = new_recorder()
    rec.start()
    rec.set_streaming(True, T)
    rec.on_market_status(EUR, "DEAL", T)
    rec.on_tick(tick(EUR, T + timedelta(seconds=10), 1.1, 1.1001))
    rec.maintenance(now=T + timedelta(minutes=5))                    # 4 min 50 s: not yet
    assert db.list_gaps(open_only=True) == []
    rec.maintenance(now=T + timedelta(minutes=5, seconds=11))       # over 5 min
    gaps = db.list_gaps(open_only=True)
    assert [(g["instrument"], g["kind"]) for g in gaps] == [("EURUSD", "QUIET")]
    assert gaps[0]["start_utc"] == T + timedelta(seconds=10)
    rec.on_tick(tick(EUR, T + timedelta(minutes=7), 1.1, 1.1001))
    rec.maintenance(now=T + timedelta(minutes=7, seconds=5))
    closed = db.list_gaps()[0]
    assert closed["end_utc"] == T + timedelta(minutes=7)


def test_no_quiet_gap_when_market_closed_or_unknown():
    rec = new_recorder()
    rec.start()
    rec.set_streaming(True, T)
    rec.on_market_status(EUR, "CLOSED", T)                           # GER40 status unknown
    rec.maintenance(now=T + timedelta(hours=2))
    assert db.list_gaps() == []


def test_quiet_threshold_is_per_instrument(monkeypatch):
    monkeypatch.setitem(config.QUIET_GAP_MINUTES, "GER40", 10)
    rec = new_recorder()
    rec.start()
    rec.set_streaming(True, T)
    rec.on_market_status(EUR, "DEAL", T)
    rec.on_market_status(DAX, "DEAL", T)
    rec.maintenance(now=T + timedelta(minutes=6))
    assert [g["instrument"] for g in db.list_gaps(open_only=True)] == ["EURUSD"]
    rec.maintenance(now=T + timedelta(minutes=11))
    assert {g["instrument"] for g in db.list_gaps(open_only=True)} == {"EURUSD", "GER40"}


def test_market_close_ends_quiet_gap():
    rec = new_recorder()
    rec.start()
    rec.set_streaming(True, T)
    rec.on_market_status(EUR, "DEAL", T)
    rec.maintenance(now=T + timedelta(minutes=6))
    rec.on_market_status(EUR, "CLOSED", T + timedelta(minutes=8))
    rec.maintenance(now=T + timedelta(minutes=9))
    gap = db.list_gaps()[0]
    assert gap["end_utc"] == T + timedelta(minutes=8) and "CLOSED" in gap["detail"]
    assert [m["status"] for m in db.list_market_status_log()] == ["CLOSED", "DEAL"]


def test_quiet_time_restarts_after_reconnection():
    rec = new_recorder()
    rec.start()
    rec.set_streaming(True, T)
    rec.on_market_status(EUR, "DEAL", T)
    rec.set_streaming(False, T + timedelta(minutes=1))
    rec.set_streaming(True, T + timedelta(minutes=10))               # long outage (logged as OUTAGE)
    rec.maintenance(now=T + timedelta(minutes=11))
    assert [g["kind"] for g in db.list_gaps()] == ["OUTAGE"] * len(config.INSTRUMENTS)


def test_engine_off_gap_on_restart():
    rec = new_recorder(now=T + timedelta(hours=1))
    rec.start(last_heartbeat=T)
    gaps = db.list_gaps()
    assert {g["kind"] for g in gaps} == {"ENGINE_OFF"} and len(gaps) == len(config.INSTRUMENTS)
    assert gaps[0]["start_utc"] == T and gaps[0]["end_utc"] == T + timedelta(hours=1)


def test_feed_status_and_close():
    rec = new_recorder()
    rec.start()
    rec.set_streaming(True, T)
    rec.on_market_status(EUR, "DEAL", T)
    rec.on_tick(tick(EUR, T + timedelta(seconds=3), 1.1, 1.1001))
    rec.set_streaming(False, T + timedelta(seconds=4))
    rec.maintenance(now=T + timedelta(seconds=5))
    eur = {f["instrument"]: f for f in db.get_feed_status()}["EURUSD"]
    assert eur["last_tick_utc"] == T + timedelta(seconds=3) and eur["market_status"] == "DEAL"
    assert eur["ticks_today"] == 1
    rec.close()
    assert db.list_gaps(open_only=True) == []                        # nothing left open


# --------------------------------------------------------------------------
# Market status subscription (broker.PriceStream)
# --------------------------------------------------------------------------
def make_stream(statuses, clients):
    def client_factory(endpoint):
        c = FakeClient(endpoint)
        clients.append(c)
        return c

    def status_sub(account, epic):
        s = FakeSub(epic)
        s.item = f"PRICE:{account}:{epic}"
        s.is_status = True
        return s

    return broker.PriceStream(
        FakeBroker(), [EUR], on_tick=lambda t: None,
        on_market_status=lambda e, s, w: statuses.append((e, s, w)),
        client_factory=client_factory, subscription_factory=FakeSub,
        status_subscription_factory=status_sub, clock=Clock())


def push_status(client, flag, ts_ms="1762682400000"):
    for sub in client.subs:
        if getattr(sub, "is_status", False):
            for listener in sub.listeners:
                listener.onItemUpdate(FakeUpdate({"DLG_FLAG": flag, "TIMESTAMP": ts_ms}))


def test_market_status_subscription_and_changes():
    statuses, clients = [], []
    stream = make_stream(statuses, clients)
    stream.start(watchdog=False)
    items = [getattr(s, "item", None) for s in clients[0].subs]
    assert f"PRICE:ACC1:{EUR}" in items
    push_status(clients[0], "DEAL ")
    push_status(clients[0], "DEAL")                  # unchanged: not reported again
    push_status(clients[0], "CLOSED ")
    assert [s for _, s, _ in statuses] == ["DEAL", "CLOSED"]
    assert statuses[0][2] == datetime(2025, 11, 9, 10, 0, tzinfo=UTC)


def test_market_status_refused_does_not_stop_prices():
    statuses, clients = [], []
    stream = make_stream(statuses, clients)
    stream.start(watchdog=False)
    for sub in clients[0].subs:
        if getattr(sub, "is_status", False):
            for listener in sub.listeners:
                listener.onSubscriptionError(21, "not allowed")
    assert "refused" in stream.market_status_error
    assert stream.halt_reason() is None


def test_real_market_status_subscription_object():
    sub = broker.make_market_status_subscription("ACC1", EUR)
    assert sub.getItems() == [f"PRICE:ACC1:{EUR}"]
    assert sub.getFields() == ["DLG_FLAG", "TIMESTAMP"]
    assert sub.getDataAdapter() == "Pricing" and sub.getMode() == "MERGE"


# --------------------------------------------------------------------------
# Engine: jobs from the dashboard, automatic halt, heartbeat
# --------------------------------------------------------------------------
def test_engine_processes_kill_and_halt_jobs():
    engine = signals.Engine()
    kill = db.request_job(db.JOB_KILL_SWITCH_ON, {"reason": "panic button"}, "dashboard")
    other = db.request_job(db.JOB_RETRAIN, requested_by="dashboard")
    engine._process_jobs()
    flag = db.get_flags()[db.KILL_SWITCH]
    assert flag["is_on"] and flag["source"] == "dashboard" and flag["reason"] == "panic button"
    assert db.get_job(kill)["status"] == "DONE"
    assert db.get_job(other)["status"] == "FAILED" and "not available yet" in db.get_job(other)["message"]
    db.request_job(db.JOB_KILL_SWITCH_OFF, {"reason": "ok again"}, "dashboard")
    engine._process_jobs()
    assert not db.get_flags()[db.KILL_SWITCH]["is_on"]


class _Reason:
    def __init__(self, reason):
        self.reason = reason

    def halt_reason(self):
        return self.reason


def test_engine_sets_halt_once():
    engine = signals.Engine()
    engine.stream = _Reason("5 stream reconnection attempts failed in a row")
    engine.broker = _Reason(None)
    engine._check_halt()
    engine._check_halt()
    halt = db.get_flags()[db.HALT]
    assert halt["is_on"] and halt["source"] == "engine" and halt["expires_utc"] is None
    assert len([e for e in db.list_flag_events() if e["flag"] == "HALT"]) == 1


def test_engine_heartbeat():
    now = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
    assert not db.engine_is_running(now=now)
    db.set_engine_status({"state": "running"}, now=now)
    assert db.engine_is_running(now=now + timedelta(seconds=60))
    assert not db.engine_is_running(now=now + timedelta(seconds=config.ENGINE_HEARTBEAT_STALE_SECONDS + 1))
    db.set_engine_status({"state": "stopped"}, now=now)
    assert not db.engine_is_running(now=now)


# --------------------------------------------------------------------------
# Partial bars (engine stopped / started / stream down during the minute)
# --------------------------------------------------------------------------
def test_partial_bars_marked():
    df = ticks_df([(T + timedelta(minutes=m, seconds=10), 1.1, 1.1001) for m in range(6)])
    engine_off = (T + timedelta(minutes=1, seconds=56), T + timedelta(minutes=3, seconds=53))
    bars = fetch_data.ticks_to_bars(df, not_receiving=[engine_off])
    assert list(bars["partial"]) == [False, True, True, True, False, False]


def test_bar_containing_session_start_is_partial():
    df = ticks_df([(T + timedelta(seconds=55), 1.1, 1.1001), (T + timedelta(minutes=1, seconds=5), 1.1, 1.1001)])
    bars = fetch_data.ticks_to_bars(df, session_start=T + timedelta(seconds=53))
    assert list(bars["partial"]) == [True, False]


def test_open_outage_marks_bars_partial():
    df = ticks_df([(T + timedelta(seconds=5), 1.1, 1.1001), (T + timedelta(minutes=2, seconds=5), 1.1, 1.1001)])
    bars = fetch_data.ticks_to_bars(df, not_receiving=[(T + timedelta(minutes=1, seconds=30), None)])
    assert list(bars["partial"]) == [False, True]


def test_recorder_marks_bars_around_restart_but_not_quiet_gaps():
    rec = new_recorder(now=T + timedelta(minutes=6, seconds=53))
    db.add_closed_gap("EURUSD", "ENGINE_OFF", T + timedelta(minutes=1, seconds=56),
                      T + timedelta(minutes=6, seconds=53))
    db.add_closed_gap("EURUSD", "QUIET", T + timedelta(minutes=8), T + timedelta(minutes=9))
    rec.start()
    for m in (0, 1, 6, 7, 8):
        rec.on_tick(tick(EUR, T + timedelta(minutes=m, seconds=55), 1.1, 1.1001))
    rec.maintenance(now=T + timedelta(minutes=10))
    bars = fetch_data.read_bars("EURUSD", T.date())
    got = dict(zip([t.minute for t in bars["time_utc"]], bars["partial"]))
    assert got == {0: False, 1: True, 6: True, 7: False, 8: False}
