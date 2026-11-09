"""
Tests for db.py. Each test gets its own fresh SQLite file in a temp folder,
so your real data/trends_analyst.db is never touched.
"""

import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

import config
import db
import timeutils

UTC = timezone.utc
T0 = datetime(2026, 11, 9, 11, 40, 42, 909000, tzinfo=UTC)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    yield config.DB_PATH


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
def test_init_creates_file_tables_and_wal(fresh_db):
    db.init_db()
    assert fresh_db.exists()
    assert db.schema_version() == len(db.MIGRATIONS)
    with closing(sqlite3.connect(fresh_db)) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"flags", "flag_events", "jobs", "variants", "variant_role_log", "predictions"} <= tables
    assert mode == "wal"


def test_init_is_repeatable(fresh_db):
    db.init_db()
    db.set_kill_switch("test", "cli")
    db.init_db()                                   # must not wipe or fail
    assert db.get_flags()[db.KILL_SWITCH]["is_on"]


# --------------------------------------------------------------------------
# Kill switch
# --------------------------------------------------------------------------
def test_kill_switch_on_off_logged_with_time_reason_source():
    assert db.trading_block_reasons(now=T0) == []
    db.set_kill_switch("market too volatile", "cli", now=T0)
    flags = db.get_flags()
    assert flags[db.KILL_SWITCH]["is_on"]
    assert flags[db.KILL_SWITCH]["changed_utc"] == T0
    assert any("kill switch ON" in r for r in db.trading_block_reasons(now=T0))

    later = T0 + timedelta(hours=3)
    db.clear_kill_switch("checked, resuming", "cli", now=later)
    assert db.trading_block_reasons(now=later) == []

    events = db.list_flag_events()
    assert [(e["flag"], e["action"], e["source"], e["reason"]) for e in events] == [
        ("KILL_SWITCH", "OFF", "cli", "checked, resuming"),
        ("KILL_SWITCH", "ON", "cli", "market too volatile"),
    ]
    assert events[1]["time_utc"] == T0


def test_kill_switch_stays_on_until_cleared():
    db.set_kill_switch("stop", "cli", now=T0)
    for days in (1, 7, 30):
        assert db.trading_block_reasons(now=T0 + timedelta(days=days))
    assert db.expire_halt(now=T0 + timedelta(days=30)) is False   # expiry never touches the kill switch
    assert db.get_flags()[db.KILL_SWITCH]["is_on"]


def test_reason_and_valid_source_required():
    with pytest.raises(db.DBError):
        db.set_kill_switch("   ", "cli")
    with pytest.raises(db.DBError):
        db.set_kill_switch("x", "someone")


def test_pending_dashboard_kill_request_blocks_before_engine_acts():
    db.request_job(db.JOB_KILL_SWITCH_ON, {"reason": "panic button"}, "dashboard", now=T0)
    assert not db.get_flags()[db.KILL_SWITCH]["is_on"]          # flag not set yet...
    assert any("requested from the dashboard" in r for r in db.trading_block_reasons(now=T0))  # ...but blocked


# --------------------------------------------------------------------------
# Halt
# --------------------------------------------------------------------------
def test_halt_without_expiry_needs_manual_clear():
    db.set_halt("5 stream reconnection attempts failed in a row", now=T0)
    assert db.expire_halt(now=T0 + timedelta(days=3)) is False
    assert any(r.startswith("HALT") for r in db.trading_block_reasons(now=T0 + timedelta(days=3)))
    db.clear_halt("network fixed", "cli", now=T0 + timedelta(days=3))
    assert db.trading_block_reasons(now=T0 + timedelta(days=3)) == []


def test_daily_loss_halt_ends_at_next_trading_day():
    reset = timeutils.next_trading_day_start_utc(T0)           # 22:00 UTC (17:00 New York)
    assert reset == datetime(2026, 11, 9, 22, 0, tzinfo=UTC)
    db.set_halt("daily loss limit 3% reached", expires_utc=reset, now=T0)
    assert db.trading_block_reasons(now=reset - timedelta(seconds=1))
    assert db.trading_block_reasons(now=reset) == []            # no longer blocks
    assert db.expire_halt(now=reset) is True                    # and is cleared + logged
    assert not db.get_flags()[db.HALT]["is_on"]
    last = db.list_flag_events(1)[0]
    assert (last["flag"], last["action"], last["source"]) == ("HALT", "OFF", "engine")


def test_kill_and_halt_reported_together():
    db.set_kill_switch("manual", "cli", now=T0)
    db.set_halt("10 disconnections in 60 minutes", now=T0)
    assert len(db.trading_block_reasons(now=T0)) == 2


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
def test_job_lifecycle():
    job_id = db.request_job(db.JOB_RETRAIN, {"note": "weekly"}, "dashboard", now=T0)
    job = db.get_job(job_id)
    assert job["status"] == "PENDING" and job["params"] == {"note": "weekly"}
    claimed = db.claim_next_job(now=T0)
    assert claimed["id"] == job_id and claimed["status"] == "RUNNING"
    db.update_job_progress(job_id, 0.5, "Training... started 15:40 MUT")
    assert db.get_job(job_id)["progress"] == 0.5
    db.finish_job(job_id, ok=True, message="done", result={"auc": 0.55})
    job = db.get_job(job_id)
    assert job["status"] == "DONE" and job["progress"] == 1 and job["result"] == {"auc": 0.55}


def test_jobs_claimed_oldest_first_and_only_once():
    ids = [db.request_job(db.JOB_REFRESH_DATA, requested_by="dashboard") for _ in range(3)]
    assert db.claim_next_job()["id"] == ids[0]
    assert db.claim_next_job()["id"] == ids[1]
    assert db.claim_next_job()["id"] == ids[2]
    assert db.claim_next_job() is None


def test_claim_by_kind():
    db.request_job(db.JOB_RETRAIN, requested_by="dashboard")
    kill = db.request_job(db.JOB_KILL_SWITCH_ON, {"reason": "x"}, "dashboard")
    assert db.claim_next_job(kinds=(db.JOB_KILL_SWITCH_ON,))["id"] == kill


def test_concurrent_claims_never_share_a_job():
    for _ in range(20):
        db.request_job(db.JOB_REFRESH_DATA, requested_by="dashboard")
    got, lock = [], threading.Lock()

    def worker():
        while (job := db.claim_next_job()) is not None:
            with lock:
                got.append(job["id"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == sorted(set(got)) and len(got) == 20


def test_unknown_job_kind_rejected():
    with pytest.raises(db.DBError):
        db.request_job("CLOSE_ALL", requested_by="dashboard")   # not until Phase 3


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
def test_only_one_active_variant_and_role_changes_logged():
    db.add_variant("with_patterns", "base+candles")
    db.add_variant("no_patterns", "base")
    db.set_variant_role("with_patterns", "ACTIVE", "engine", "first model")
    db.set_variant_role("no_patterns", "ACTIVE", "dashboard", "won comparison")
    roles = {v["name"]: v["role"] for v in db.get_variants()}
    assert roles == {"no_patterns": "ACTIVE", "with_patterns": "SHADOW"}   # previous one kept as shadow
    assert db.get_active_variant()["name"] == "no_patterns"
    log = db.list_variant_role_log()
    assert (log[0]["variant"], log[0]["old_role"], log[0]["new_role"]) == ("no_patterns", "OFF", "ACTIVE")
    assert (log[1]["variant"], log[1]["new_role"]) == ("with_patterns", "SHADOW")


def test_database_itself_refuses_two_active(fresh_db):
    db.add_variant("a", "base")
    db.add_variant("b", "base")
    db.set_variant_role("a", "ACTIVE", "engine")
    with closing(sqlite3.connect(fresh_db)) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE variants SET role = 'ACTIVE' WHERE name = 'b'")


def test_variant_errors():
    db.add_variant("a", "base")
    with pytest.raises(db.DBError):
        db.add_variant("a", "base")
    with pytest.raises(db.DBError):
        db.set_variant_role("a", "LIVE", "engine")
    with pytest.raises(db.DBError):
        db.set_variant_role("missing", "SHADOW", "engine")


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
def make_pred(**overrides):
    pred = dict(
        variant="with_patterns", is_shadow=False, instrument="EURUSD",
        epic=config.INSTRUMENTS["EURUSD"]["epic"],
        bar_time_utc=datetime(2026, 11, 9, 8, 45, tzinfo=UTC), direction="BUY",
        entry=1.0850, stop_loss=1.0832, take_profit=1.0877, atr=0.0018,
        probability=0.52, expected_value_r=0.24,
        close_by_utc=datetime(2026, 11, 9, 16, 0, tzinfo=UTC),
        pattern="Hammer at previous-day low - supports BUY",
    )
    pred.update(overrides)
    return pred


def test_save_and_close_prediction_with_utc_times():
    pid = db.save_prediction(make_pred(), now=T0)
    shadow = db.save_prediction(make_pred(variant="no_patterns", is_shadow=True))
    assert [p["id"] for p in db.get_open_predictions()] == [pid, shadow]
    assert [p["id"] for p in db.get_open_predictions(include_shadow=False)] == [pid]
    p = db.get_open_predictions()[0]
    assert p["bar_time_utc"] == datetime(2026, 11, 9, 8, 45, tzinfo=UTC)
    assert p["created_utc"] == T0 and p["is_shadow"] is False

    db.close_prediction(pid, "TP", 1.0877, datetime(2026, 11, 9, 10, 10, tzinfo=UTC), 1.44)
    closed = db.list_predictions(variant="with_patterns")[0]
    assert (closed["status"], closed["outcome"], closed["result_r"]) == ("CLOSED", "TP", 1.44)
    with pytest.raises(db.DBError):
        db.close_prediction(pid, "SL", 1.0832, T0, -1.0)        # can't close twice


def test_prediction_validation():
    with pytest.raises(db.DBError):
        db.save_prediction(make_pred(probability=None))
    with pytest.raises(db.DBError):
        db.save_prediction(make_pred(instrument="BTCUSD"))
    with pytest.raises(db.DBError):
        db.save_prediction(make_pred(direction="LONG"))
    with pytest.raises(db.DBError):
        db.save_prediction(make_pred(probability=1.5))
    with pytest.raises(db.DBError):
        db.save_prediction(make_pred(colour="red"))
    with pytest.raises(ValueError):                              # naive time rejected
        db.save_prediction(make_pred(bar_time_utc=datetime(2026, 11, 9, 8, 45)))


def test_second_connection_sees_writes(fresh_db):
    """Engine and dashboard are separate programs: a plain second connection sees the data."""
    db.set_kill_switch("from engine", "engine", now=T0)
    with closing(sqlite3.connect(fresh_db)) as conn:
        assert conn.execute("SELECT is_on FROM flags WHERE name = 'KILL_SWITCH'").fetchone()[0] == 1


def test_market_info_from_ig_details():
    db.upsert_market_info("EURUSD", {
        "epic": "CS.D.EURUSD.MINI.IP", "name": "EUR/USD Mini", "pip_size": 0.0001,
        "one_pip_means": "0.0001 USD/EUR", "decimal_places_factor": 5, "value_of_one_pip": "1.00",
        "contract_size": 10000.0, "currencies": [{"code": "USD", "is_default": True}],
        "min_deal_size": {"value": 0.1, "unit": "POINTS"},
        "min_stop_or_limit_distance": {"value": 2.0, "unit": "POINTS"},
    }, now=T0)
    info = db.get_market_info()["EURUSD"]
    assert (info["pip_size"], info["decimals"], info["currency"]) == (0.0001, 5, "USD")
    assert (info["min_deal_size"], info["min_stop_distance"]) == (0.1, 2.0)
    assert info["updated_utc"] == T0
