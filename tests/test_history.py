"""
Tests for the Dukascopy history download (fetch_data.py, Phase 2 step 1).

No network: the library call is replaced by a fake that returns synthetic
ticks. Temporary folders and database only; your real data/ is never touched.
"""

import json
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version

import pandas as pd
import pytest
import requests

import config
import db
import fetch_data

UTC = timezone.utc


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "DUKASCOPY_TICKS_DIR", tmp_path / "data" / "dukascopy" / "ticks")
    monkeypatch.setattr(config, "DUKASCOPY_BARS_1M_DIR", tmp_path / "data" / "dukascopy" / "bars_1m")
    monkeypatch.setattr(config, "DUKASCOPY_MIN_SECONDS_BETWEEN_REQUESTS", 0.0)
    yield tmp_path


class FakeTick:
    """Same attributes as dukascopy_python's Tick."""

    def __init__(self, ts, bid, ask):
        self.timestamp, self.bid, self.ask = ts, bid, ask
        self.bid_volume, self.ask_volume, self.volume = 1.5, 2.0, 3.5


class FakeLibrary:
    """Stands in for the dukascopy_python module: records calls, returns synthetic ticks."""

    def __init__(self, fail_at=None, error=None, ticks_per_hour=3):
        self.calls = []
        self.fail_at = fail_at or set()      # hour starts that raise `error`
        self.error = error
        self.ticks_per_hour = ticks_per_hour

    def fetch_ticks(self, instrument, start, end, *, version=None, create_session=None, logger=None):
        self.calls.append((instrument, start, end, version))
        session = create_session()          # what the library does; checks our session works
        assert isinstance(session, fetch_data._PoliteSession)
        if start in self.fail_at:
            raise self.error
        out = []
        for i in range(self.ticks_per_hour):
            ts = start + timedelta(minutes=i * 10, seconds=5)
            bid = 1.08 + i * 0.0001 + 0.000000001     # float noise, as the library produces
            out.append(FakeTick(ts, bid, bid + 0.00008))
        return out


@pytest.fixture
def fake_lib(monkeypatch):
    lib = FakeLibrary()
    monkeypatch.setattr(fetch_data, "_library", lambda: lib)
    return lib


WED = date(2026, 3, 4)
SAT = date(2026, 3, 7)
SUN = date(2026, 3, 8)


# --------------------------------------------------------------------------
# Library version and instrument codes (real library, no network)
# --------------------------------------------------------------------------
def test_pinned_library_version_installed():
    assert version("dukascopy-python") == fetch_data.DUKASCOPY_LIBRARY_VERSION == "6.0.0"
    with open(config.PROJECT_ROOT / "requirements.txt") as f:
        assert "dukascopy-python==6.0.0" in f.read()


@pytest.mark.parametrize("inst,decimals", [("EURUSD", 5), ("GBPUSD", 5), ("USDJPY", 3),
                                           ("US500", 3), ("GER40", 3)])
def test_every_instrument_code_exists_in_library(inst, decimals):
    meta = fetch_data.instrument_meta(inst)
    assert meta["code"] == config.DUKASCOPY_SYMBOLS[inst]
    assert meta["priceScale"] == decimals


def test_symbols_cover_all_instruments():
    assert set(config.DUKASCOPY_SYMBOLS) == set(config.INSTRUMENTS)
    assert set(config.HISTORY_APPROVED_FOR_MODEL) == set(config.INSTRUMENTS)
    assert not config.HISTORY_APPROVED_FOR_MODEL["US500"]
    assert not config.HISTORY_APPROVED_FOR_MODEL["GER40"]


# --------------------------------------------------------------------------
# Calendar: weekend skip window, lag, days
# --------------------------------------------------------------------------
@pytest.mark.parametrize("when,skipped", [
    (datetime(2026, 3, 6, 23, tzinfo=UTC), False),   # Friday 23:00
    (datetime(2026, 3, 7, 0, tzinfo=UTC), True),     # Saturday 00:00
    (datetime(2026, 3, 7, 23, tzinfo=UTC), True),    # Saturday 23:00
    (datetime(2026, 3, 8, 19, tzinfo=UTC), True),    # Sunday 19:00
    (datetime(2026, 3, 8, 20, tzinfo=UTC), False),   # Sunday 20:00 (before the 21:00 summer open)
    (datetime(2026, 3, 9, 0, tzinfo=UTC), False),    # Monday
])
def test_skip_window(when, skipped):
    assert fetch_data.is_skipped_hour(when) is skipped


def test_skip_window_is_before_the_earliest_sunday_open():
    # Forex reopens at 17:00 New York = 21:00 UTC in northern summer at the earliest.
    assert config.DUKASCOPY_SKIP_FROM == (5, 0)
    assert config.DUKASCOPY_SKIP_UNTIL <= (6, 21)


def test_default_end_keeps_a_lag():
    now = datetime(2026, 11, 9, 15, 30, tzinfo=UTC)
    assert fetch_data.default_history_end(now) == datetime(2026, 11, 6, tzinfo=UTC)


def test_days_are_utc_days_end_excluded():
    assert fetch_data._days(datetime(2026, 3, 2, tzinfo=UTC), datetime(2026, 3, 5, tzinfo=UTC)) == \
        [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)]


# --------------------------------------------------------------------------
# Throttle and session
# --------------------------------------------------------------------------
def test_throttle_spaces_requests(monkeypatch):
    monkeypatch.setattr(config, "DUKASCOPY_MIN_SECONDS_BETWEEN_REQUESTS", 1.0)
    now = [100.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    th = fetch_data._Throttle(clock=lambda: now[0], sleep=sleep)
    th.wait()                      # first request: no wait
    now[0] += 0.25
    th.wait()                      # 0.25 s later: waits the remaining 0.75 s
    now[0] += 5
    th.wait()                      # long after: no wait
    assert slept == [pytest.approx(0.75)]


def test_session_has_timeout_retries_and_counts(monkeypatch):
    seen = {}

    def fake_send(self, request, **kwargs):
        seen.update(kwargs)
        r = requests.Response()
        r.status_code = 200
        r._content = b"{}"
        return r

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    counter = fetch_data._RequestCounter()
    s = fetch_data._PoliteSession(counter)
    s.get("https://jetta.dukascopy.com/x")
    assert seen["timeout"] == config.DUKASCOPY_REQUEST_TIMEOUT_SECONDS
    assert counter.count == 1
    retry = s.get_adapter("https://jetta.dukascopy.com").max_retries
    assert retry.total == config.DUKASCOPY_HTTP_RETRIES
    assert 429 in retry.status_forcelist and 403 not in retry.status_forcelist


# --------------------------------------------------------------------------
# One hour
# --------------------------------------------------------------------------
def test_fetch_hour_converts_and_rounds(fake_lib):
    h = datetime(2026, 3, 4, 10, tzinfo=UTC)
    df = fetch_data.fetch_dukascopy_hour("EURUSD", h)
    assert list(df.columns) == fetch_data.HISTORY_TICK_COLUMNS
    assert len(df) == 3
    assert str(df["time_utc"].dt.tz) == "UTC"
    assert df["bid"].iloc[0] == 1.08                       # float noise rounded to 5 decimals
    assert df["offer"].iloc[0] == pytest.approx(1.08008)
    assert (df["offer"] >= df["bid"]).all()
    instrument, start, end, feed = fake_lib.calls[0]
    assert (instrument, start, end, feed) == ("EUR-USD", h, h + timedelta(hours=1), config.DUKASCOPY_FEED_VERSION)


def test_fetch_hour_rejects_naive_time(fake_lib):
    with pytest.raises(ValueError):
        fetch_data.fetch_dukascopy_hour("EURUSD", datetime(2026, 3, 4, 10))


def test_ticks_outside_hour_are_refused():
    h = datetime(2026, 3, 4, 10, tzinfo=UTC)
    with pytest.raises(fetch_data.DukascopyError, match="outside"):
        fetch_data._ticks_to_frame([FakeTick(h + timedelta(hours=1), 1.08, 1.0801)], 5, h,
                                   h + timedelta(hours=1))


class _HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


@pytest.mark.parametrize("error,blocked,text", [
    (_HTTPError(403), True, "403"),
    (json.JSONDecodeError("Expecting value", "<html>", 0), True, "not price data"),
    (_HTTPError(404), False, "404"),
    (requests.exceptions.ConnectionError("down"), False, "network"),
    (requests.exceptions.RetryError("too many 429"), False, "after"),
    (KeyError("times"), False, "unexpected format"),
])
def test_errors_are_classified(error, blocked, text):
    e = fetch_data._classify_error(error)
    assert e.blocked is blocked
    assert text in str(e)


# --------------------------------------------------------------------------
# One day
# --------------------------------------------------------------------------
def test_download_day_writes_file_and_records_done(fake_lib):
    stats = fetch_data.download_day("EURUSD", WED)
    assert stats["hours_requested"] == 24 and stats["hours_skipped"] == 0
    assert stats["tick_count"] == 72
    ticks = fetch_data.read_history_ticks("EURUSD", WED)
    assert len(ticks) == 72 and ticks["time_utc"].is_monotonic_increasing
    rows = db.get_history_days("dukascopy", "EURUSD")
    assert [r["day"] for r in rows] == [WED]
    assert rows[0]["file_bytes"] == fetch_data.history_tick_file("EURUSD", WED).stat().st_size > 0


def test_weekend_hours_skipped(fake_lib):
    sat = fetch_data.download_day("EURUSD", SAT)
    sun = fetch_data.download_day("EURUSD", SUN)
    assert sat["hours_requested"] == 0 and sat["hours_skipped"] == 24
    assert not fetch_data.history_tick_file("EURUSD", SAT).exists()   # nothing to store
    assert sun["hours_requested"] == 4 and sun["hours_skipped"] == 20   # 20:00-23:00
    assert {c[1].hour for c in fake_lib.calls} == {20, 21, 22, 23}
    assert len(db.get_history_days("dukascopy", "EURUSD")) == 2           # both still "done"


def test_no_skip_requests_every_hour(fake_lib):
    stats = fetch_data.download_day("EURUSD", SAT, skip_closed=False)
    assert stats["hours_requested"] == 24


def test_failed_hour_leaves_day_not_done(fake_lib):
    fake_lib.fail_at = {datetime(2026, 3, 4, 13, tzinfo=UTC)}
    fake_lib.error = _HTTPError(500)
    with pytest.raises(fetch_data.DukascopyError, match="EURUSD 2026-03-04 13:00 UTC"):
        fetch_data.download_day("EURUSD", WED)
    assert db.get_history_days("dukascopy", "EURUSD") == []
    assert not fetch_data.history_tick_file("EURUSD", WED).exists()


# --------------------------------------------------------------------------
# Many days: resume, stop rules, monthly bars
# --------------------------------------------------------------------------
def test_download_is_resumable(fake_lib, monkeypatch):
    monkeypatch.setattr(fetch_data, "default_history_end", lambda now=None: datetime(2026, 4, 1, tzinfo=UTC))
    start, end = datetime(2026, 3, 2, tzinfo=UTC), datetime(2026, 3, 5, tzinfo=UTC)
    first = fetch_data.download_history(["EURUSD"], start, end)
    calls = len(fake_lib.calls)
    assert first["instruments"]["EURUSD"]["downloaded"] == 3 and calls == 72
    second = fetch_data.download_history(["EURUSD"], start, end)
    assert second["instruments"]["EURUSD"] == {"already_done": 3, "downloaded": 0, "failed": 0,
                                               "ticks": 0, "bars_by_month": {}}
    assert len(fake_lib.calls) == calls                     # nothing downloaded again


def test_download_never_goes_past_the_lag(fake_lib, monkeypatch):
    monkeypatch.setattr(fetch_data, "default_history_end", lambda now=None: datetime(2026, 3, 3, tzinfo=UTC))
    fetch_data.download_history(["EURUSD"], datetime(2026, 3, 2, tzinfo=UTC), datetime(2026, 3, 10, tzinfo=UTC))
    assert [r["day"] for r in db.get_history_days("dukascopy", "EURUSD")] == [date(2026, 3, 2)]


def test_blocked_stops_immediately(fake_lib, monkeypatch):
    monkeypatch.setattr(fetch_data, "default_history_end", lambda now=None: datetime(2026, 4, 1, tzinfo=UTC))
    fake_lib.fail_at = {datetime(2026, 3, 3, h, tzinfo=UTC) for h in range(24)}
    fake_lib.error = _HTTPError(403)
    res = fetch_data.download_history(["EURUSD", "GBPUSD"], datetime(2026, 3, 2, tzinfo=UTC),
                                      datetime(2026, 3, 6, tzinfo=UTC))
    assert "refused" in res["stopped"]
    assert [r["day"] for r in db.get_history_days("dukascopy")] == [date(2026, 3, 2)]
    assert "GBPUSD" not in res["instruments"]               # stopped before the next instrument
    # The day finished before the block still has its bars.
    assert fetch_data.history_bar_file("EURUSD", 2026, 3).exists()


def test_stops_after_failed_days_in_a_row(fake_lib, monkeypatch):
    monkeypatch.setattr(fetch_data, "default_history_end", lambda now=None: datetime(2026, 4, 1, tzinfo=UTC))
    fake_lib.fail_at = {datetime(2026, 3, d, 0, tzinfo=UTC) for d in range(2, 20)}
    fake_lib.error = _HTTPError(500)
    res = fetch_data.download_history(["EURUSD"], datetime(2026, 3, 2, tzinfo=UTC), datetime(2026, 3, 20, tzinfo=UTC))
    assert "3 days in a row" in res["stopped"]
    assert res["instruments"]["EURUSD"]["failed"] == config.DUKASCOPY_STOP_AFTER_FAILED_DAYS


def test_month_bars_match_ig_bar_format(fake_lib, monkeypatch):
    monkeypatch.setattr(fetch_data, "default_history_end", lambda now=None: datetime(2026, 5, 1, tzinfo=UTC))
    res = fetch_data.download_history(["EURUSD"], datetime(2026, 3, 30, tzinfo=UTC), datetime(2026, 4, 2, tzinfo=UTC))
    # 30 and 31 March, 1 April; 24 hours x 3 ticks in different minutes -> 72 bars per day.
    assert res["instruments"]["EURUSD"]["bars_by_month"] == {"2026-03": 144, "2026-04": 72}
    march = pd.read_parquet(fetch_data.history_bar_file("EURUSD", 2026, 3))
    april = pd.read_parquet(fetch_data.history_bar_file("EURUSD", 2026, 4))
    assert list(march.columns) == fetch_data.BAR_COLUMNS
    assert not march["partial"].any()
    assert len(march) == 144 and len(april) == 72
    # Bars are labelled by START: the tick at 10:00:05 is in the 10:00 bar.
    t = pd.to_datetime(march["time_utc"], utc=True)
    assert t.iloc[0] == pd.Timestamp("2026-03-30 00:00", tz="UTC")
    assert march["spread_mean"].iloc[0] == pytest.approx(0.00008)
    assert (t.dt.month == 3).all()


def test_month_bars_refuse_when_a_tick_file_is_missing(fake_lib):
    fetch_data.download_day("EURUSD", WED)
    fetch_data.history_tick_file("EURUSD", WED).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        fetch_data.build_month_bars("EURUSD", 2026, 3)


# --------------------------------------------------------------------------
# Final-check lock and approval
# --------------------------------------------------------------------------
def test_final_check_dates_are_fixed_and_sensible():
    assert config.FINAL_CHECK_START_UTC == datetime(2026, 5, 1, tzinfo=UTC)
    assert config.FINAL_CHECK_END_UTC == datetime(2026, 11, 1, tzinfo=UTC)
    assert config.HISTORY_START_UTC < config.PILOT_START_UTC < config.PILOT_END_UTC \
        <= fetch_data.development_end_utc() < config.FINAL_CHECK_START_UTC
    assert fetch_data.development_end_utc() == datetime(2026, 4, 30, tzinfo=UTC)


def _d(*a):
    return datetime(*a, tzinfo=UTC)


@pytest.mark.parametrize("inst,start,end,purpose,ok", [
    ("EURUSD", _d(2019, 1, 1), _d(2026, 4, 30), "development", True),
    ("EURUSD", _d(2019, 1, 1), _d(2026, 4, 30, 0, 1), "development", False),   # into the gap day
    ("EURUSD", _d(2026, 6, 1), _d(2026, 7, 1), "development", False),          # inside final check
    ("EURUSD", _d(2026, 11, 2), _d(2026, 11, 5), "development", False),        # after it (future data)
    ("US500", _d(2020, 1, 1), _d(2020, 2, 1), "development", False),           # not approved yet
    ("US500", _d(2026, 11, 2), _d(2026, 11, 5), "source_comparison", True),
    ("GER40", _d(2026, 4, 1), _d(2026, 4, 29), "source_comparison", True),
    ("GER40", _d(2026, 4, 1), _d(2026, 5, 2), "source_comparison", False),     # touches final check
    ("EURUSD", _d(2026, 5, 1), _d(2026, 11, 1), "final_evaluation", True),
    ("EURUSD", _d(2026, 4, 1), _d(2026, 11, 1), "final_evaluation", False),    # starts too early
    ("US500", _d(2026, 5, 1), _d(2026, 11, 1), "final_evaluation", False),     # not approved
    ("EURUSD", _d(2020, 1, 1), _d(2020, 2, 1), "training", False),             # unknown purpose
])
def test_history_access_rules(inst, start, end, purpose, ok):
    if ok:
        fetch_data.check_history_access(inst, start, end, purpose)
    else:
        with pytest.raises(fetch_data.HistoryAccessError):
            fetch_data.check_history_access(inst, start, end, purpose)


def test_final_evaluation_needs_reason_and_is_logged(fake_lib):
    with pytest.raises(fetch_data.HistoryAccessError, match="reason"):
        fetch_data.load_history_1m("EURUSD", _d(2026, 5, 1), _d(2026, 11, 1), "final_evaluation")
    assert db.list_final_check_access() == []
    fetch_data.load_history_1m("EURUSD", _d(2026, 5, 1), _d(2026, 11, 1), "final_evaluation",
                               reason="final check of variant A")
    log = db.list_final_check_access()
    assert len(log) == 1 and log[0]["instruments"] == "EURUSD"
    assert log[0]["reason"] == "final check of variant A"


def test_load_history_returns_only_the_requested_range(fake_lib, monkeypatch):
    monkeypatch.setattr(fetch_data, "default_history_end", lambda now=None: datetime(2026, 5, 1, tzinfo=UTC))
    fetch_data.download_history(["EURUSD"], _d(2026, 3, 31), _d(2026, 4, 2))
    bars = fetch_data.load_history_1m("EURUSD", _d(2026, 3, 31, 12), _d(2026, 4, 1, 12), "development")
    t = pd.to_datetime(bars["time_utc"], utc=True)
    assert t.min() >= pd.Timestamp("2026-03-31 12:00", tz="UTC")
    assert t.max() < pd.Timestamp("2026-04-01 12:00", tz="UTC")
    assert len(bars) == 72                                  # 24 hours x 3 bars
    assert len(fetch_data.load_history_1m("EURUSD", _d(2019, 1, 1), _d(2019, 2, 1), "development")) == 0


# --------------------------------------------------------------------------
# Pilot report and status
# --------------------------------------------------------------------------
def test_pilot_report_and_status(fake_lib, monkeypatch):
    monkeypatch.setattr(fetch_data, "default_history_end", lambda now=None: datetime(2026, 11, 6, tzinfo=UTC))
    fetch_data.download_history(["EURUSD"], config.PILOT_START_UTC, config.PILOT_END_UTC, skip_closed=False)
    text = "\n".join(fetch_data.pilot_report(["EURUSD"], config.PILOT_START_UTC, config.PILOT_END_UTC))
    assert "7 days" in text
    # The fake returns ticks in every hour, including the weekend: the report must say so.
    assert "the skip rule would lose these" in text
    assert "estimate 2019-01-01 to 2026-11-06" in text
    status = "\n".join(fetch_data.history_status())
    assert "EURUSD" in status and "NOT approved" in status and "read 0 time(s)" in status


def test_cli_status_runs(capsys, monkeypatch):
    monkeypatch.setattr(fetch_data.timeutils, "configure_logging", lambda **kw: None)
    assert fetch_data.main(["status"]) == 0
    assert "Final-check period: 2026-05-01 to 2026-11-01" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The real library, with a fake server (HTTP is answered locally)
# --------------------------------------------------------------------------
# The reply format below is copied from what dukascopy-python 6.0.0's feed-2
# parser reads (impl/v2 _parse_ticks), not from any Dukascopy documentation.
def _fake_server(monkeypatch, status=200, body=None):
    urls = []

    def send(self, request, **kwargs):
        urls.append(request.url)
        r = requests.Response()
        r.status_code = status
        r.url = request.url
        r._content = json.dumps(body).encode() if body is not None else b"<html>challenge</html>"
        return r

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return urls


def test_real_library_feed2_parsing_through_our_session(monkeypatch):
    t0 = int(datetime(2026, 3, 4, 10, 0, 1, tzinfo=UTC).timestamp() * 1000)
    body = {"timestamp": t0, "multiplier": 0.00001, "ask": 1.08010, "bid": 1.08000,
            "times": [0, 1500, 60000], "asks": [0, 1, -2], "bids": [0, 2, -1],
            "askVolumes": [1_000_000, 2_000_000, 500_000], "bidVolumes": [1_500_000, 1_000_000, 250_000]}
    urls = _fake_server(monkeypatch, body=body)
    df = fetch_data.fetch_dukascopy_hour("EURUSD", datetime(2026, 3, 4, 10, tzinfo=UTC))
    assert len(urls) == 1 and "jetta.dukascopy.com/v1/ticks/EUR-USD/2026/3/4/10" in urls[0]
    assert df["bid"].tolist() == [1.08, 1.08002, 1.08001]
    assert df["offer"].tolist() == [1.0801, 1.08011, 1.08009]
    assert df["time_utc"].iloc[0] == pd.Timestamp("2026-03-04 10:00:01", tz="UTC")
    assert df["offer_volume"].iloc[0] == 1.0


def test_real_library_403_is_reported_as_blocked(monkeypatch):
    _fake_server(monkeypatch, status=403, body={"error": "forbidden"})
    with pytest.raises(fetch_data.DukascopyError) as e:
        fetch_data.fetch_dukascopy_hour("EURUSD", datetime(2026, 3, 4, 10, tzinfo=UTC))
    assert e.value.blocked and e.value.status_code == 403


def test_real_library_html_page_is_reported_as_blocked(monkeypatch):
    _fake_server(monkeypatch, status=200, body=None)       # a web page instead of data
    with pytest.raises(fetch_data.DukascopyError) as e:
        fetch_data.fetch_dukascopy_hour("EURUSD", datetime(2026, 3, 4, 10, tzinfo=UTC))
    assert e.value.blocked
