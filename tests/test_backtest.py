"""
Tests for the cost report (backtest.py). Synthetic prices and synthetic IG
recordings; temporary folders only.
"""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

import backtest as bt
import config
import features as ft
import fetch_data
import labels as lb
from tests.test_feature_set import market_minutes, random_bars
from tests.test_labels import flat_bars, set_minute

UTC = timezone.utc
PIP = 0.0001


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "BARS_1M_DIR", tmp_path / "data" / "bars_1m")
    monkeypatch.setattr(config, "DUKASCOPY_BARS_1M_DIR", tmp_path / "data" / "dukascopy" / "bars_1m")
    yield tmp_path


def record_ig(instrument, hours, spread, day="2026-11-10"):
    """Fake IG recordings: one minute bar per minute of the given UTC hours."""
    t = pd.DatetimeIndex([])
    for h in hours:
        t = t.append(pd.date_range(f"{day} {h:02d}:00", periods=60, freq="1min", tz="UTC"))
    m = 1.1
    df = pd.DataFrame({"time_utc": t, "bid_open": m, "bid_high": m, "bid_low": m, "bid_close": m,
                       "offer_open": m + spread, "offer_high": m + spread, "offer_low": m + spread,
                       "offer_close": m + spread, "tick_count": 3, "spread_mean": spread,
                       "spread_max": spread, "partial": False})[fetch_data.BAR_COLUMNS]
    fetch_data.write_parquet_atomic(df, fetch_data.bar_file(instrument, date.fromisoformat(day)))


# --------------------------------------------------------------------------
# Today's rule reproduces the labels exactly
# --------------------------------------------------------------------------
def test_todays_rule_matches_the_labels():
    hist = random_bars(market_minutes("2025-02-24", 12), seed=5)
    record_ig("EURUSD", range(13, 19), 0.9 * PIP)
    b1 = ft.to_indexed(hist)
    lab, _ = lb.build_labels(hist, "EURUSD")
    _, res = bt.run_costs(b1, "EURUSD", scenarios=[("today", 60, 0)])
    t = res[0]["trades"]
    for side in ("buy", "sell"):
        got = t[t["side"] == side].set_index("entry_utc")
        want = lab.set_index("entry_utc")
        common = got.index.intersection(want.index)
        assert len(common) == len(want) > 0
        np.testing.assert_allclose(got.loc[common, "result_r"], want.loc[common, f"{side}_result_r"], atol=1e-9)
        assert (got.loc[common, "outcome"] == want.loc[common, f"{side}_outcome"]).all()
        assert (got["funding_r"] == 0).all()                 # closed before the rollover


# --------------------------------------------------------------------------
# Close-by times and funding
# --------------------------------------------------------------------------
@pytest.mark.parametrize("entry,days,close", [
    ("2025-03-03 13:00", 1, "2025-03-04 18:00"),     # Monday -> Tuesday 22:00 MUT
    ("2025-03-03 13:00", 5, "2025-03-07 18:00"),     # Monday + 5 weekdays -> capped at Friday
    ("2025-03-06 15:00", 2, "2025-03-07 18:00"),     # Thursday + 2 -> Friday
    ("2025-03-07 13:00", 1, "2025-03-07 18:00"),     # Friday -> same day
    ("2025-03-05 13:00", 0, "2025-03-05 17:00"),     # today's rule: 4 hours
])
def test_close_by_days(entry, days, close):
    got = bt.close_by_days(pd.DatetimeIndex([pd.Timestamp(entry, tz="UTC")]), days)[0]
    assert got == pd.Timestamp(close, tz="UTC")


def test_rollovers_and_wednesday_triple():
    start, end = pd.Timestamp("2025-03-03", tz="UTC"), pd.Timestamp("2025-03-08", tz="UTC")
    r, w = bt.rollovers(start, end, wednesday_triple=True)
    times = pd.to_datetime(r, utc=True)
    # 17:00 New York = 22:00 UTC in winter
    assert pd.Timestamp("2025-03-04 22:00", tz="UTC") in times
    entry = bt.ft.ns(pd.DatetimeIndex([pd.Timestamp("2025-03-04 13:00", tz="UTC")]))
    exit_ = bt.ft.ns(pd.DatetimeIndex([pd.Timestamp("2025-03-06 18:00", tz="UTC")]))
    assert bt.nights_held(entry, exit_, r, w)[0] == 1 + 3          # Tuesday + Wednesday (x3)
    r2, w2 = bt.rollovers(start, end, wednesday_triple=False)
    assert bt.nights_held(entry, exit_, r2, w2)[0] == 2
    summer, _ = bt.rollovers(pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-07-03", tz="UTC"), True)
    assert pd.Timestamp("2025-07-01 21:00", tz="UTC") in pd.to_datetime(summer, utc=True)   # daylight saving


def test_funding_costs():
    nights = np.array([1.0, 4.0])
    assert np.allclose(bt.funding_cost("EURUSD", "buy", nights, np.array([1.1, 1.1])), [1.36 * PIP, 5.44 * PIP])
    assert np.allclose(bt.funding_cost("EURUSD", "sell", nights, np.array([1.1, 1.1])), [0.47 * PIP, 1.88 * PIP])
    px = np.array([6000.0, 6000.0])
    assert np.allclose(bt.funding_cost("US500", "buy", nights, px), nights * 6000 * 0.08 / 360)
    assert np.allclose(bt.funding_cost("US500", "sell", nights, px), nights * 6000 * 0.04 / 360)


# --------------------------------------------------------------------------
# Spreads and slippage
# --------------------------------------------------------------------------
def test_spread_table_sources_and_night_doubling():
    hist = flat_bars([date(2025, 3, 4), date(2025, 3, 5)])
    b1 = ft.to_indexed(hist)
    b1["spread_mean"] = 0.3 * PIP
    record_ig("EURUSD", [13, 14, 19], 0.9 * PIP)
    tab = bt.spread_table("EURUSD", b1)
    assert tab.loc[19, "source"] == "IG recorded" and tab.loc[19, "used"] == pytest.approx(0.9 * PIP)
    assert tab.loc[13, "source"] == "fixed (config)" and tab.loc[13, "used"] == pytest.approx(0.9 * PIP)
    assert tab.loc[3, "source"].startswith("estimate") and tab.loc[3, "used"] == pytest.approx(0.9 * PIP)  # 0.3 x 3
    s = bt.spread_per_minute(b1, tab, night_multiple=2.0)
    hour = b1.index.hour
    assert np.allclose(s[hour == 3], 1.8 * PIP) and np.allclose(s[hour == 14], 0.9 * PIP)


def test_spread_table_needs_recordings():
    b1 = ft.to_indexed(flat_bars([date(2025, 3, 4)]))
    with pytest.raises(ValueError, match="needs your recorded IG spreads"):
        bt.spread_table("EURUSD", b1)


def test_slippage_rules():
    b1 = ft.to_indexed(flat_bars([date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5)]))
    s = bt.slippage_per_minute("EURUSD", b1)
    h = b1.index.hour
    assert np.allclose(s[(h >= 13) & (h < 18)], 0.25 * PIP) and np.allclose(s[h < 13], 1.0 * PIP)
    si = bt.slippage_per_minute("US500", b1)
    a1 = 10 * PIP                                               # flat bars: 1-hour ATR = 10 pips
    t = pd.Timestamp
    day = b1.index >= t("2025-03-05", tz="UTC")
    assert si[day & (b1.index == t("2025-03-05 13:00", tz="UTC"))][0] == pytest.approx(a1 * 0.25 / 13.6)
    assert si[day & (b1.index == t("2025-03-05 03:00", tz="UTC"))][0] == pytest.approx(4 * a1 * 0.25 / 13.6)
    # US cash open 09:30 New York = 14:30 UTC in March (before US daylight saving): doubled for 15 minutes
    assert si[b1.index == t("2025-03-05 14:35", tz="UTC")][0] == pytest.approx(2 * a1 * 0.25 / 13.6)
    assert si[b1.index == t("2025-03-05 14:45", tz="UTC")][0] == pytest.approx(a1 * 0.25 / 13.6)


# --------------------------------------------------------------------------
# Daily ATR without lookahead; a multi-day trade
# --------------------------------------------------------------------------
def test_daily_atr_waits_for_the_day_to_complete():
    hist = random_bars(market_minutes("2025-01-27", 30), seed=2)
    b1 = ft.to_indexed(hist)
    base = ft.resample_bars(b1, 5).index
    a = bt.atr_at_decision(b1, base, "D")
    spiked = b1.copy()
    day = (spiked.index >= pd.Timestamp("2025-02-20 12:00", tz="UTC")) & (spiked.index < pd.Timestamp("2025-02-20 13:00", tz="UTC"))
    spiked.loc[day, ["bid_high", "offer_high"]] += 0.01
    b = bt.atr_at_decision(spiked, base, "D")
    # The trading day containing the spike ends at 17:00 New York (22:00 UTC) on 20 Feb.
    before = np.asarray(base + pd.Timedelta(minutes=5) < pd.Timestamp("2025-02-20 22:00", tz="UTC"))
    np.testing.assert_array_equal(np.nan_to_num(a[before], nan=-1), np.nan_to_num(b[before], nan=-1))
    assert b[~before][-1] > a[~before][-1]


def test_two_day_trade_hits_stop_at_night_with_night_slippage_and_funding():
    days = [date(2025, 3, d) for d in (3, 4, 5, 6, 7)]
    hist = flat_bars(days)
    record_ig("EURUSD", range(13, 19), 0.9 * PIP)
    hist = set_minute(hist, "2025-03-05 02:00", low=1.0985)          # Wednesday 06:00 MUT: buy stop hit
    b1 = ft.to_indexed(hist)
    _, res = bt.run_costs(b1, "EURUSD", scenarios=[("1h ATR, 2 days", 60, 2)])
    t = res[0]["trades"]
    r = t[(t["side"] == "buy") & (t["entry_utc"] == pd.Timestamp("2025-03-04 13:00", tz="UTC"))].iloc[0]
    assert r["outcome"] == "SL"
    assert r["nights"] == 1                                            # Tuesday's rollover only
    assert r["slippage_r"] == pytest.approx(1.0 * PIP / (10 * PIP))    # outside your window: 1 pip
    assert r["funding_r"] == pytest.approx(1.36 / 10)
    assert r["result_r"] == pytest.approx(-1 - 0.1 - 0.136)


# --------------------------------------------------------------------------
# Report and command line (cost-analysis permission)
# --------------------------------------------------------------------------
def test_cost_analysis_permission_allows_unapproved_but_keeps_final_check_locked(monkeypatch):
    monkeypatch.setitem(config.HISTORY_APPROVED_FOR_MODEL, "US500", False)       # as if not approved
    s, e = datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 2, 1, tzinfo=UTC)
    fetch_data.check_history_access("US500", s, e, fetch_data.PURPOSE_COST_ANALYSIS)
    with pytest.raises(fetch_data.HistoryAccessError):
        fetch_data.check_history_access("US500", s, e, fetch_data.PURPOSE_DEVELOPMENT)
    with pytest.raises(fetch_data.HistoryAccessError):
        fetch_data.check_history_access("US500", datetime(2026, 6, 1, tzinfo=UTC),
                                        datetime(2026, 7, 1, tzinfo=UTC), fetch_data.PURPOSE_COST_ANALYSIS)


def test_cli_report(capsys, monkeypatch):
    hist = random_bars(market_minutes("2025-02-03", 33), seed=9)
    tt = pd.to_datetime(hist["time_utc"], utc=True)
    for (y, m), part in hist.groupby([tt.dt.year, tt.dt.month]):
        fetch_data.write_parquet_atomic(part.reset_index(drop=True), fetch_data.history_bar_file("EURUSD", y, m))
    record_ig("EURUSD", range(13, 19), 0.9 * PIP)
    monkeypatch.setattr(bt.timeutils, "configure_logging", lambda **kw: None)
    assert bt.main(["costs", "--instrument", "EURUSD", "--save"]) == 0
    out = capsys.readouterr().out
    for part in ("Cost report: EURUSD", "Spread used by UTC hour", "A. Spreads as above",
                 "B. Spreads outside your window x2", "By entry weekday", "today: 1h ATR, same day - buy",
                 "daily ATR, 5 days - sell", "Saved to"):
        assert part in out, part


def test_cost_vs_move():
    t = pd.DataFrame({"result_r": [1.0, -1.0, 1.0, -1.0], "spread_r": 0.05, "slippage_r": 0.01, "funding_r": 0.04,
                      "day": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-07-02", "2026-01-01"])})
    m = bt.cost_vs_move(t)
    assert m["days_per_year"] == pytest.approx(3 / 1.0, rel=0.01)          # 3 decision days in one year
    assert m["years_to_see"] == pytest.approx(m["trades_to_see"] / m["days_per_year"])
    assert m["cost_r"] == pytest.approx(0.10)
    assert m["sd_r"] == pytest.approx(np.std([1, -1, 1, -1], ddof=1))
    assert m["cost_sd"] == pytest.approx(0.10 / m["sd_r"])
    assert m["trades_to_see"] == pytest.approx((2 / m["cost_sd"]) ** 2)


def test_index_report_says_points(capsys, monkeypatch):
    hist = random_bars(market_minutes("2025-02-03", 33), seed=4)
    hist[[c for c in hist.columns if c.startswith(("bid_", "offer_"))]] *= 5000      # an index-like price
    hist["spread_mean"] = 0.5
    tt = pd.to_datetime(hist["time_utc"], utc=True)
    for (y, m), part in hist.groupby([tt.dt.year, tt.dt.month]):
        fetch_data.write_parquet_atomic(part.reset_index(drop=True), fetch_data.history_bar_file("US500", y, m))
    record_ig("US500", range(13, 19), 0.4)
    monkeypatch.setattr(bt.timeutils, "configure_logging", lambda **kw: None)
    assert bt.main(["costs", "--instrument", "US500"]) == 0
    out = capsys.readouterr().out
    assert "(pts)" in out and "(pips)" not in out and "Spread used by UTC hour (points" in out
    assert "A2. Cost relative to the typical move" in out and "decisions to see" in out and "years" in out
