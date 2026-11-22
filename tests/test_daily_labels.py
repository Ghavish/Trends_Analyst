"""
Tests for the Phase 2B daily-decision labels (labels.py, daily section).
Flat EUR/USD at 1.1000; each 1-minute bar spans +-5 pips, so a trading day's
range and the daily ATR(14) are 10 pips: stop 10 pips, take profit 30 pips.
IG spread 0.9 pips all day (recorded in the window; night estimated from
Dukascopy x the same ratio), stop slippage 0.25 pips in the window, 1 pip
outside. Funding: buy 1.36 pips, sell 0.47 pips a night, Wednesday x3.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import config
import features as ft
import fetch_data
import labels as lb
import train as tr

PIP = 0.0001
H = 0.00001                     # half of Dukascopy's own (tiny) spread in the synthetic bars


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "data" / "datasets")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "BARS_1M_DIR", tmp_path / "data" / "bars_1m")
    monkeypatch.setattr(config, "DUKASCOPY_BARS_1M_DIR", tmp_path / "data" / "dukascopy" / "bars_1m")
    record_ig()
    yield tmp_path


def record_ig(day="2026-11-10"):
    """IG recordings 13:00-18:59 UTC at 0.9 pips (Dukascopy's 0.2 -> ratio 4.5 -> 0.9 at night too)."""
    t = pd.date_range(f"{day} 13:00", periods=6 * 60, freq="1min", tz="UTC")
    m, s = 1.1, 0.9 * PIP
    df = pd.DataFrame({"time_utc": t, "bid_open": m, "bid_high": m, "bid_low": m, "bid_close": m,
                       "offer_open": m + s, "offer_high": m + s, "offer_low": m + s, "offer_close": m + s,
                       "tick_count": 3, "spread_mean": s, "spread_max": s, "partial": False})
    fetch_data.write_parquet_atomic(df[fetch_data.BAR_COLUMNS], fetch_data.bar_file("EURUSD", date.fromisoformat(day)))


def weekdays(first, last):
    return [d.date() for d in pd.bdate_range(first, last)]


def flat_bars(days, mid=1.1000, half_range=5 * PIP):
    t = pd.DatetimeIndex([], tz="UTC")
    for d in days:
        t = t.append(pd.date_range(f"{d} 00:00", periods=24 * 60, freq="1min", tz="UTC"))
    df = pd.DataFrame({"time_utc": t,
                       "bid_open": mid - H, "bid_high": mid + half_range - H, "bid_low": mid - half_range - H,
                       "bid_close": mid - H, "offer_open": mid + H, "offer_high": mid + half_range + H,
                       "offer_low": mid - half_range + H, "offer_close": mid + H,
                       "tick_count": 5, "spread_mean": 2 * H, "spread_max": 2 * H, "partial": False})
    return df[fetch_data.BAR_COLUMNS]


def set_minutes(df, start, end=None, open_=None, high=None, low=None, close=None):
    """Change the MID prices of the minutes in [start, end] (one minute if end is None)."""
    t = pd.to_datetime(df["time_utc"], utc=True)
    a = pd.Timestamp(start, tz="UTC")
    rows = (t == a) if end is None else ((t >= a) & (t <= pd.Timestamp(end, tz="UTC")))
    assert rows.sum() >= 1
    for name, v in (("open", open_), ("high", high), ("low", low), ("close", close)):
        if v is not None:
            df.loc[rows, f"bid_{name}"] = v - H
            df.loc[rows, f"offer_{name}"] = v + H
    return df


def drop_minutes(df, start, end):
    t = pd.to_datetime(df["time_utc"], utc=True)
    keep = (t < pd.Timestamp(start, tz="UTC")) | (t > pd.Timestamp(end, tz="UTC"))
    return df[keep].reset_index(drop=True)


DAYS = weekdays("2025-01-27", "2025-03-14")     # Presidents Day (NYSE holiday): Monday 17 Feb 2025


def build(df):
    wide, skipped, _ = lb.build_daily_labels(ft.to_indexed(df), "EURUSD")
    return wide, skipped


def at(wide, when):
    r = wide[pd.to_datetime(wide["entry_utc"], utc=True) == pd.Timestamp(when, tz="UTC")]
    assert len(r) == 1, f"no single decision at {when}"
    return r.iloc[0]


# --------------------------------------------------------------------------
# Decision times, close-by times, holidays
# --------------------------------------------------------------------------
def test_one_decision_monday_to_thursday_at_1715_mut():
    wide, skipped = build(flat_bars(DAYS))
    e = pd.to_datetime(wide["entry_utc"], utc=True)
    assert (e.dt.hour == 13).all() and (e.dt.minute == 15).all()              # 17:15 MUT = 13:15 UTC
    assert set(e.dt.tz_convert(config.TZ_MAURITIUS).dt.day_of_week) == {0, 1, 2, 3}
    assert e.dt.normalize().is_unique
    assert pd.Timestamp("2025-02-17 13:15", tz="UTC") not in set(e)            # NYSE holiday: no decision
    assert skipped["holiday or year-end"] == 1
    assert skipped["daily ATR not available yet"] > 0                          # the first weeks: ATR warming up
    assert (pd.to_datetime(wide["decision_bar_utc"], utc=True) == e - pd.Timedelta(minutes=5)).all()


@pytest.mark.parametrize("entry, close", [
    ("2025-03-06 13:15", "2025-03-07 13:15"),   # Thursday -> Friday's decision time
    ("2025-03-03 13:15", "2025-03-04 13:15"),   # Monday -> Tuesday
    ("2025-04-17 13:15", "2025-04-18 13:15"),   # Thursday before Good Friday: still Friday's time
    ("2025-03-07 13:15", "2025-03-10 13:15"),   # (a Friday would close Monday; no Friday decisions)
])
def test_close_by_is_the_next_weekdays_decision_time(entry, close):
    got = lb.daily_close_by(pd.DatetimeIndex([pd.Timestamp(entry, tz="UTC")]))
    assert got[0] == pd.Timestamp(close, tz="UTC")


# --------------------------------------------------------------------------
# Barriers, costs, the label
# --------------------------------------------------------------------------
def test_flat_day_times_out_losing_spread_and_funding_with_wednesday_triple():
    wide, _ = build(flat_bars(DAYS))
    tue = at(wide, "2025-03-04 13:15")
    assert tue["atr_d"] == pytest.approx(10 * PIP) and tue["sl_dist"] == pytest.approx(10 * PIP)
    assert tue["buy_outcome"] == "TIMEOUT" and tue["sell_outcome"] == "TIMEOUT"
    assert tue["buy_hours_held"] == pytest.approx(24.0)
    assert tue["buy_nights"] == 1 and tue["buy_result_r"] == pytest.approx(-0.09 - 0.136)
    assert tue["sell_result_r"] == pytest.approx(-0.09 - 0.047)
    assert tue["buy_win"] == 0 and tue["sell_win"] == 0                        # win = result above 0
    wed = at(wide, "2025-03-05 13:15")
    assert wed["buy_nights"] == 3 and wed["buy_funding_r"] == pytest.approx(3 * 0.136)
    thu = at(wide, "2025-03-06 13:15")
    assert thu["buy_nights"] == 1 and pd.Timestamp(thu["close_by_utc"]) == pd.Timestamp("2025-03-07 13:15", tz="UTC")


def test_take_profit_at_three_daily_atr_and_stop_at_one():
    df = set_minutes(flat_bars(DAYS), "2025-03-04 20:00", high=1.1035)       # +35 pips before the rollover
    wide, _ = build(df)
    r = at(wide, "2025-03-04 13:15")
    assert r["buy_outcome"] == "TP" and r["buy_result_r"] == pytest.approx(3.0) and r["buy_nights"] == 0
    assert r["buy_win"] == 1
    assert r["sell_outcome"] == "SL"                                           # sell stop 10 pips away
    # outside your window (20:00 UTC = 00:00 MUT): 1 pip slippage
    assert r["sell_result_r"] == pytest.approx(-1.0 - 0.1)
    assert r["sell_win"] == 0


def test_small_gain_after_costs_is_a_win():
    df = set_minutes(flat_bars(DAYS), "2025-03-03 13:16", "2025-03-04 23:59", open_=1.1004, high=1.1009,
                     low=1.0999, close=1.1004)
    wide, _ = build(df)
    r = at(wide, "2025-03-03 13:15")                                           # Monday, exit Tuesday 13:15 at +4 pips
    assert r["buy_outcome"] == "TIMEOUT"
    assert r["buy_result_r"] == pytest.approx(0.4 - 0.09 - 0.136) and r["buy_win"] == 1
    assert r["sell_result_r"] == pytest.approx(-0.4 - 0.09 - 0.047) and r["sell_win"] == 0


def test_market_shut_at_close_time_exits_at_the_first_price_after_it_reopens():
    df = drop_minutes(flat_bars(DAYS), "2025-03-07 13:00", "2025-03-07 13:39")   # Friday: no prices 13:00-13:39
    df = set_minutes(df, "2025-03-07 13:40", "2025-03-07 13:59", open_=1.1005, high=1.1006, low=1.1004,
                     close=1.1005)
    wide, skipped = build(df)
    r = at(wide, "2025-03-06 13:15")                                           # Thursday's decision
    assert r["buy_outcome"] == "TIMEOUT"
    assert r["buy_hours_held"] == pytest.approx(24 + 25 / 60)                  # exit 13:40, not before the gap
    assert r["buy_result_r"] == pytest.approx(0.5 - 0.09 - 0.136)              # the gap counts
    assert r["sell_result_r"] == pytest.approx(-0.5 - 0.09 - 0.047)
    assert skipped["no price after the close time"] == 0


def test_reopening_beyond_the_stop_fills_at_the_reopening_price():
    df = drop_minutes(flat_bars(DAYS), "2025-03-07 13:00", "2025-03-07 13:39")
    df = set_minutes(df, "2025-03-07 13:40", "2025-03-07 13:59", open_=1.0980, high=1.0981, low=1.0979,
                     close=1.0980)
    wide, _ = build(df)
    r = at(wide, "2025-03-06 13:15")
    assert r["buy_outcome"] == "SL"
    # entry 1.10000 + 0.45 pip; exit 1.0980 - 0.45 pip - 0.25 pip slippage (in your window)
    assert r["buy_result_r"] == pytest.approx(-(20 + 0.45 + 0.45 + 0.25) / 10 - 0.136)
    assert r["buy_slippage_r"] > 1.0                                           # the gap is counted as slippage
    assert r["sell_outcome"] == "TIMEOUT" and r["sell_result_r"] == pytest.approx(2.0 - 0.09 - 0.047)


def test_data_gap_longer_than_allowed_drops_the_decision():
    df = drop_minutes(flat_bars(DAYS), "2025-03-04 14:00", "2025-03-04 23:59")
    df = df[pd.to_datetime(df["time_utc"], utc=True).dt.normalize() != pd.Timestamp("2025-03-05", tz="UTC")]
    df = df.reset_index(drop=True)
    wide_ok, _ = build(df)                                                     # a 1.4-day gap is allowed (4 days)
    assert pd.Timestamp("2025-03-04 13:15", tz="UTC") in set(pd.to_datetime(wide_ok["entry_utc"], utc=True))
    short = df[pd.to_datetime(df["time_utc"], utc=True) < pd.Timestamp("2025-03-11", tz="UTC")]
    wide, skipped = build(short.reset_index(drop=True))
    assert pd.Timestamp("2025-03-10 13:15", tz="UTC") not in set(pd.to_datetime(wide["entry_utc"], utc=True))
    assert skipped["no price after the close time"] >= 1


def test_no_lookahead_decisions_do_not_change_when_later_data_is_cut():
    df = set_minutes(flat_bars(DAYS), "2025-03-12 15:00", high=1.1060)
    full, _ = build(df)
    cut, _ = build(df[pd.to_datetime(df["time_utc"], utc=True) < pd.Timestamp("2025-03-12 13:16", tz="UTC")]
                   .reset_index(drop=True))
    done = pd.to_datetime(cut["close_by_utc"], utc=True) <= pd.Timestamp("2025-03-12 13:15", tz="UTC")
    keys = cut[done]["entry_utc"]
    a = full.set_index("entry_utc").loc[keys]
    b = cut.set_index("entry_utc").loc[keys]
    for c in ("atr_d", "buy_result_r", "sell_result_r", "buy_outcome"):
        assert (a[c].to_numpy() == b[c].to_numpy()).all()


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def test_baselines_use_only_data_before_the_test_period():
    days = pd.bdate_range("2020-01-01", "2021-12-31")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"day": days, "buy_result_r": rng.normal(0, 1, len(days)),
                       "sell_result_r": rng.normal(0, 1, len(days))})
    before_2021 = df["day"] < "2021-01-01"
    df.loc[before_2021, "sell_result_r"] += 0.5                    # sells better before the test period
    df.loc[~before_2021, "buy_result_r"] += 0.5                    # ... buys better in it (baseline can't know)
    for s in ("buy", "sell"):
        df[f"{s}_win"] = (df[f"{s}_result_r"] > 0).astype(np.int8)
    rnd = tr.Round(1, pd.Timestamp("2020-07-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-07-01"))
    base, trades = lb.daily_baselines(df, [rnd])
    r = base.iloc[0]
    test = df[(df["day"] >= "2021-01-01") & (df["day"] < "2021-07-01")]
    assert r["side"] == "sell" and r["decisions"] == len(test)
    assert r["avg_r"] == pytest.approx(test["sell_result_r"].mean())
    _, cal, _ = tr.split(df, rnd)
    before = pd.concat([tr.split(df, rnd)[0], cal])
    assert r["sell_base_rate"] == pytest.approx(before["sell_win"].mean())
    assert r["sell_base_log_loss"] == pytest.approx(
        tr.log_loss(test["sell_win"], np.full(len(test), before["sell_win"].mean())))
    assert len(trades) == len(test) and (trades["side"] == "sell").all()


# --------------------------------------------------------------------------
# Guard, save, report, command line
# --------------------------------------------------------------------------
def test_instruments_not_cleared_overnight_are_refused(capsys):
    assert config.HISTORY_APPROVED_OVERNIGHT["US500"] is False
    with pytest.raises(fetch_data.HistoryAccessError, match="overnight"):
        lb.daily_build_and_save("US500")
    assert lb.main(["daily-build", "--instrument", "US500"]) == 1
    assert "STOPPED" in capsys.readouterr().out


def test_daily_build_save_and_report(capsys):
    df = flat_bars(DAYS)
    for (y, m), part in df.groupby([pd.to_datetime(df["time_utc"]).dt.year, pd.to_datetime(df["time_utc"]).dt.month]):
        fetch_data.write_parquet_atomic(part.reset_index(drop=True), fetch_data.history_bar_file("EURUSD", y, m))
    assert lb.main(["daily-build", "--instrument", "EURUSD", "--save"]) == 0
    out = capsys.readouterr().out
    assert "Monday-Thursday 17:15 MUT" in out and "Same direction" in out and "cost/sd" in out
    path = lb.daily_label_file("EURUSD")
    meta = json.loads(path.with_suffix(".json").read_text())
    s = meta["settings"]
    assert (s["sl_atr"], s["tp_atr"], s["decision_mut"], s["entry_weekdays"]) == (1.0, 3.0, [17, 15], [0, 1, 2, 3])
    assert meta["end_utc"].startswith("2026-04-30")                           # development period only
    saved = pd.read_parquet(path)
    assert len(saved) == meta["rows"] > 0
    assert len(list((config.DATA_DIR / "reports").glob("daily_labels_report_EURUSD_*.txt"))) == 1
    assert lb.main(["daily-report", "--instrument", "EURUSD"]) == 0
