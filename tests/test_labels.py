"""
Tests for labels.py on synthetic prices with known outcomes.
Flat EUR/USD at 1.1000; each 1-minute bar spans +-5 pips, so the 1-hour ATR
is 10 pips: SL 10 pips, TP 15 pips. IG spread 0.9 pips, stop slippage 0.25.
"""

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

import config
import features as ft
import fetch_data
import labels as lb

UTC = timezone.utc
PIP = 0.0001
S = 0.9 * PIP           # IG spread
SLIP = 0.25 * PIP


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "data" / "datasets")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "DUKASCOPY_BARS_1M_DIR", tmp_path / "data" / "dukascopy" / "bars_1m")
    yield tmp_path


def flat_bars(days, half_range=5 * PIP, mid=1.1000):
    """24h of 1-minute bars per day; Dukascopy's own spread tiny (labels use IG's)."""
    t = pd.DatetimeIndex([], tz="UTC")
    for d in days:
        t = t.append(pd.date_range(f"{d} 00:00", periods=24 * 60, freq="1min", tz="UTC"))
    h = 0.00001
    df = pd.DataFrame({"time_utc": t,
                       "bid_open": mid - h, "bid_high": mid + half_range - h, "bid_low": mid - half_range - h,
                       "bid_close": mid - h, "offer_open": mid + h, "offer_high": mid + half_range + h,
                       "offer_low": mid - half_range + h, "offer_close": mid + h,
                       "tick_count": 5, "spread_mean": 2 * h, "spread_max": 2 * h, "partial": False})
    return df[fetch_data.BAR_COLUMNS]


def set_minute(df, when, open_=None, high=None, low=None, close=None):
    """Change one minute's MID prices (bid/offer keep the tiny Dukascopy spread)."""
    h = 0.00001
    row = pd.to_datetime(df["time_utc"], utc=True) == pd.Timestamp(when, tz="UTC")
    assert row.sum() == 1
    for name, v in (("open", open_), ("high", high), ("low", low), ("close", close)):
        if v is not None:
            df.loc[row, f"bid_{name}"] = v - h
            df.loc[row, f"offer_{name}"] = v + h
    return df


MON, TUE, WED = date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5)


def build(df):
    return lb.build_labels(df, "EURUSD")


def row_at(lab, when):
    r = lab[pd.to_datetime(lab["entry_utc"], utc=True) == pd.Timestamp(when, tz="UTC")]
    assert len(r) == 1, f"no single entry at {when}"
    return r.iloc[0]


# --------------------------------------------------------------------------
# Window, close-by time, barriers
# --------------------------------------------------------------------------
def test_entries_only_17_to_20_mut_and_closed_by_22_mut():
    lab, skipped = build(flat_bars([MON, TUE, WED]))
    e = pd.to_datetime(lab["entry_utc"], utc=True)
    mut = e.dt.tz_convert("Indian/Mauritius")
    assert mut.dt.strftime("%H:%M").min() == "17:00" and mut.dt.strftime("%H:%M").max() == "20:00"
    # Monday's first hour has no 1-hour ATR yet (14 completed hours needed).
    assert len(lab) == 25 + 37 + 37
    assert skipped["1-hour ATR not available yet"] == 12
    cb = pd.to_datetime(lab["close_by_utc"], utc=True)
    assert row_at(lab, "2025-03-05 13:00")["close_by_utc"] == pd.Timestamp("2025-03-05 17:00", tz="UTC")  # 4 h
    assert row_at(lab, "2025-03-05 15:30")["close_by_utc"] == pd.Timestamp("2025-03-05 18:00", tz="UTC")  # 22:00 MUT
    assert (cb.dt.tz_convert("Indian/Mauritius").dt.hour <= 22).all()


def test_barriers_from_one_hour_atr():
    lab, _ = build(flat_bars([MON, TUE, WED]))
    r = row_at(lab, "2025-03-05 13:00")
    assert r["atr_1h"] == pytest.approx(10 * PIP)
    assert r["sl_dist"] == pytest.approx(10 * PIP) and r["tp_dist"] == pytest.approx(15 * PIP)
    assert not r["floor_applied"]
    assert r["buy_entry"] == pytest.approx(1.1000 + S / 2)       # buy at the offer
    assert r["sell_entry"] == pytest.approx(1.1000 - S / 2)      # sell at the bid
    assert r["buy_tp"] == pytest.approx(r["buy_entry"] + 15 * PIP)
    assert r["sell_sl"] == pytest.approx(r["sell_entry"] + 10 * PIP)


def test_floor_keeps_the_ratio():
    lab, _ = build(flat_bars([MON, TUE, WED], half_range=2 * PIP))     # ATR 4 pips < 9-pip floor
    r = row_at(lab, "2025-03-05 13:00")
    assert r["floor_applied"]
    assert r["sl_dist"] == pytest.approx(9 * PIP) and r["tp_dist"] == pytest.approx(13.5 * PIP)


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------
def test_flat_market_times_out_losing_the_spread():
    lab, _ = build(flat_bars([MON, TUE, WED]))
    r = row_at(lab, "2025-03-05 13:00")
    for side in ("buy", "sell"):
        assert r[f"{side}_outcome"] == "TIMEOUT" and r[f"{side}_win"] == 0
        assert r[f"{side}_result_r"] == pytest.approx(-S / (10 * PIP))       # -0.09 R
        assert r[f"{side}_exit_utc"] == pd.Timestamp("2025-03-05 17:00", tz="UTC")


def test_rise_hits_buy_tp_and_sell_sl_with_slippage():
    df = set_minute(flat_bars([MON, TUE, WED]), "2025-03-05 13:30", high=1.1020)
    r = row_at(build(df)[0], "2025-03-05 13:00")
    assert r["buy_outcome"] == "TP" and r["buy_win"] == 1
    assert r["buy_result_r"] == pytest.approx(1.5)
    assert r["buy_exit_utc"] == pd.Timestamp("2025-03-05 13:30", tz="UTC")
    assert r["sell_outcome"] == "SL"
    assert r["sell_exit_price"] == pytest.approx(r["sell_sl"] + SLIP)
    assert r["sell_result_r"] == pytest.approx(-1 - SLIP / (10 * PIP))      # -1.025 R


def test_tp_and_sl_in_the_same_minute_counts_as_sl():
    df = set_minute(flat_bars([MON, TUE, WED]), "2025-03-05 13:20", high=1.1030, low=1.0970)
    r = row_at(build(df)[0], "2025-03-05 13:00")
    for side in ("buy", "sell"):
        assert r[f"{side}_outcome"] == "SL" and bool(r[f"{side}_same_minute"])


def test_first_barrier_in_time_wins():
    df = flat_bars([MON, TUE, WED])
    df = set_minute(df, "2025-03-05 13:10", low=1.0985)      # buy SL first (13:10) ...
    df = set_minute(df, "2025-03-05 13:40", high=1.1030)     # ... TP later (13:40)
    r = row_at(build(df)[0], "2025-03-05 13:00")
    assert r["buy_outcome"] == "SL" and not r["buy_same_minute"]
    assert r["buy_exit_utc"] == pd.Timestamp("2025-03-05 13:10", tz="UTC")


def test_price_gapping_past_the_stop_fills_at_the_open():
    df = set_minute(flat_bars([MON, TUE, WED]), "2025-03-05 13:40", open_=1.0980, low=1.0975, close=1.0980)
    r = row_at(build(df)[0], "2025-03-05 13:00")
    assert r["buy_outcome"] == "SL"
    assert r["buy_exit_price"] == pytest.approx(1.0980 - S / 2 - SLIP)     # worse than the SL level
    assert r["buy_result_r"] < -2


# --------------------------------------------------------------------------
# Skipped entries
# --------------------------------------------------------------------------
def test_holiday_and_early_close_days_have_no_entries():
    days = [date(2025, 11, 26), date(2025, 11, 27), date(2025, 11, 28)]   # Wed, Thanksgiving, early close
    lab, skipped = build(flat_bars(days))
    got = set(pd.to_datetime(lab["entry_utc"], utc=True).dt.date)
    assert got == {date(2025, 11, 26)}
    assert skipped["holiday or year-end"] == 2 * 37


def test_gap_blocks_trades_across_it_and_the_hour_after():
    df = flat_bars([MON, TUE, WED])
    t = pd.to_datetime(df["time_utc"], utc=True)
    df = df[~((t >= pd.Timestamp("2025-03-05 14:00", tz="UTC")) & (t < pd.Timestamp("2025-03-05 14:30", tz="UTC")))]
    lab, skipped = build(df)
    e = pd.to_datetime(lab[pd.to_datetime(lab["entry_utc"], utc=True).dt.date == WED]["entry_utc"], utc=True)
    # Wed entries 13:00-13:55 would run across the gap; 14:00 has no price (inside the gap);
    # 14:35-15:30 are within 12 bars after it. The first allowed entry is 15:35.
    assert e.min() == pd.Timestamp("2025-03-05 15:35", tz="UTC")
    assert skipped["gap in the data during the trade"] == 12
    assert skipped["no price at the entry minute"] == 1
    assert skipped["within 12 bars after a gap or weekend"] == 12


def test_no_lookahead_in_the_barriers():
    calm = flat_bars([MON, TUE, WED])
    spike = set_minute(calm.copy(), "2025-03-05 13:30", high=1.1100)
    a, b = build(calm)[0], build(spike)[0]
    ea, eb = pd.to_datetime(a["entry_utc"], utc=True), pd.to_datetime(b["entry_utc"], utc=True)
    before = ea < pd.Timestamp("2025-03-05 14:00", tz="UTC")     # the 13:00 hour completes at 14:00
    assert np.allclose(a.loc[before, "sl_dist"], b.loc[eb < pd.Timestamp("2025-03-05 14:00", tz="UTC"), "sl_dist"])
    assert row_at(b, "2025-03-05 14:00")["sl_dist"] > row_at(a, "2025-03-05 14:00")["sl_dist"]


# --------------------------------------------------------------------------
# Saving and the report
# --------------------------------------------------------------------------
def test_build_save_and_report(capsys):
    df = set_minute(flat_bars([MON, TUE, WED]), "2025-03-05 13:30", high=1.1020)
    t = pd.to_datetime(df["time_utc"], utc=True)
    fetch_data.write_parquet_atomic(df.reset_index(drop=True), fetch_data.history_bar_file("EURUSD", 2025, 3))
    assert lb.main(["build", "--save"]) == 0
    out = capsys.readouterr().out
    assert "entry 17:xx MUT (13 UTC)" in out and "timeout" in out
    path = lb.label_file("EURUSD")
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["settings"]["barrier_atr_minutes"] == 60 and meta["settings"]["entry_last_mut"] == [20, 0]
    assert meta["end_utc"].startswith("2026-04-30")                   # development period only
    saved = pd.read_parquet(path)
    assert len(saved) == meta["rows"] == 99
    assert len(list((config.DATA_DIR / "reports").glob("labels_report_EURUSD_*.txt"))) == 1


def test_settings_match_decisions():
    s = lb.settings("EURUSD")
    assert (s["sl_atr_multiple"], s["tp_atr_multiple"], s["stop_floor_spreads"]) == (1.0, 1.5, 10)
    assert (s["entry_first_mut"], s["entry_last_mut"], s["close_by_mut"]) == ((17, 0), (20, 0), (22, 0))
    assert s["max_hold_hours"] == 4 and s["ig_spread_pips"] == 0.9 and s["sl_slippage_pips"] == 0.25
