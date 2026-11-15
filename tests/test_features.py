"""
Tests for features.py (Phase 2 step 2a): bars, ATR, no-lookahead alignment,
and the ATR report. Synthetic data, temporary folders only.
"""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import config
import features as ft
import fetch_data

UTC = timezone.utc


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "DUKASCOPY_BARS_1M_DIR", tmp_path / "data" / "dukascopy" / "bars_1m")
    yield tmp_path


def bars_1m(times, mids, rng=0.0002, spread=0.00002):
    """1-minute bid/offer bars around given mid closes; high/low = mid +- rng/2."""
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    mids = np.asarray(mids, float)
    h = spread / 2
    return pd.DataFrame({
        "time_utc": times,
        "bid_open": mids - h, "bid_high": mids + rng / 2 - h, "bid_low": mids - rng / 2 - h, "bid_close": mids - h,
        "offer_open": mids + h, "offer_high": mids + rng / 2 + h, "offer_low": mids - rng / 2 + h, "offer_close": mids + h,
        "tick_count": 3, "spread_mean": spread, "spread_max": spread, "partial": False,
    })[fetch_data.BAR_COLUMNS]


def minutes(start, n):
    return pd.date_range(start, periods=n, freq="1min", tz="UTC")


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------
def test_five_minute_bars_labelled_by_start_with_correct_ohlc():
    t = minutes("2026-03-04 10:00", 10)
    b = ft.to_indexed(bars_1m(t, np.arange(10) * 0.0001 + 1.08))
    five = ft.resample_bars(b, 5)
    assert list(five.index) == [pd.Timestamp("2026-03-04 10:00", tz="UTC"), pd.Timestamp("2026-03-04 10:05", tz="UTC")]
    first = five.iloc[0]
    assert first["bid_open"] == pytest.approx(1.08 - 0.00001)
    assert first["bid_close"] == pytest.approx(1.0804 - 0.00001)
    assert first["bid_high"] == pytest.approx(1.0804 + 0.0001 - 0.00001)
    assert first["n_minutes"] == 5 and first["tick_count"] == 15


def test_no_bar_for_a_period_without_data():
    t = minutes("2026-03-04 10:00", 5).append(minutes("2026-03-04 10:20", 5))
    five = ft.resample_bars(ft.to_indexed(bars_1m(t, [1.08] * 10)), 5)
    assert [x.strftime("%H:%M") for x in five.index] == ["10:00", "10:20"]


def test_hourly_bars_align_to_the_hour():
    t = minutes("2026-03-04 09:30", 90)
    h = ft.resample_bars(ft.to_indexed(bars_1m(t, [1.08] * 90)), 60)
    assert [x.strftime("%H:%M") for x in h.index] == ["09:00", "10:00"]
    assert list(h["n_minutes"]) == [30, 60]


# --------------------------------------------------------------------------
# ATR
# --------------------------------------------------------------------------
def test_gap_rule():
    idx = pd.DatetimeIndex(["2026-03-04 10:00", "2026-03-04 10:05", "2026-03-04 10:25",
                            "2026-03-04 10:45"], tz="UTC")
    # 10:05 bar ends 10:10; next starts 10:25 -> 15 min empty = not a gap; 10:30 -> 10:45 also 15.
    assert list(ft.follows_previous(idx, 5)) == [False, True, True, True]
    idx2 = idx.insert(4, pd.Timestamp("2026-03-04 11:10", tz="UTC"))   # 10:50 -> 11:10 = 20 min: gap
    assert list(ft.follows_previous(idx2, 5))[-1] is False


def test_atr_constant_range_and_wilder_smoothing():
    idx = pd.date_range("2026-03-04 10:00", periods=30, freq="5min", tz="UTC")
    m = pd.DataFrame({"open": 1.0, "high": 1.0010, "low": 1.0, "close": 1.0005}, index=idx)
    a = ft.atr(m, 5, n=14)
    assert a.iloc[:13].isna().all()
    assert a.iloc[-1] == pytest.approx(0.0010)
    # Wilder: a jump in true range moves ATR by 1/14 of the difference.
    m.loc[idx[20], "high"] = 1.0024                                # TR 0.0024 instead of 0.0010
    b = ft.atr(m, 5, n=14)
    assert b.iloc[20] == pytest.approx(b.iloc[19] + (0.0024 - b.iloc[19]) / 14)


def test_atr_ignores_the_jump_across_a_weekend():
    fri = pd.date_range("2026-03-06 20:00", periods=20, freq="5min", tz="UTC")
    sun = pd.date_range("2026-03-08 21:00", periods=1, freq="5min", tz="UTC")
    idx = fri.append(sun)
    m = pd.DataFrame({"open": 1.0, "high": 1.0010, "low": 1.0, "close": 1.0005}, index=idx)
    m.loc[sun, ["open", "high", "low", "close"]] = [1.0100, 1.0110, 1.0100, 1.0105]   # 100-pip weekend jump
    a = ft.atr(m, 5, n=14)
    assert a.iloc[-1] == pytest.approx(a.iloc[-2])                 # TR stays 10 pips (high - low)


# --------------------------------------------------------------------------
# No lookahead: longer timeframes only when complete
# --------------------------------------------------------------------------
def test_hourly_value_used_only_once_its_hour_is_complete():
    hourly = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-03-04 09:00", "2026-03-04 10:00"], tz="UTC"))
    base = pd.date_range("2026-03-04 10:45", periods=4, freq="5min", tz="UTC")   # 10:45 .. 11:00
    out = ft.align_completed(hourly, 60, base, 5)
    # 10:45 and 10:50 bars complete at 10:50 / 10:55: only the 09:00 hour (done at 10:00) is known.
    # 10:55 bar completes at 11:00: the 10:00 hour is now complete.
    assert list(out) == [1.0, 1.0, 2.0, 2.0]


def test_atr_table_uses_no_future_hour(monkeypatch):
    # A spike in the 11:00-12:00 hour must not change any 1h-ATR value before 11:55.
    t = minutes("2026-03-03 00:00", 2 * 24 * 60)          # a day of history first (14 hourly bars)
    mids = np.full(len(t), 1.08)
    calm = bars_1m(t, mids)
    spiky = calm.copy()
    hour11 = (t >= pd.Timestamp("2026-03-04 11:00", tz="UTC")) & (t < pd.Timestamp("2026-03-04 12:00", tz="UTC"))
    for c in ("bid_high", "offer_high"):
        spiky.loc[hour11, c] += 0.0050
    a, b = ft.atr_table(calm), ft.atr_table(spiky)
    before = a.index < pd.Timestamp("2026-03-04 11:55", tz="UTC")
    assert (a.loc[before, "atr_60"].fillna(-1) == b.loc[before, "atr_60"].fillna(-1)).all()
    assert b.loc[pd.Timestamp("2026-03-04 11:55", tz="UTC"), "atr_60"] > a.loc[pd.Timestamp("2026-03-04 11:55", tz="UTC"), "atr_60"]


def test_next_range_not_measured_across_the_weekend():
    t = minutes("2026-03-06 14:00", 8 * 60).append(minutes("2026-03-08 21:00", 4 * 60))   # Fri 14:00-21:59
    tab = ft.atr_table(bars_1m(t, [1.08] * len(t)))
    fri = tab.loc[tab.index < pd.Timestamp("2026-03-07", tz="UTC"), "next_range"]
    assert fri.iloc[0] == pytest.approx(0.0002)        # 14:00 + 4h is still Friday: measured
    assert fri.iloc[-1:].isna().all()                   # 21:55 + 4h would cross the weekend


# --------------------------------------------------------------------------
# Thin days and the report
# --------------------------------------------------------------------------
def write_history(days, thin=()):
    frames = []
    for d in days:
        n = 120 if d in thin else 24 * 60
        t = minutes(f"{d} 00:00", n)
        frames.append(bars_1m(t, 1.08 + np.random.default_rng(d.toordinal()).normal(0, 0.0002, n).cumsum()))
    allb = pd.concat(frames, ignore_index=True)
    tt = pd.to_datetime(allb["time_utc"], utc=True)
    for (y, m), part in allb.groupby([tt.dt.year, tt.dt.month]):
        fetch_data.write_parquet_atomic(part.reset_index(drop=True), fetch_data.history_bar_file("EURUSD", y, m))


def test_thin_days_found():
    days = [date(2025, 12, d) for d in (22, 23, 24, 26, 29, 30, 31)]
    write_history(days, thin={date(2025, 12, 24), date(2025, 12, 31)})
    bars = fetch_data.load_history_1m("EURUSD", datetime(2025, 12, 22, tzinfo=UTC),
                                      datetime(2026, 1, 1, tzinfo=UTC), "development")
    assert ft.thin_days(ft.atr_table(bars)) == [date(2025, 12, 24), date(2025, 12, 31)]


def test_atr_report_runs_and_shows_hours_in_utc_and_mut(capsys):
    write_history([date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5)])
    assert ft.main(["atr-report", "--from", "2025-03-03", "--to", "2025-03-06", "--save"]) == 0
    out = capsys.readouterr().out
    assert "ATR report: EURUSD" in out and "10x-spread floor = 9 pips" in out
    assert "09 UTC (13 MUT)" in out and "2025:" in out
    assert len(list((config.DATA_DIR / "reports").glob("atr_report_EURUSD_*.txt"))) == 1


def test_atr_report_respects_the_final_check_lock(capsys):
    assert ft.main(["atr-report", "--from", "2026-03-01", "--to", "2026-06-01"]) == 1
    assert "final-check" in capsys.readouterr().out
