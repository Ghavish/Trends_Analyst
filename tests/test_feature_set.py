"""
Tests for the model's features (features.compute_features, Phase 2 step 2b).

The most important one is the truncation test: features computed on data
cut off at a random decision time must EXACTLY equal the full-data values at
that time. If any feature peeked at a later bar, they would differ.
Synthetic data, temporary folders only.
"""

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import config
import features as ft
import fetch_data
import labels as lb

UTC = timezone.utc
PIP = 0.0001


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "data" / "datasets")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "DUKASCOPY_BARS_1M_DIR", tmp_path / "data" / "dukascopy" / "bars_1m")
    yield tmp_path


def market_minutes(start: str, days: int) -> pd.DatetimeIndex:
    """Minutes like a forex week: closed Friday 22:00 to Sunday 22:00 UTC."""
    t = pd.date_range(start, periods=days * 24 * 60, freq="1min", tz="UTC")
    wd, hr = t.day_of_week, t.hour
    closed = ((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22))
    return t[~closed]


def random_bars(t: pd.DatetimeIndex, seed=0, drop_share=0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = t[rng.random(len(t)) > drop_share]                    # a few quiet minutes with no bar
    close = 1.10 + np.cumsum(rng.normal(0, 0.6 * PIP, len(t)))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.1 * PIP, len(t))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5 * PIP, len(t)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5 * PIP, len(t)))
    h = 0.00001
    df = pd.DataFrame({"time_utc": t, "bid_open": open_ - h, "bid_high": high - h, "bid_low": low - h,
                       "bid_close": close - h, "offer_open": open_ + h, "offer_high": high + h,
                       "offer_low": low + h, "offer_close": close + h, "tick_count": 5,
                       "spread_mean": 2 * h, "spread_max": 2 * h, "partial": False})
    return df[fetch_data.BAR_COLUMNS]


def ohlc_bars(t, o, hi, lo, c):
    """1-minute bars that all carry the same mid OHLC (so 5-minute bars = that OHLC)."""
    h = 0.00001
    n = len(t)
    o, hi, lo, c = (np.broadcast_to(np.asarray(x, float), (n,)) for x in (o, hi, lo, c))
    df = pd.DataFrame({"time_utc": t, "bid_open": o - h, "bid_high": hi - h, "bid_low": lo - h,
                       "bid_close": c - h, "offer_open": o + h, "offer_high": hi + h,
                       "offer_low": lo + h, "offer_close": c + h, "tick_count": 5,
                       "spread_mean": 2 * h, "spread_max": 2 * h, "partial": False})
    return df[fetch_data.BAR_COLUMNS]


# One shared 33-day random history (includes weekends and Presidents' Day 2025).
HIST = random_bars(market_minutes("2025-01-27", 33), seed=7)
B1 = ft.to_indexed(HIST)
FULL, GROUPS = ft.compute_features(B1)


# --------------------------------------------------------------------------
# No lookahead: the truncation test
# --------------------------------------------------------------------------
def _cut_times():
    rng = np.random.default_rng(42)
    idx = FULL.index[FULL.index >= pd.Timestamp("2025-02-03", tz="UTC")]
    picks = list(rng.choice(idx, size=14, replace=False))
    # Plus awkward moments: just before/at an hour's end, first bars after a weekend, a holiday.
    for s in ("2025-02-26 10:50", "2025-02-26 10:55", "2025-02-24 13:00",
              "2025-02-16 22:00", "2025-02-17 14:35", "2025-02-28 21:55"):
        ts = pd.Timestamp(s, tz="UTC")
        if ts in FULL.index:
            picks.append(ts)
    return sorted(pd.Timestamp(p) for p in picks)


@pytest.mark.parametrize("t", _cut_times(), ids=lambda t: t.strftime("%m-%d_%H%M"))
def test_features_on_data_cut_at_t_equal_full_data_at_t(t):
    decision = t + pd.Timedelta(minutes=config.DECISION_BAR_MINUTES)
    cut = B1[B1.index < decision]                      # everything known when the bar completes
    part, _ = ft.compute_features(cut)
    assert part.index[-1] == t
    np.testing.assert_array_equal(part.loc[t].to_numpy(), FULL.loc[t].to_numpy(),
                                  err_msg=f"features at {t} change when later data exists")


def test_truncation_covers_every_feature_with_values():
    later = FULL[FULL.index >= pd.Timestamp("2025-02-26", tz="UTC")]
    missing = [c for c in FULL.columns if later[c].isna().all()]
    assert missing == []                               # every feature is actually exercised


# --------------------------------------------------------------------------
# Groups and specific features
# --------------------------------------------------------------------------
def test_groups():
    assert set(GROUPS) == {"base", "candles", "trend_1h"}
    assert sum(len(v) for v in GROUPS.values()) == FULL.shape[1]
    assert "minutes_to_close_by" in GROUPS["base"] and "engulfing_bull" in GROUPS["candles"]
    assert "h_ret_24" in GROUPS["trend_1h"]


@pytest.mark.parametrize("bar,london,ny,left", [
    ("2025-03-05 12:55", 13 * 60, 8 * 60, 240),        # winter: decision 13:00 UTC = 13:00 London, 08:00 NY
    ("2025-03-05 15:25", 15 * 60 + 30, 10 * 60 + 30, 150),
    ("2025-07-02 12:55", 14 * 60, 9 * 60, 240),        # summer: 14:00 London, 09:00 NY
])
def test_time_features_use_local_clocks(bar, london, ny, left):
    day = pd.Timestamp(bar).normalize().tz_localize(None)
    t = pd.date_range(day, periods=24 * 60, freq="1min", tz="UTC")
    f, _ = ft.compute_features(ft.to_indexed(ohlc_bars(t, 1.1, 1.1001, 1.0999, 1.1)))
    row = f.loc[pd.Timestamp(bar, tz="UTC")]
    assert (row["london_minute_of_day"], row["ny_minute_of_day"], row["minutes_to_close_by"]) == (london, ny, left)
    assert row["london_open"] == 1 and row["new_york_open"] == 1 and row["overlap_open"] == 1


def test_usual_uses_only_previous_days_with_data_and_skips_holidays():
    days = pd.bdate_range("2025-01-02", "2025-03-14", tz="UTC")
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=13) for d in days])
    vals = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    vals.loc[pd.Timestamp("2025-02-17 13:00", tz="UTC")] = 1000.0     # Presidents' Day: must be ignored
    u = ft._usual(vals, 20)
    assert u.iloc[:20].isna().all()
    used = [i for i, d in enumerate(days) if d.date() != date(2025, 2, 17)]
    for k in range(25, len(idx)):
        prev = [vals.iloc[i] for i in used if i < k][-20:]
        assert u.iloc[k] == pytest.approx(np.median(prev))


def test_usual_ratio_is_one_in_a_steady_market():
    t = market_minutes("2025-01-27", 40)
    f, _ = ft.compute_features(ft.to_indexed(ohlc_bars(t, 1.1, 1.1003, 1.0997, 1.1)))
    late = f[f.index >= pd.Timestamp("2025-03-03", tz="UTC")]
    assert late["atr1h_vs_usual"].dropna().round(9).eq(1).all()
    assert late["atr5_vs_usual"].dropna().round(9).eq(1).all()


def test_confirmed_swing_known_only_k_bars_later():
    idx = pd.date_range("2025-03-05 10:00", periods=12, freq="5min", tz="UTC")
    high = pd.Series([1, 2, 3, 4, 3, 2, 1, 1, 1, 1, 1, 1], index=idx, dtype=float)
    low = high - 0.5
    sh, _ = ft._confirmed_swings(high, low, 2)
    # Peak at bar 3 (value 4) beats 2 bars each side; it is known only at bar 5.
    assert sh.iloc[:5].isna().all() and sh.iloc[5] == 4


def test_candle_flags():
    t = pd.date_range("2025-03-05 10:00", periods=10, freq="1min", tz="UTC")
    o = np.r_[[1.1010] * 5, [1.0995] * 5]
    c = np.r_[[1.1000] * 5, [1.1015] * 5]               # bearish bar, then bigger bullish bar
    hi = np.r_[[1.1012] * 5, [1.1016] * 5]
    lo = np.r_[[1.0998] * 5, [1.0994] * 5]
    f, _ = ft.compute_features(ft.to_indexed(ohlc_bars(t, o, hi, lo, c)))
    second = f.loc[pd.Timestamp("2025-03-05 10:05", tz="UTC")]
    assert second["engulfing_bull"] == 1 and second["engulfing_bear"] == 0
    assert second["outside_bar"] == 1 and second["inside_bar"] == 0
    assert second["close_position"] == pytest.approx((1.1015 - 1.0994) / (1.1016 - 1.0994))


def test_pin_and_doji():
    t = pd.date_range("2025-03-05 10:00", periods=5, freq="1min", tz="UTC")
    pin = ft.compute_features(ft.to_indexed(ohlc_bars(t, 1.1008, 1.1010, 1.1000, 1.1009)))[0].iloc[0]
    assert pin["pin_bull"] == 1 and pin["pin_bear"] == 0
    doji = ft.compute_features(ft.to_indexed(ohlc_bars(t, 1.1005, 1.1010, 1.1000, 1.10052)))[0].iloc[0]
    assert doji["doji"] == 1


def test_one_hour_group_waits_for_the_hour_to_complete():
    spiked = HIST.copy()
    tt = pd.to_datetime(spiked["time_utc"], utc=True)
    hour = (tt >= pd.Timestamp("2025-02-26 10:00", tz="UTC")) & (tt < pd.Timestamp("2025-02-26 11:00", tz="UTC"))
    spiked.loc[hour, ["bid_high", "offer_high"]] += 0.0050
    f2, groups = ft.compute_features(ft.to_indexed(spiked))
    cols = groups["trend_1h"]
    before = FULL.index < pd.Timestamp("2025-02-26 10:55", tz="UTC")      # 10:55 bar completes at 11:00
    np.testing.assert_array_equal(FULL.loc[before, cols].to_numpy(), f2.loc[before, cols].to_numpy())
    at = pd.Timestamp("2025-02-26 10:55", tz="UTC")
    assert not np.array_equal(FULL.loc[at, cols].to_numpy(), f2.loc[at, cols].to_numpy())


def test_window_rows_and_warmup():
    rows = ft.window_rows(FULL)
    ready = rows.index + pd.Timedelta(minutes=5)
    mut = ready.tz_convert("Indian/Mauritius")
    assert mut.strftime("%H:%M").min() == "17:00" and mut.strftime("%H:%M").max() == "20:00"
    assert rows.index.min() >= FULL.index.min() + pd.Timedelta(days=config.FEATURE_WARMUP_DAYS)


def test_every_label_after_warmup_has_features():
    lab, _ = lb.build_labels(HIST, "EURUSD")
    rows = ft.window_rows(FULL)
    sig = pd.DatetimeIndex(pd.to_datetime(lab["signal_bar_utc"], utc=True))
    after = sig[sig >= rows.index.min()]
    assert len(after) > 0 and after.isin(rows.index).all()
    # The "time left" feature equals the labels' close-by time.
    j = lab.set_index(pd.DatetimeIndex(pd.to_datetime(lab["signal_bar_utc"], utc=True))).loc[after]
    left = (pd.to_datetime(j["close_by_utc"], utc=True) - pd.to_datetime(j["entry_utc"], utc=True)).dt.total_seconds() / 60
    assert np.array_equal(left.to_numpy(), rows.loc[after, "minutes_to_close_by"].to_numpy())


def test_build_and_describe(capsys):
    tt = pd.to_datetime(HIST["time_utc"], utc=True)
    for (y, m), part in HIST.groupby([tt.dt.year, tt.dt.month]):
        fetch_data.write_parquet_atomic(part.reset_index(drop=True), fetch_data.history_bar_file("EURUSD", y, m))
    assert ft.main(["build", "--save"]) == 0
    out = capsys.readouterr().out
    assert "[base]" in out and "[candles]" in out and "[trend_1h]" in out
    path = ft.feature_file("EURUSD")
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["end_utc"].startswith("2026-04-30") and meta["settings"]["usual_atr_days"] == 20
    saved = pd.read_parquet(path)
    assert "signal_bar_utc" in saved.columns and len(saved) == meta["rows"] > 0
    assert len(list((config.DATA_DIR / "reports").glob("features_EURUSD_*.txt"))) == 1
