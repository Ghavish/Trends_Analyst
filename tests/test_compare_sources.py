"""
Tests for compare_sources.py on synthetic data with KNOWN differences:
IG = Dukascopy + a fixed level gap, a different spread, optionally a delay.
Temporary folders only.
"""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import compare_sources as cs
import config
import fetch_data

UTC = timezone.utc
DAY1, DAY2 = date(2026, 11, 10), date(2026, 11, 11)


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "BARS_1M_DIR", tmp_path / "data" / "bars_1m")
    monkeypatch.setattr(config, "DUKASCOPY_BARS_1M_DIR", tmp_path / "data" / "dukascopy" / "bars_1m")
    monkeypatch.setattr(fetch_data, "default_history_end", lambda now=None: datetime(2026, 11, 12, tzinfo=UTC))
    yield tmp_path


def make_bars(mid: pd.Series, spread: float, partial=None) -> pd.DataFrame:
    """1-minute bid/offer bars around a mid-price path (index = bar start)."""
    rng = np.random.default_rng(1)
    wick = np.abs(rng.normal(0, 0.00005, len(mid)))
    hi, lo = mid + wick, mid - wick
    half = spread / 2
    return pd.DataFrame({
        "time_utc": mid.index,
        "bid_open": mid - half, "bid_high": hi - half, "bid_low": lo - half, "bid_close": mid - half,
        "offer_open": mid + half, "offer_high": hi + half, "offer_low": lo + half, "offer_close": mid + half,
        "tick_count": 5, "spread_mean": spread, "spread_max": spread,
        "partial": partial if partial is not None else False,
    })[fetch_data.BAR_COLUMNS].reset_index(drop=True)


def price_path(days, seed=0) -> pd.Series:
    idx = pd.DatetimeIndex([], tz="UTC")
    for d in days:
        idx = idx.append(pd.date_range(f"{d} 00:00", f"{d} 21:59", freq="1min", tz="UTC"))
    steps = np.random.default_rng(seed).normal(0, 0.0001, len(idx))
    return pd.Series(1.08 + np.cumsum(steps), index=idx)


def write_ig(inst, bars):
    t = pd.to_datetime(bars["time_utc"], utc=True)
    for d, part in bars.groupby(t.dt.date):
        fetch_data.write_parquet_atomic(part.reset_index(drop=True), fetch_data.bar_file(inst, d))


def write_duka(inst, bars):
    t = pd.to_datetime(bars["time_utc"], utc=True)
    for (y, m), part in bars.groupby([t.dt.year, t.dt.month]):
        fetch_data.write_parquet_atomic(part.reset_index(drop=True), fetch_data.history_bar_file(inst, y, m))


def setup_pair(delay_minutes=0, level=0.00020, ig_spread=0.00006, dk_spread=0.00002, partial_first=0):
    dk_mid = price_path([DAY1, DAY2])
    ig_mid = dk_mid.shift(delay_minutes).dropna() + level      # IG lags by delay_minutes
    partial = np.zeros(len(ig_mid), dtype=bool)
    partial[:partial_first] = True
    write_duka("EURUSD", make_bars(dk_mid, dk_spread))
    write_ig("EURUSD", make_bars(ig_mid, ig_spread, partial))
    return dk_mid


def compare():
    return cs.compare_instrument("EURUSD", datetime(2026, 11, 10, tzinfo=UTC), datetime(2026, 11, 12, tzinfo=UTC))


# --------------------------------------------------------------------------
def test_identical_timing_level_gap_and_spreads_are_measured():
    setup_pair(level=0.00020)
    res = compare()
    assert res["common_bars"] == 2 * 22 * 60
    assert max(res["lags"], key=res["lags"].get) == 0
    assert res["lags"][0] > 0.99
    assert res["level"]["median"] == pytest.approx(0.00020, abs=1e-9)
    assert res["atr_ratio"]["median"] == pytest.approx(1.0, abs=0.01)
    assert res["range_ratio"]["median"] == pytest.approx(1.0, abs=0.01)
    assert res["corr_5m"] > 0.99
    assert res["spread_ig_all"] == pytest.approx(0.00006)
    assert res["spread_dk_all"] == pytest.approx(0.00002)
    assert res["hours"].loc[22:].sum().sum() == 0          # no bars 22:00-23:59 in this data
    assert res["hours"].loc[9, "both"] == 2 * 60


def test_a_one_minute_delay_is_found():
    setup_pair(delay_minutes=1)
    res = compare()
    assert max(res["lags"], key=res["lags"].get) == 1          # IG behind by 1 minute
    text = "\n".join(cs.format_report(res, pip=0.0001))
    assert "IG 1 minute(s) behind Dukascopy" in text


def test_partial_ig_bars_are_left_out():
    setup_pair(partial_first=30)
    res = compare()
    assert res["ig_partial"] == 30
    assert res["common_bars"] == 2 * 22 * 60 - 30


def test_report_in_ig_pips_with_mauritius_time():
    setup_pair(level=0.00020)
    text = "\n".join(cs.format_report(compare(), pip=0.0001))
    assert "IG pips" in text and "median 2," in text           # 0.0002 = 2 pips
    assert "09 UTC (13 MUT)" in text
    assert "in step (as expected)" in text


def test_missing_dukascopy_days_are_named_with_the_fix():
    dk_mid = price_path([DAY1])
    write_duka("EURUSD", make_bars(dk_mid, 0.00002))
    write_ig("EURUSD", make_bars(price_path([DAY1, DAY2]), 0.00006))
    text = "\n".join(cs.format_report(compare(), pip=None))
    assert "no bars yet for IG day(s): 2026-11-11" in text
    assert "download --instruments EURUSD --from 2026-11-11" in text


def test_no_overlap_yet_message():
    write_ig("EURUSD", make_bars(price_path([date(2026, 11, 12)]), 0.00006))
    text = "\n".join(cs.run(["EURUSD"], None, None))
    assert "No overlap yet" in text and "2026-11-16" in text


def test_nothing_recorded_message():
    assert "No IG recordings" in cs.run(["EURUSD"], None, None)[0]


def test_final_check_period_stays_locked():
    text = "\n".join(cs.run(["EURUSD"], datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 5, tzinfo=UTC)))
    assert "not compared" in text and "final-check" in text


def test_indices_can_be_compared_although_not_approved_for_the_model():
    dk_mid = price_path([DAY1, DAY2]) * 5000
    write_duka("US500", make_bars(dk_mid, 0.5))
    write_ig("US500", make_bars(dk_mid + 1.0, 0.4))
    res = cs.compare_instrument("US500", datetime(2026, 11, 10, tzinfo=UTC), datetime(2026, 11, 12, tzinfo=UTC))
    assert res["level"]["median"] == pytest.approx(1.0)


def test_cli_saves_report(tmp_path, capsys):
    setup_pair()
    assert cs.main(["--instruments", "EURUSD", "--save"]) == 0
    out = capsys.readouterr().out
    assert "Saved to" in out
    assert len(list((tmp_path / "data" / "reports").glob("compare_sources_*.txt"))) == 1
