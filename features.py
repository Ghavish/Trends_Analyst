"""
features.py - bars, ATR and the model's features, calculated from completed
bars only (no lookahead).

Building blocks (step 2a):
  * resample_bars()   5-, 15-, 60-minute bid/offer bars from 1-minute bars,
                      labelled by START (a 10:00 5-minute bar covers 10:00-10:05
                      and is complete at 10:05)
  * atr()             average true range (Wilder's smoothing) on mid prices;
                      across a gap (> config.GAP_MINUTES without data, e.g. a
                      weekend) the previous close is not used
  * align_completed() puts a longer timeframe's value (e.g. 1-hour ATR) next to
                      each 5-minute bar, using only longer bars already
                      COMPLETE when that 5-minute bar completes
  * close_by_times()  when a trade must be closed (shared with labels.py)

The features (step 2b), compute_features(), in three groups:
  * base      time (London / New York local clocks, weekday), minutes
              until the close-by time, volatility (5-min / 1-hour ATR, each vs
              its usual level at that time of day over the previous 20 days
              with data), momentum, distance from 20/50-bar EMAs, day structure
              (trading day from 17:00 New York)
  * candles   candle shape, the two bars before, engulfing / pin / doji /
              inside / outside flags, distance to confirmed 5-minute swings
  * trend_1h  1-hour trend from completed hours: changes over 1/4/12/24 hours,
              EMAs and slope, position in the 24-hour range, confirmed swings
Distances are in 1-hour ATRs (the unit of TP and SL). The candles and
trend_1h groups can be switched off per model variant (step 2c).

    python features.py atr-report --save   # typical ATR by hour vs the IG spread
    python features.py build --save        # EUR/USD features -> data/datasets/
    python features.py describe            # summary of the saved features

History is read through fetch_data.load_history_1m() with the development
purpose, so the final-check period stays locked.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import fetch_data
import timeutils

PRICE_COLS = [f"{side}_{f}" for side in ("bid", "offer") for f in ("open", "high", "low", "close")]


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------
def to_indexed(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """1-minute bars (time_utc column) -> indexed by bar start, sorted."""
    df = bars_1m.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("time_utc"), utc=True), name="time_utc")
    return df[~df.index.duplicated(keep="last")].sort_index()


def resample_bars(bars_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Longer bars from 1-minute bars (indexed by start). Each bar is labelled by
    its START and covers [start, start + minutes). A period with no 1-minute
    bar at all gets no bar (nothing is invented). n_minutes = how many
    1-minute bars it was built from.
    """
    r = bars_1m.resample(f"{minutes}min", label="left", closed="left", origin="epoch")
    out = pd.DataFrame({
        "bid_open": r["bid_open"].first(), "bid_high": r["bid_high"].max(),
        "bid_low": r["bid_low"].min(), "bid_close": r["bid_close"].last(),
        "offer_open": r["offer_open"].first(), "offer_high": r["offer_high"].max(),
        "offer_low": r["offer_low"].min(), "offer_close": r["offer_close"].last(),
        "tick_count": r["tick_count"].sum(), "spread_mean": r["spread_mean"].mean(),
        "spread_max": r["spread_max"].max(), "n_minutes": r["bid_close"].count(),
    })
    return out[out["n_minutes"] > 0]


def mid(bars: pd.DataFrame) -> pd.DataFrame:
    """Mid-price OHLC, (bid + offer) / 2. High/low are the average of the bid
    and offer highs/lows: close to, but not exactly, the true mid high/low."""
    return pd.DataFrame({
        "open": (bars["bid_open"] + bars["offer_open"]) / 2,
        "high": (bars["bid_high"] + bars["offer_high"]) / 2,
        "low": (bars["bid_low"] + bars["offer_low"]) / 2,
        "close": (bars["bid_close"] + bars["offer_close"]) / 2,
    }, index=bars.index)


def follows_previous(index: pd.DatetimeIndex, minutes: int) -> pd.Series:
    """True where the previous bar ended no more than GAP_MINUTES before this one started."""
    t = pd.Series(index, index=index)
    return t.diff() <= pd.Timedelta(minutes=minutes + config.GAP_MINUTES)


def atr(m: pd.DataFrame, minutes: int, n: int | None = None) -> pd.Series:
    """
    Average true range of mid bars `m` (Wilder's smoothing: each new value =
    previous + (true range - previous) / n). The true range uses the previous
    close only if there is no gap in between; after a gap it is high - low.
    Look-backs count bars, so a weekend is simply skipped. NaN until n bars exist.
    """
    n = n or config.ATR_BARS
    prev = m["close"].shift(1).where(follows_previous(m.index, minutes))
    tr = pd.concat([m["high"] - m["low"], (m["high"] - prev).abs(), (m["low"] - prev).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def align_completed(values: pd.Series, values_minutes: int, base_index: pd.DatetimeIndex,
                    base_minutes: int) -> pd.Series:
    """
    For each base bar (e.g. 5-minute), the latest `values` (e.g. 1-hour ATR)
    whose bar is COMPLETE when the base bar completes. A 10:00-11:00 hourly
    bar is complete at 11:00, so it is first used by the 10:55 5-minute bar
    (complete at 11:00), never by 10:50. No lookahead.
    """
    left = pd.DataFrame({"ready": base_index + pd.Timedelta(minutes=base_minutes)})
    right = pd.DataFrame({"ready": values.index + pd.Timedelta(minutes=values_minutes),
                          "value": values.to_numpy()}).dropna()
    merged = pd.merge_asof(left, right, on="ready", direction="backward")
    return pd.Series(merged["value"].to_numpy(), index=base_index, name=values.name)


def ns(times) -> np.ndarray:
    """UTC times -> int64 nanoseconds (pandas may store other units; never mix them)."""
    return pd.DatetimeIndex(times).as_unit("ns").asi8


def local_time_utc(index: pd.DatetimeIndex, tz: str, hm: tuple[int, int]) -> pd.DatetimeIndex:
    """For each UTC time, that calendar day's hh:mm in zone `tz`, as UTC."""
    local = pd.DatetimeIndex(index).tz_convert(tz)
    return (local.normalize() + pd.Timedelta(hours=hm[0], minutes=hm[1])).tz_convert("UTC")


def close_by_times(entries) -> pd.DatetimeIndex:
    """When a trade entered at each time must be closed: the earliest of entry +
    MAX_HOLD_HOURS, CLOSE_BY_MUT and ROLLOVER_CUTOFF_NY (shared by labels and features)."""
    e = pd.DatetimeIndex(entries)
    t = np.minimum.reduce([ns(e + pd.Timedelta(hours=config.MAX_HOLD_HOURS)),
                           ns(local_time_utc(e, config.TZ_MAURITIUS, config.CLOSE_BY_MUT)),
                           ns(local_time_utc(e, config.TZ_NEW_YORK, config.ROLLOVER_CUTOFF_NY))])
    return pd.DatetimeIndex(pd.to_datetime(t, utc=True))


def in_entry_window(ready: pd.DatetimeIndex) -> np.ndarray:
    """True where a bar completing at `ready` is inside your entry window (MUT)."""
    mut = pd.DatetimeIndex(ready).tz_convert(config.TZ_MAURITIUS)
    hm = mut.hour * 60 + mut.minute
    first = config.ENTRY_FIRST_MUT[0] * 60 + config.ENTRY_FIRST_MUT[1]
    last = config.ENTRY_LAST_MUT[0] * 60 + config.ENTRY_LAST_MUT[1]
    return np.asarray((hm >= first) & (hm <= last))


# --------------------------------------------------------------------------
# Features (Phase 2 step 2b) - every value uses only bars complete at the
# decision time (the close of the 5-minute signal bar). A truncation test
# (tests/test_features.py) checks this: features computed on data cut off at
# a random time equal the full-data values at that time.
# --------------------------------------------------------------------------
FEATURES_VERSION = 2               # 2: five redundant time features dropped (15 Nov 2026)


def _usual(values: pd.Series, n_days: int) -> pd.Series:
    """
    Median of `values` at the same time of day (5-minute slot, UTC) over the
    previous n_days days that have a value there; no-entry days (holidays,
    24 Dec - 1 Jan) don't count. Today is never included. NaN until n_days exist.
    """
    idx = values.index
    day = idx.normalize()
    slot = idx.hour * 60 + idx.minute
    days = pd.DatetimeIndex(day.unique()).sort_values()
    holiday = np.array([timeutils.no_entry_reason(d.date()) is not None for d in days])
    table = (pd.DataFrame({"day": day, "slot": slot, "v": values.to_numpy()})
             .pivot(index="day", columns="slot", values="v").reindex(days))   # one row per day
    table[holiday] = np.nan
    usual = np.full(table.shape, np.nan)
    for j, col in enumerate(table.columns):
        x = table[col].dropna()                                    # days with a value in this slot
        med = x.rolling(n_days, min_periods=n_days).median()       # includes that day ...
        usual[:, j] = med.reindex(days).ffill().shift(1).to_numpy()   # ... so use it from the next day
    return pd.Series(usual[days.get_indexer(day), table.columns.get_indexer(slot)], index=idx)


def _confirmed_swings(high: pd.Series, low: pd.Series, k: int) -> tuple[pd.Series, pd.Series]:
    """
    Latest confirmed swing high and low. A swing high is a bar whose high is
    above the k bars before AND the k bars after it, so it is only known k
    bars later - and that is when it is used (no lookahead).
    """
    def side(x: pd.Series, sign: int) -> pd.Series:
        y = sign * x
        before = y.shift(1).rolling(k, min_periods=k).max()
        after = y[::-1].shift(1).rolling(k, min_periods=k).max()[::-1]
        is_swing = (y > before) & (y > after)
        return x.where(is_swing).shift(k).ffill()
    return side(high, 1), side(low, -1)


def _by_atr(x, atr_1h):
    return x / atr_1h


def compute_features(b1: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Features for every 5-minute bar (indexed by bar START; the decision is
    made when the bar completes). b1: 1-minute bars indexed by start.
    Returns (features, {group: [columns]}): groups "base", "candles", "trend_1h".
    """
    minutes = config.DECISION_BAR_MINUTES
    base = resample_bars(b1, minutes)
    m = mid(base)
    idx = base.index
    ready = idx + pd.Timedelta(minutes=minutes)
    o, h, l, c = m["open"], m["high"], m["low"], m["close"]
    hourly = mid(resample_bars(b1, 60))
    atr_hourly = atr(hourly, 60)
    a1 = align_completed(atr_hourly, 60, idx, minutes)          # the unit for all distances
    a5 = atr(m, minutes)
    f = pd.DataFrame(index=idx)
    groups: dict[str, list[str]] = {"base": [], "candles": [], "trend_1h": []}

    def add(group, name, values):
        f[name] = np.asarray(values, dtype=float)
        groups[group].append(name)

    # ---- time and sessions (market-local clocks) ----
    lon = ready.tz_convert(config.TZ_LONDON)
    nyc = ready.tz_convert(config.TZ_NEW_YORK)
    add("base", "london_minute_of_day", lon.hour * 60 + lon.minute)
    add("base", "ny_minute_of_day", nyc.hour * 60 + nyc.minute)
    add("base", "weekday", ready.tz_convert(config.TZ_MAURITIUS).day_of_week)
    # (Session flags and "minutes since the open" were dropped in version 2:
    # inside your window both sessions are always open, and minutes since the
    # open only repeat the local clock.)
    add("base", "minutes_to_close_by", (ns(close_by_times(ready)) - ns(ready)) / 60e9)

    # ---- volatility ----
    add("base", "atr5_to_atr1h", a5 / a1)
    add("base", "atr1h_vs_usual", a1 / _usual(a1, config.USUAL_ATR_DAYS))
    add("base", "atr5_vs_usual", a5 / _usual(a5, config.USUAL_ATR_DAYS))
    add("base", "bar_range_atr", _by_atr(h - l, a1))

    # ---- momentum ----
    for n in config.MOMENTUM_BARS:
        add("base", f"ret_{n}", _by_atr(c - c.shift(n), a1))
    for span in config.EMA_SPANS:
        add("base", f"dist_ema{span}", _by_atr(c - c.ewm(span=span, adjust=False).mean(), a1))

    # ---- day structure (the trading day starts at 17:00 New York) ----
    tday = (idx.tz_convert(config.TZ_NEW_YORK) - pd.Timedelta(hours=config.TRADING_DAY_RESET_HOUR)
            ).tz_localize(None).normalize()
    g = m.groupby(tday)
    day_open = g["open"].transform("first")
    day_hi, day_lo = g["high"].cummax(), g["low"].cummin()
    daily = g.agg(high=("high", "max"), low=("low", "min"))
    prev = daily.shift(1).reindex(tday)
    add("base", "dist_day_open", _by_atr(c - day_open, a1))
    add("base", "dist_prev_day_high", _by_atr(c - prev["high"].to_numpy(), a1))
    add("base", "dist_prev_day_low", _by_atr(c - prev["low"].to_numpy(), a1))
    add("base", "day_range_position", (c - day_lo) / (day_hi - day_lo).replace(0, np.nan))

    # ---- candlestick group (switchable) ----
    rng = (h - l).replace(0, np.nan)
    body = c - o
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    add("candles", "body_atr", _by_atr(body, a1))
    add("candles", "upper_wick_atr", _by_atr(upper, a1))
    add("candles", "lower_wick_atr", _by_atr(lower, a1))
    add("candles", "close_position", (c - l) / rng)
    for lag in (1, 2):
        add("candles", f"body_atr_lag{lag}", _by_atr(body, a1).shift(lag))
        add("candles", f"close_position_lag{lag}", ((c - l) / rng).shift(lag))
    pb, po, pc, ph, pl = body.shift(1), o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    add("candles", "engulfing_bull", (pb < 0) & (body > 0) & (o <= pc) & (c >= po))
    add("candles", "engulfing_bear", (pb > 0) & (body < 0) & (o >= pc) & (c <= po))
    absb = body.abs()
    add("candles", "pin_bull", (lower >= config.PIN_MIN_WICK_TO_BODY * absb) & (lower >= config.PIN_MIN_WICK_SHARE * rng))
    add("candles", "pin_bear", (upper >= config.PIN_MIN_WICK_TO_BODY * absb) & (upper >= config.PIN_MIN_WICK_SHARE * rng))
    add("candles", "doji", absb <= config.DOJI_MAX_BODY_SHARE * rng)
    add("candles", "inside_bar", (h < ph) & (l > pl))
    add("candles", "outside_bar", (h > ph) & (l < pl))
    sw_hi, sw_lo = _confirmed_swings(h, l, config.SWING_BARS_5M)
    add("candles", "dist_swing_high", _by_atr(c - sw_hi, a1))
    add("candles", "dist_swing_low", _by_atr(c - sw_lo, a1))

    # ---- 1-hour trend group (switchable; completed hours only) ----
    hc = hourly["close"]
    hf = pd.DataFrame(index=hourly.index)
    for n in config.TREND_1H_HOURS:
        hf[f"h_ret_{n}"] = (hc - hc.shift(n)) / atr_hourly
    emas = {span: hc.ewm(span=span, adjust=False).mean() for span in config.EMA_SPANS}
    for span, e in emas.items():
        hf[f"h_dist_ema{span}"] = (hc - e) / atr_hourly
    fast = emas[config.EMA_SPANS[0]]
    hf[f"h_ema{config.EMA_SPANS[0]}_slope"] = (fast - fast.shift(1)) / atr_hourly
    n = config.RANGE_1H_HOURS
    hi_n = hourly["high"].rolling(n, min_periods=n).max()
    lo_n = hourly["low"].rolling(n, min_periods=n).min()
    hf[f"h_range{n}_position"] = (hc - lo_n) / (hi_n - lo_n).replace(0, np.nan)
    hsw_hi, hsw_lo = _confirmed_swings(hourly["high"], hourly["low"], config.SWING_BARS_1H)
    hf["h_dist_swing_high"] = (hc - hsw_hi) / atr_hourly
    hf["h_dist_swing_low"] = (hc - hsw_lo) / atr_hourly
    for col in hf.columns:
        add("trend_1h", col, align_completed(hf[col], 60, idx, minutes))
    return f, groups


def feature_file(instrument: str) -> Path:
    return config.DATASETS_DIR / f"{instrument}_features_v{FEATURES_VERSION}.parquet"


def feature_settings() -> dict:
    names = ("DECISION_BAR_MINUTES", "ATR_BARS", "GAP_MINUTES", "USUAL_ATR_DAYS", "MOMENTUM_BARS",
             "EMA_SPANS", "SWING_BARS_5M", "SWING_BARS_1H", "TREND_1H_HOURS", "RANGE_1H_HOURS",
             "DOJI_MAX_BODY_SHARE", "PIN_MIN_WICK_TO_BODY", "PIN_MIN_WICK_SHARE",
             "FEATURE_WARMUP_DAYS", "ENTRY_FIRST_MUT", "ENTRY_LAST_MUT", "CLOSE_BY_MUT",
             "MAX_HOLD_HOURS", "ROLLOVER_CUTOFF_NY", "TRADING_DAY_RESET_HOUR")
    return {"features_version": FEATURES_VERSION, **{k.lower(): getattr(config, k) for k in names}}


def window_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Rows whose bar completes inside your entry window, after the warm-up."""
    ready = feats.index + pd.Timedelta(minutes=config.DECISION_BAR_MINUTES)
    start = feats.index.min() + pd.Timedelta(days=config.FEATURE_WARMUP_DAYS)
    keep = in_entry_window(ready) & np.asarray(feats.index >= start)
    out = feats[keep].copy()
    out.index.name = "signal_bar_utc"
    return out


def build_and_save(instrument: str) -> dict:
    import json
    start, end = config.HISTORY_START_UTC, fetch_data.development_end_utc()
    bars = fetch_data.load_history_1m(instrument, start, end, fetch_data.PURPOSE_DEVELOPMENT)
    feats, groups = compute_features(to_indexed(bars))
    rows = window_rows(feats)
    path = feature_file(instrument)
    fetch_data.write_parquet_atomic(rows.reset_index(), path, compression="zstd")
    meta = {"settings": feature_settings(), "groups": groups, "instrument": instrument,
            "start_utc": start.isoformat(), "end_utc": end.isoformat(), "rows": len(rows),
            "created_utc": timeutils.now_utc().isoformat()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=list), encoding="utf-8")
    return meta


def describe(instrument: str) -> list[str]:
    import json
    path = feature_file(instrument)
    if not path.exists():
        return ["No features yet: run  python features.py build"]
    df = pd.read_parquet(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    lines = [f"Features: {instrument}, {len(df):,} rows (entry window only, after "
             f"{config.FEATURE_WARMUP_DAYS} days of warm-up), version {meta['settings']['features_version']}",
             "Distances in 1-hour ATRs. 'present' = share of rows with a value.", ""]
    for group, cols in meta["groups"].items():
        lines.append(f"[{group}] {len(cols)} features")
        lines.append(f"  {'feature':<26} {'present':>8} {'5%':>9} {'median':>9} {'95%':>9}")
        for col in cols:
            x = df[col]
            q = x.quantile([0.05, 0.5, 0.95])
            lines.append(f"  {col:<26} {100 * x.notna().mean():>7.1f}% {q[0.05]:>9.3g} {q[0.5]:>9.3g} {q[0.95]:>9.3g}")
        lines.append("")
    return lines


# --------------------------------------------------------------------------
# ATR report (to choose the TP / SL distances, question 2 of step 2)
# --------------------------------------------------------------------------
ATR_TIMEFRAMES = (5, 15, 60)       # minutes: the decision bars and two longer options
FORWARD_HOURS = 4                  # "how far price travels next" window (the max holding time)


def atr_table(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """One row per 5-minute bar: ATR(14) of 5-, 15- and 60-minute bars (completed
    bars only), the 1-minute bar count, and the high-low range of the NEXT 4 hours
    (future data: used only in this report, never as a feature)."""
    b1 = to_indexed(bars_1m)
    base = resample_bars(b1, config.DECISION_BAR_MINUTES)
    m5 = mid(base)
    out = pd.DataFrame(index=base.index)
    out["ticks"] = base["tick_count"]
    for tf in ATR_TIMEFRAMES:
        if tf == config.DECISION_BAR_MINUTES:
            out[f"atr_{tf}"] = atr(m5, tf)
        else:
            m = mid(resample_bars(b1, tf))
            out[f"atr_{tf}"] = align_completed(atr(m, tf), tf, base.index, config.DECISION_BAR_MINUTES)
    n_fwd = FORWARD_HOURS * 60 // config.DECISION_BAR_MINUTES
    rev_hi = m5["high"][::-1].rolling(n_fwd, min_periods=n_fwd).max()[::-1].shift(-1)
    rev_lo = m5["low"][::-1].rolling(n_fwd, min_periods=n_fwd).min()[::-1].shift(-1)
    span = pd.Series(m5.index, index=m5.index).shift(-n_fwd) - m5.index.to_series()
    ok = span <= pd.Timedelta(hours=FORWARD_HOURS, minutes=config.GAP_MINUTES)   # no weekend inside
    out["next_range"] = (rev_hi - rev_lo).where(ok)
    return out


def thin_days(table: pd.DataFrame, share: float = 0.4) -> list:
    """Weekdays (UTC) whose tick count is below `share` of that year's median weekday."""
    daily = table["ticks"].groupby(table.index.date).sum()
    daily.index = pd.to_datetime(daily.index)
    daily = daily[daily.index.day_of_week < 5]
    med = daily.groupby(daily.index.year).transform("median")
    return [d.date() for d in daily[daily < share * med].index]


def atr_report(instrument: str, start: datetime, end: datetime) -> list[str]:
    bars = fetch_data.load_history_1m(instrument, start, end, fetch_data.PURPOSE_DEVELOPMENT)
    if bars.empty:
        return [f"No {instrument} bars between {start:%Y-%m-%d} and {end:%Y-%m-%d}."]
    pip = config.PIP_SIZE[instrument]
    spread = config.IG_SPREAD_PIPS[instrument]
    floor = 10 * spread
    t = atr_table(bars)
    p = t[[c for c in t.columns if c.startswith("atr_") or c == "next_range"]] / pip
    lines = [f"ATR report: {instrument}, {start:%Y-%m-%d} to {end:%Y-%m-%d} (not included), "
             f"{len(t):,} five-minute bars",
             f"All figures in pips (1 pip = {pip}). IG spread assumed {spread} pips; "
             f"10x-spread floor = {floor:g} pips. ATR = {config.ATR_BARS} bars, Wilder's smoothing, mid prices.",
             "",
             "Typical (median) values by hour of day. 'Spread % of 1xATR' = how much of a 1xATR",
             "stop the spread alone costs. 'Floor binds' = share of bars where 1xATR < the floor.",
             "'Next 4h range' = how far price travelled (high - low) in the following 4 hours.",
             ""]
    head = (f"  {'Hour':<17}| {'ATR 5m':>7} {'15m':>6} {'1h':>6} | {'Spread % of 1xATR':^20} | "
            f"{'Floor binds':^17} | {'Next 4h':>7}")
    lines += [head, f"  {'':<17}| {'':>7} {'':>6} {'':>6} | {'5m':>6} {'15m':>6} {'1h':>6} | "
                    f"{'5m':>5} {'15m':>5} {'1h':>5} | {'range':>7}", "  " + "-" * (len(head) - 2)]
    by_hour = p.groupby(p.index.hour)
    med = by_hour.median()
    for h in range(24):
        if h not in med.index:
            continue
        row = med.loc[h]
        g = by_hour.get_group(h)
        pct = [100 * spread / row[f"atr_{tf}"] for tf in ATR_TIMEFRAMES]
        binds = [100 * (g[f"atr_{tf}"] < floor).mean() for tf in ATR_TIMEFRAMES]
        lines.append(f"  {h:02d} UTC ({(h + 4) % 24:02d} MUT) | {row['atr_5']:>7.1f} {row['atr_15']:>6.1f} "
                     f"{row['atr_60']:>6.1f} | {pct[0]:>5.0f}% {pct[1]:>5.0f}% {pct[2]:>5.0f}% | "
                     f"{binds[0]:>4.0f}% {binds[1]:>4.0f}% {binds[2]:>4.0f}% | {row['next_range']:>7.1f}")
    lines += ["", "Typical (median) ATR by year, all hours:"]
    by_year = p.groupby(p.index.year).median()
    for y, row in by_year.iterrows():
        lines.append(f"  {y}:  5m {row['atr_5']:.1f}   15m {row['atr_15']:.1f}   1h {row['atr_60']:.1f}   "
                     f"next 4h range {row['next_range']:.1f}")
    thin = thin_days(t)
    lines += ["", f"Thin weekdays (ticks below 40% of that year's usual weekday): {len(thin)}"]
    by_y: dict[int, list] = {}
    for d in thin:
        by_y.setdefault(d.year, []).append(d)
    for y, ds in sorted(by_y.items()):
        lines.append(f"  {y}: " + ", ".join(f"{d:%d %b}" for d in ds))
    gaps = (~follows_previous(t.index, config.DECISION_BAR_MINUTES)).sum() - 1
    lines += ["", f"Gaps of more than {config.GAP_MINUTES} min (weekends included): {gaps:,}",
              "Hours are UTC, so with daylight saving a session shifts by one hour between "
              "winter and summer."]
    return lines


def _parse_day(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _save_report(text: str, name: str) -> None:
    folder = config.DATA_DIR / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{timeutils.now_utc():%Y-%m-%d_%H%M}UTC.txt"
    path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Features and reports for the model.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("atr-report", help="typical ATR by hour vs the IG spread (to choose TP / SL)")
    p.add_argument("--from", dest="start", type=_parse_day, default=None,
                   help="first UTC day (default: start of the history)")
    p.add_argument("--to", dest="end", type=_parse_day, default=None,
                   help="day AFTER the last one (default: end of the development period)")
    for name, text in (("build", "make the features (development period) and describe them"),
                       ("describe", "describe the saved features")):
        sub.add_parser(name, help=text)
    for sp in sub.choices.values():
        sp.add_argument("--instrument", default="EURUSD", choices=list(config.IG_SPREAD_PIPS))
        sp.add_argument("--save", action="store_true", help="also save the report in data/reports/")
    args = parser.parse_args(argv)
    try:
        if args.command == "atr-report":
            start = args.start or config.HISTORY_START_UTC
            end = args.end or fetch_data.development_end_utc()
            lines = atr_report(args.instrument, start, end)
            name = f"atr_report_{args.instrument}"
        else:
            if args.command == "build":
                meta = build_and_save(args.instrument)
                print(f"Saved {meta['rows']:,} rows to {feature_file(args.instrument)}\n")
            lines = describe(args.instrument)
            name = f"features_{args.instrument}"
    except fetch_data.HistoryAccessError as exc:
        print(f"STOPPED: {exc}")
        return 1
    text = "\n".join(lines)
    print(text)
    if args.save:
        _save_report(text, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
