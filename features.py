"""
features.py - bars, ATR and (step 2b) the model's features, calculated from
completed bars only (no lookahead).

So far (Phase 2 step 2a):
  * resample_bars()   5-, 15-, 60-minute bid/offer bars from 1-minute bars,
                      labelled by START (a 10:00 5-minute bar covers 10:00-10:05
                      and is complete at 10:05)
  * atr()             average true range (Wilder's smoothing) on mid prices;
                      across a gap (> config.GAP_MINUTES without data, e.g. a
                      weekend) the previous close is not used
  * align_completed() puts a longer timeframe's value (e.g. 1-hour ATR) next to
                      each 5-minute bar, using only longer bars already
                      COMPLETE when that 5-minute bar completes
  * the ATR report:   python features.py atr-report [--instrument EURUSD] [--save]

History is read through fetch_data.load_history_1m(), with the development
purpose, so the final-check period stays locked.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Features and reports for the model.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("atr-report", help="typical ATR by hour vs the IG spread (to choose TP / SL)")
    p.add_argument("--instrument", default="EURUSD", choices=list(config.IG_SPREAD_PIPS))
    p.add_argument("--from", dest="start", type=_parse_day, default=None,
                   help="first UTC day (default: start of the history)")
    p.add_argument("--to", dest="end", type=_parse_day, default=None,
                   help="day AFTER the last one (default: end of the development period)")
    p.add_argument("--save", action="store_true", help="also save the report in data/reports/")
    args = parser.parse_args(argv)
    start = args.start or config.HISTORY_START_UTC
    end = args.end or fetch_data.development_end_utc()
    try:
        lines = atr_report(args.instrument, start, end)
    except fetch_data.HistoryAccessError as exc:
        print(f"STOPPED: {exc}")
        return 1
    text = "\n".join(lines)
    print(text)
    if args.save:
        folder = config.DATA_DIR / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"atr_report_{args.instrument}_{timeutils.now_utc():%Y-%m-%d_%H%M}UTC.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
