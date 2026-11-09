"""
compare_sources.py - compares Dukascopy's history with the prices you record
from IG, minute by minute, on the days both have (read only).

Why (Sections 4.2, 6.2, 11.3): the model trains on Dukascopy but trades on
IG. This shows whether the two agree well enough - above all for US 500 and
Germany 40, which stay out of the model until you have seen these results
(config.HISTORY_APPROVED_FOR_MODEL). The numbers are for you to judge; the
script decides nothing.

    python compare_sources.py                          # all instruments, every overlapping day
    python compare_sources.py --instruments US500 GER40
    python compare_sources.py --from 2026-11-09 --to 2026-11-20 --save

What is measured (on mid prices, (bid + offer) / 2, of complete 1-minute bars;
IG bars marked partial are left out):
  1. Time alignment - how well 1-minute moves match when one source is
     shifted by -3..+3 minutes. Both are meant to be UTC, so the best match
     should be at 0.
  2. Price level - IG minus Dukascopy at the same minute: typical gap, its
     spread across minutes, and by hour of day and by day (does it drift?).
  3. Movement - 5-minute bars built from the 1-minute bars of each source:
     how closely their moves agree, and IG's bar range and ATR divided by
     Dukascopy's (TP and SL are set from ATR, so this matters most).
  4. Spreads - typical spread per hour of day in each source.
  5. Trading hours - 1-minute bars per hour of day in each source. Note: IG
     sends a price only when it changes, so a quiet minute can have no IG bar
     without being a gap.

Dukascopy bars are read through fetch_data.load_history_1m() with the
source-comparison purpose, so the final-check period stays locked.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import db
import fetch_data
import timeutils

LAGS = range(-3, 4)          # minutes tried for the time alignment
ATR_BARS = 14                # ATR length for this comparison only (a common default;
                             # the model's own ATR length is chosen in Phase 2 step 2)


def reports_dir() -> Path:
    return config.DATA_DIR / "reports"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def ig_days(instrument: str) -> list[date]:
    """UTC days with a recorded IG bar file."""
    folder = config.BARS_1M_DIR / instrument
    if not folder.exists():
        return []
    days = []
    for f in folder.glob("*.parquet"):
        try:
            days.append(datetime.strptime(f.stem, "%Y-%m-%d").date())
        except ValueError:
            continue
    return sorted(days)


def load_ig(instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Recorded IG 1-minute bars with start <= time_utc < end (all, including partial)."""
    frames = [fetch_data.read_bars(instrument, d) for d in ig_days(instrument)
              if start.date() <= d < end.date() + timedelta(days=1)]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=fetch_data.BAR_COLUMNS)
    bars = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(bars["time_utc"], utc=True)
    return bars[(t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))].reset_index(drop=True)


def mid_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Mid-price OHLC and mean spread, indexed by bar start (UTC).
    high/low use the average of the bid and offer highs/lows - close to, but
    not exactly, the true mid high/low."""
    if bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "spread"],
                            index=pd.DatetimeIndex([], tz="UTC", name="time_utc"))
    idx = pd.DatetimeIndex(pd.to_datetime(bars["time_utc"], utc=True), name="time_utc")
    out = pd.DataFrame({
        "open": ((bars["bid_open"] + bars["offer_open"]) / 2).to_numpy(float),
        "high": ((bars["bid_high"] + bars["offer_high"]) / 2).to_numpy(float),
        "low": ((bars["bid_low"] + bars["offer_low"]) / 2).to_numpy(float),
        "close": ((bars["bid_close"] + bars["offer_close"]) / 2).to_numpy(float),
        "spread": bars["spread_mean"].to_numpy(float),
    }, index=idx)
    return out[~out.index.duplicated(keep="last")].sort_index()


def five_minute(m: pd.DataFrame) -> pd.DataFrame:
    """5-minute bars from 1-minute bars, labelled by START (10:00 = 10:00-10:05)."""
    if m.empty:
        return m[["open", "high", "low", "close"]]
    r = m.resample("5min", label="left", closed="left")
    out = pd.DataFrame({"open": r["open"].first(), "high": r["high"].max(),
                        "low": r["low"].min(), "close": r["close"].last()})
    return out.dropna()


def atr(b: pd.DataFrame, n: int = ATR_BARS) -> pd.Series:
    prev = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - prev).abs(), (b["low"] - prev).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(n).mean()


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------
def lag_correlations(ig: pd.Series, dk: pd.Series) -> dict[int, float]:
    """
    Correlation of 1-minute mid moves when IG is compared with Dukascopy
    `lag` minutes EARLIER (lag > 0: IG is behind Dukascopy). Each source's
    price is carried forward over at most 5 quiet minutes (IG only sends
    changes), never across longer gaps.
    """
    if ig.empty or dk.empty:
        return {}
    grid = pd.date_range(min(ig.index.min(), dk.index.min()), max(ig.index.max(), dk.index.max()),
                         freq="1min")
    r_ig = ig.reindex(grid).ffill(limit=5).diff()
    r_dk = dk.reindex(grid).ffill(limit=5).diff()
    out = {}
    for lag in LAGS:
        pair = pd.concat([r_ig, r_dk.shift(lag)], axis=1).dropna()
        pair = pair[(pair.iloc[:, 0] != 0) | (pair.iloc[:, 1] != 0)]   # ignore minutes neither moved
        if len(pair) >= 30 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
            out[lag] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return out


def _q(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "median": float(s.median()),
            "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95))}


def compare_instrument(instrument: str, start: datetime, end: datetime) -> dict:
    ig_all = load_ig(instrument, start, end)
    ig_partial = int(ig_all["partial"].astype(bool).sum()) if len(ig_all) else 0
    ig_ok = ig_all[~ig_all["partial"].astype(bool)] if len(ig_all) else ig_all
    dk_all = fetch_data.load_history_1m(instrument, start, end, fetch_data.PURPOSE_SOURCE_COMPARISON)
    ig, dk = mid_bars(ig_ok), mid_bars(dk_all)
    res: dict = {"instrument": instrument, "start": start, "end": end,
                 "ig_bars": len(ig_all), "ig_partial": ig_partial, "dk_bars": len(dk),
                 "ig_days": sorted({t.date() for t in ig.index}),
                 "dk_days": sorted({t.date() for t in dk.index})}
    common = ig.index.intersection(dk.index)
    res["common_bars"] = len(common)
    res["common_days"] = sorted({t.date() for t in common})
    # 5. hours (computed even without overlap: shows what each source has)
    hours = pd.DataFrame({
        "ig": pd.Series(1, index=ig.index).groupby(ig.index.hour).sum() if len(ig) else pd.Series(dtype=int),
        "dk": pd.Series(1, index=dk.index).groupby(dk.index.hour).sum() if len(dk) else pd.Series(dtype=int),
        "both": pd.Series(1, index=common).groupby(common.hour).sum() if len(common) else pd.Series(dtype=int),
    }).reindex(range(24)).fillna(0).astype(int)
    res["hours"] = hours
    res["only_ig"] = len(ig.index.difference(dk.index))
    res["only_dk"] = len(dk.index.difference(ig.index))
    if len(common) == 0:
        return res
    # 1. time alignment
    res["lags"] = lag_correlations(ig["close"], dk["close"])
    # 2. price level
    diff = ig.loc[common, "close"] - dk.loc[common, "close"]
    res["level"] = _q(diff)
    res["level_rel"] = _q(diff / dk.loc[common, "close"])
    res["level_by_hour"] = diff.groupby(diff.index.hour).median()
    res["level_by_day"] = diff.groupby(diff.index.date).median()
    # 3. movement on 5-minute bars
    ig5, dk5 = five_minute(ig), five_minute(dk)
    c5 = ig5.index.intersection(dk5.index)
    r = pd.concat([ig5["close"].diff(), dk5["close"].diff()], axis=1, sort=True).loc[c5].dropna()
    res["corr_5m"] = float(r.iloc[:, 0].corr(r.iloc[:, 1])) if len(r) >= 30 else None
    rng_dk = (dk5["high"] - dk5["low"]).loc[c5]
    rng_ig = (ig5["high"] - ig5["low"]).loc[c5]
    res["range_ratio"] = _q((rng_ig / rng_dk.replace(0, np.nan)))
    res["atr_ratio"] = _q((atr(ig5) / atr(dk5).replace(0, np.nan)).loc[c5])
    res["bars_5m"] = len(c5)
    # 4. spreads
    res["spread_ig"] = ig["spread"].groupby(ig.index.hour).median()
    res["spread_dk"] = dk["spread"].groupby(dk.index.hour).median()
    res["spread_ig_all"] = float(ig["spread"].median())
    res["spread_dk_all"] = float(dk["spread"].median())
    return res


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _hour(h: int) -> str:
    return f"{h:02d} UTC ({(h + 4) % 24:02d} MUT)"          # Mauritius is UTC+4 all year


def _fmt(x: float | None, unit: tuple[str, float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x / unit[1]:.4g}"


def format_report(res: dict, pip: float | None) -> list[str]:
    unit = ("IG pips", pip) if pip else ("price units", 1.0)
    inst = res["instrument"]
    lines = [f"{inst}  ({config.INSTRUMENTS[inst]['name']} - IG vs Dukascopy {config.DUKASCOPY_SYMBOLS[inst]})",
             f"  Range {res['start']:%Y-%m-%d} to {res['end']:%Y-%m-%d} (not included). "
             f"IG: {res['ig_bars']:,} bars on {len(res['ig_days'])} day(s), {res['ig_partial']:,} partial left out. "
             f"Dukascopy: {res['dk_bars']:,} bars on {len(res['dk_days'])} day(s)."]
    missing = sorted(set(res["ig_days"]) - set(res["dk_days"]))
    if missing:
        lines.append(f"  Dukascopy has no bars yet for IG day(s): {', '.join(d.isoformat() for d in missing)}"
                     f"  -> python fetch_data.py download --instruments {inst} --from {missing[0].isoformat()}")
    lines.append(f"  Minutes in both: {res['common_bars']:,} on {len(res['common_days'])} day(s); "
                 f"only IG: {res['only_ig']:,}; only Dukascopy: {res['only_dk']:,}")
    if res["common_bars"] == 0:
        lines.append("  Nothing to compare yet.")
        return lines + [""]
    lags = res.get("lags") or {}
    if lags:
        best = max(lags, key=lags.get)
        where = ("in step (as expected)" if best == 0 else
                 f"IG {abs(best)} minute(s) {'behind' if best > 0 else 'ahead of'} Dukascopy - check timestamps")
        lines.append(f"  1. Time alignment: best match at {best:+d} min -> {where}")
        lines.append("     correlation of 1-min moves by shift: "
                     + ", ".join(f"{k:+d}: {v:.3f}" for k, v in sorted(lags.items())))
    else:
        lines.append("  1. Time alignment: too few moving minutes to measure")
    lv, rel = res["level"], res["level_rel"]
    lines.append(f"  2. Price level, IG minus Dukascopy ({unit[0]}): median {_fmt(lv['median'], unit)}, "
                 f"90% of minutes between {_fmt(lv['p05'], unit)} and {_fmt(lv['p95'], unit)} "
                 f"(median {rel['median'] * 100:+.4f}% of price)")
    by_day = res["level_by_day"]
    if len(by_day) > 1:
        lines.append("     by day: " + ", ".join(f"{d:%m-%d} {_fmt(v, unit)}" for d, v in by_day.items()))
    corr = res["corr_5m"]
    lines.append(f"  3. Movement on {res['bars_5m']:,} common 5-min bars: correlation of moves "
                 f"{corr:.3f}" if corr is not None else f"  3. Movement: too few common 5-min bars ({res['bars_5m']})")
    rr, ar = res["range_ratio"], res["atr_ratio"]
    if rr.get("n"):
        lines.append(f"     IG / Dukascopy bar range: median {rr['median']:.3f} (90%: {rr['p05']:.3f}-{rr['p95']:.3f})")
    if ar.get("n"):
        lines.append(f"     IG / Dukascopy ATR({ATR_BARS} x 5 min): median {ar['median']:.3f} "
                     f"(90%: {ar['p05']:.3f}-{ar['p95']:.3f})")
    lines.append(f"  4. Spread ({unit[0]}), median of all minutes: IG {_fmt(res['spread_ig_all'], unit)}, "
                 f"Dukascopy {_fmt(res['spread_dk_all'], unit)}")
    lines.append("  4-5. By hour of day          IG bars  Duka bars  in both | IG spread  Duka spread | IG-Duka level")
    for h, row in res["hours"].iterrows():
        if row["ig"] == 0 and row["dk"] == 0:
            continue
        lines.append(f"     {_hour(h)}   {row['ig']:>8,} {row['dk']:>9,} {row['both']:>8,} | "
                     f"{_fmt(res['spread_ig'].get(h), unit):>9}  {_fmt(res['spread_dk'].get(h), unit):>11} | "
                     f"{_fmt(res['level_by_hour'].get(h), unit):>13}")
    return lines + [""]


def default_range(instruments: list[str]) -> tuple[datetime, datetime] | None:
    """From the first recorded IG day to the newest day Dukascopy can have."""
    firsts = [ig_days(i)[0] for i in instruments if ig_days(i)]
    if not firsts:
        return None
    first = min(firsts)
    start = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
    return start, fetch_data.default_history_end()


def run(instruments: list[str], start: datetime | None, end: datetime | None) -> list[str]:
    rng = default_range(instruments)
    if start is None or end is None:
        if rng is None:
            return ["No IG recordings found in data/bars_1m yet."]
        start, end = start or rng[0], end or rng[1]
    if start >= end:
        first_possible = start + timedelta(days=config.DUKASCOPY_LAG_DAYS + 1)
        return [f"No overlap yet: IG recording starts {start:%Y-%m-%d}, and Dukascopy days are "
                f"downloaded {config.DUKASCOPY_LAG_DAYS} days late. The first day can be compared "
                f"from {first_possible:%Y-%m-%d}."]
    info = db.get_market_info()
    now = timeutils.now_utc()
    lines = [f"Source comparison made {timeutils.fmt_dual(now, with_date=True)}", ""]
    for inst in instruments:
        try:
            res = compare_instrument(inst, start, end)
        except fetch_data.HistoryAccessError as exc:
            lines += [f"{inst}: not compared - {exc}", ""]
            continue
        lines += format_report(res, (info.get(inst) or {}).get("pip_size"))
    lines.append("Hours are the hour of day of the bar start. IG sends a price only when it changes, "
                 "so a quiet minute can have no IG bar.")
    return lines


def _parse_day(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Dukascopy history with your IG recordings.")
    parser.add_argument("--instruments", nargs="+", choices=list(config.INSTRUMENTS), default=None)
    parser.add_argument("--from", dest="start", type=_parse_day, default=None,
                        help="first UTC day, YYYY-MM-DD (default: first IG recording day)")
    parser.add_argument("--to", dest="end", type=_parse_day, default=None,
                        help="day AFTER the last one (default: newest day Dukascopy can have)")
    parser.add_argument("--save", action="store_true", help="also save the report in data/reports/")
    args = parser.parse_args(argv)
    lines = run(args.instruments or list(config.INSTRUMENTS), args.start, args.end)
    text = "\n".join(lines)
    print(text)
    if args.save:
        reports_dir().mkdir(parents=True, exist_ok=True)
        path = reports_dir() / f"compare_sources_{timeutils.now_utc():%Y-%m-%d_%H%M}UTC.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
