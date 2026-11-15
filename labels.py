"""
labels.py - triple-barrier labels (plan Section 7.1): for every possible
entry in your trading window, did a BUY (and, separately, a SELL) hit its
take profit before its stop loss, or did it time out?

Rules (decided 15 Nov 2026, settings in config.py; fixed, not tuned):
  * Decision bars: 5 minutes. A signal comes when a bar completes; the trade
    enters at the next bar's open (the same instant: a 10:00 bar completes
    and the trade opens at 10:05).
  * Entries only 17:00-20:00 MUT (13:00-16:00 UTC all year); none on days
    from timeutils.no_entry_reason() (24 Dec - 1 Jan, NYSE / London holidays,
    NYSE early closes); none in the first BARS_AFTER_GAP_NO_TRADE bars after
    a gap or the weekend; none before the 1-hour ATR exists.
  * Prices: Dukascopy mid plus IG's spread (config.IG_SPREAD_PIPS). A buy
    enters at the offer and is exited at the bid; a sell the other way round.
  * SL = 1 x ATR of completed 1-hour bars, at least STOP_FLOOR_SPREADS IG
    spreads; TP = 1.5 x the SL distance. Measured from the entry price.
  * Time limit: the earlier of entry + 4 hours, 22:00 MUT and 16:45 New York.
  * Exits are checked on 1-minute bars. If TP and SL are both reached in the
    same minute, SL is assumed (the cautious choice). A stop is filled
    SL_SLIPPAGE_PIPS worse (or at the minute's open if price gapped past it).
    A timed-out trade closes at the last minute's close.
  * Result in R (1 R = the SL distance): TP = +1.5, SL = about -1 minus slippage,
    timeout = whatever it was. win = 1 only for TP.

Only completed data before the entry is used for the barriers; the path
after the entry is used only to find the outcome (that is what a label is).

    python labels.py build      # EUR/USD, all development history -> data/datasets/
    python labels.py report     # outcome rates by entry hour and year, skipped entries
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import features as ft
import fetch_data
import timeutils

LABELS_VERSION = 1
CHUNK = 4000                     # entries processed at a time (keeps memory small)
SIDES = ("buy", "sell")


def label_file(instrument: str) -> Path:
    return config.DATASETS_DIR / f"{instrument}_labels_v{LABELS_VERSION}.parquet"


def settings(instrument: str) -> dict:
    """Everything that decides the labels; saved next to the file."""
    return {
        "labels_version": LABELS_VERSION, "instrument": instrument,
        "decision_bar_minutes": config.DECISION_BAR_MINUTES,
        "barrier_atr_minutes": config.BARRIER_ATR_MINUTES, "atr_bars": config.ATR_BARS,
        "sl_atr_multiple": config.SL_ATR_MULTIPLE, "tp_atr_multiple": config.TP_ATR_MULTIPLE,
        "stop_floor_spreads": config.STOP_FLOOR_SPREADS, "max_hold_hours": config.MAX_HOLD_HOURS,
        "entry_first_mut": config.ENTRY_FIRST_MUT, "entry_last_mut": config.ENTRY_LAST_MUT,
        "close_by_mut": config.CLOSE_BY_MUT, "rollover_cutoff_ny": config.ROLLOVER_CUTOFF_NY,
        "ig_spread_pips": config.IG_SPREAD_PIPS[instrument],
        "sl_slippage_pips": config.SL_SLIPPAGE_PIPS[instrument],
        "pip_size": config.PIP_SIZE[instrument], "gap_minutes": config.GAP_MINUTES,
        "bars_after_gap_no_trade": config.BARS_AFTER_GAP_NO_TRADE,
        "skip_year_end": config.SKIP_YEAR_END, "skip_exchange_holidays": config.SKIP_EXCHANGE_HOLIDAYS,
        "skip_nyse_early_closes": config.SKIP_NYSE_EARLY_CLOSES,
    }


# --------------------------------------------------------------------------
# Candidate entries
# --------------------------------------------------------------------------
_ns = ft.ns


def candidates(b1: pd.DataFrame, instrument: str) -> tuple[pd.DataFrame, dict]:
    """
    Possible entries (one per 5-minute bar completing inside the window) with
    their barriers and close-by time, plus counts of entries skipped and why.
    b1: 1-minute bars indexed by start (features.to_indexed).
    """
    minutes = config.DECISION_BAR_MINUTES
    base = ft.resample_bars(b1, minutes)
    entry = base.index + pd.Timedelta(minutes=minutes)          # signal bar completes = entry time
    entry_mut = entry.tz_convert(config.TZ_MAURITIUS)
    hm = entry_mut.hour * 60 + entry_mut.minute
    first = config.ENTRY_FIRST_MUT[0] * 60 + config.ENTRY_FIRST_MUT[1]
    last = config.ENTRY_LAST_MUT[0] * 60 + config.ENTRY_LAST_MUT[1]
    in_window = np.asarray((hm >= first) & (hm <= last))

    hourly = ft.mid(ft.resample_bars(b1, config.BARRIER_ATR_MINUTES))
    atr_h = ft.align_completed(ft.atr(hourly, config.BARRIER_ATR_MINUTES), config.BARRIER_ATR_MINUTES,
                               base.index, minutes).to_numpy()
    # Bars since the last gap (0 = the first bar after a gap or the weekend).
    new_run = (~ft.follows_previous(base.index, minutes)).to_numpy()
    since_gap = pd.Series(np.ones(len(base), int)).groupby(np.cumsum(new_run)).cumsum().to_numpy() - 1

    reasons = np.zeros(len(base), bool)                       # only looked up inside the window
    days = entry_mut.date
    reasons[in_window] = [timeutils.no_entry_reason(d) is not None for d in days[in_window]]
    has_entry_price = np.asarray(entry.isin(b1.index))

    skipped = {}
    keep = in_window.copy()
    for name, bad in (("holiday or year-end", reasons),
                      ("1-hour ATR not available yet", np.isnan(atr_h)),
                      (f"within {config.BARS_AFTER_GAP_NO_TRADE} bars after a gap or weekend",
                       since_gap < config.BARS_AFTER_GAP_NO_TRADE),
                      ("no price at the entry minute", ~has_entry_price)):
        hit = keep & bad
        skipped[name] = int(hit.sum())
        keep &= ~bad

    pip = config.PIP_SIZE[instrument]
    floor = config.STOP_FLOOR_SPREADS * config.IG_SPREAD_PIPS[instrument] * pip
    e = entry[keep]
    atr_k = atr_h[keep]
    sl_dist = np.maximum(config.SL_ATR_MULTIPLE * atr_k, floor)
    close_by = ft.close_by_times(e)
    cand = pd.DataFrame({
        "signal_bar_utc": base.index[keep],
        "entry_utc": e,
        "close_by_utc": close_by,
        "atr_1h": atr_k,
        "sl_dist": sl_dist,
        "tp_dist": sl_dist * config.TP_ATR_MULTIPLE / config.SL_ATR_MULTIPLE,
        "floor_applied": config.SL_ATR_MULTIPLE * atr_k < floor,
    })
    return cand.reset_index(drop=True), skipped


# --------------------------------------------------------------------------
# Outcomes on 1-minute bars
# --------------------------------------------------------------------------
def outcomes(b1: pd.DataFrame, cand: pd.DataFrame, instrument: str) -> tuple[pd.DataFrame, dict]:
    """Walk each candidate forward minute by minute (vectorised, in chunks)."""
    pip = config.PIP_SIZE[instrument]
    s = config.IG_SPREAD_PIPS[instrument] * pip
    slip = config.SL_SLIPPAGE_PIPS[instrument] * pip
    m = ft.mid(b1)
    t = _ns(b1.index)
    mo, mh, ml, mc = (m[c].to_numpy() for c in ("open", "high", "low", "close"))
    n = len(t)
    k = config.MAX_HOLD_HOURS * 60 + 1
    step = np.arange(k)
    gap_ns = (1 + config.GAP_MINUTES) * 60 * 10**9
    out_rows = []
    skipped = {"gap in the data during the trade": 0}
    for c0 in range(0, len(cand), CHUNK):
        c = cand.iloc[c0:c0 + CHUNK]
        pos = np.searchsorted(t, _ns(c["entry_utc"]))
        idx = np.minimum(pos[:, None] + step, n - 1)
        inside = (pos[:, None] + step < n)
        cb = _ns(c["close_by_utc"])
        tt = t[idx]
        valid = inside & (tt < cb[:, None])
        # The data must be continuous from entry to close-by (no gap > GAP_MINUTES).
        dt = np.diff(np.where(valid, tt, np.nan), axis=1)
        broken = np.nanmax(np.where(np.isnan(dt), 0, dt), axis=1, initial=0) > gap_ns
        last_j = valid.sum(axis=1) - 1
        last_t = tt[np.arange(len(c)), np.maximum(last_j, 0)]
        broken |= (cb - last_t) > gap_ns
        sl_d, tp_d = c["sl_dist"].to_numpy(), c["tp_dist"].to_numpy()
        res = {}
        for side in SIDES:
            sign = 1.0 if side == "buy" else -1.0
            entry = mo[pos] + sign * s / 2
            tp = entry + sign * tp_d
            sl = entry - sign * sl_d
            # Price at which this side would close: the bid for a buy, the offer for a sell.
            exit_hi = mh[idx] - sign * s / 2
            exit_lo = ml[idx] - sign * s / 2
            if side == "buy":
                hit_tp, hit_sl = valid & (exit_hi >= tp[:, None]), valid & (exit_lo <= sl[:, None])
            else:
                hit_tp, hit_sl = valid & (exit_lo <= tp[:, None]), valid & (exit_hi >= sl[:, None])
            j_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), k)
            j_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), k)
            is_sl = (j_sl < k) & (j_sl <= j_tp)
            is_tp = (j_tp < k) & ~is_sl
            ambiguous = (j_sl < k) & (j_sl == j_tp)
            rows = np.arange(len(c))
            j_exit = np.where(is_sl, j_sl, np.where(is_tp, j_tp, np.maximum(last_j, 0)))
            ix = idx[rows, j_exit]
            open_exit = mo[ix] - sign * s / 2
            sl_fill = (np.minimum(sl, open_exit) if side == "buy" else np.maximum(sl, open_exit)) - sign * slip
            timeout_px = mc[ix] - sign * s / 2
            exit_px = np.where(is_sl, sl_fill, np.where(is_tp, tp, timeout_px))
            exit_t = np.where(is_sl | is_tp, t[ix], cb)
            res[f"{side}_entry"] = entry
            res[f"{side}_tp"] = tp
            res[f"{side}_sl"] = sl
            res[f"{side}_outcome"] = np.where(is_sl, "SL", np.where(is_tp, "TP", "TIMEOUT"))
            res[f"{side}_exit_utc"] = pd.to_datetime(exit_t, utc=True)
            res[f"{side}_exit_price"] = exit_px
            res[f"{side}_result_r"] = sign * (exit_px - entry) / sl_d
            res[f"{side}_win"] = is_tp.astype(np.int8)
            res[f"{side}_same_minute"] = ambiguous
        part = pd.DataFrame(res, index=c.index)
        skipped["gap in the data during the trade"] += int(broken.sum())
        out_rows.append(pd.concat([c, part], axis=1)[~broken])
    if not out_rows:
        return pd.DataFrame(), skipped
    return pd.concat(out_rows).reset_index(drop=True), skipped


# --------------------------------------------------------------------------
# Build, save, report
# --------------------------------------------------------------------------
def build_labels(bars_1m: pd.DataFrame, instrument: str) -> tuple[pd.DataFrame, dict]:
    b1 = ft.to_indexed(bars_1m)
    cand, skipped = candidates(b1, instrument)
    labels, skipped2 = outcomes(b1, cand, instrument)
    return labels, {**skipped, **skipped2}


def build_and_save(instrument: str, start: datetime | None = None, end: datetime | None = None) -> dict:
    start = start or config.HISTORY_START_UTC
    end = end or fetch_data.development_end_utc()
    bars = fetch_data.load_history_1m(instrument, start, end, fetch_data.PURPOSE_DEVELOPMENT)
    labels, skipped = build_labels(bars, instrument)
    path = label_file(instrument)
    fetch_data.write_parquet_atomic(labels, path, compression="zstd")
    meta = {"settings": settings(instrument), "start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "rows": len(labels), "skipped": skipped, "created_utc": timeutils.now_utc().isoformat()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=list), encoding="utf-8")
    return meta


def _rates(df: pd.DataFrame, side: str) -> str:
    n = len(df)
    if n == 0:
        return "-"
    o = df[f"{side}_outcome"].value_counts()
    return (f"{n:>6,}  TP {100 * o.get('TP', 0) / n:4.1f}%  SL {100 * o.get('SL', 0) / n:4.1f}%  "
            f"timeout {100 * o.get('TIMEOUT', 0) / n:4.1f}%  avg {df[f'{side}_result_r'].mean():+.3f} R")


def report(instrument: str) -> list[str]:
    path = label_file(instrument)
    if not path.exists():
        return [f"No labels yet: run  python labels.py build"]
    df = pd.read_parquet(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    st = meta["settings"]
    pip = st["pip_size"]
    entry_mut = pd.to_datetime(df["entry_utc"], utc=True).dt.tz_convert(config.TZ_MAURITIUS)
    lines = [f"Labels: {instrument}, {meta['start_utc'][:10]} to {meta['end_utc'][:10]} (not included), "
             f"{meta['rows']:,} entries, made {meta['created_utc'][:16]} UTC",
             f"SL = {st['sl_atr_multiple']} x 1-hour ATR({st['atr_bars']}), at least {st['stop_floor_spreads']} x "
             f"{st['ig_spread_pips']}-pip spread; TP = {st['tp_atr_multiple'] / st['sl_atr_multiple']:g} x SL; "
             f"stop slippage {st['sl_slippage_pips']} pips; entries 17:00-20:00 MUT, closed by 22:00 MUT.",
             f"Stop distance (pips): median {df['sl_dist'].median() / pip:.1f}, 10% {df['sl_dist'].quantile(0.1) / pip:.1f}, "
             f"90% {df['sl_dist'].quantile(0.9) / pip:.1f}; floor applied on {100 * df['floor_applied'].mean():.1f}% of entries.",
             "Break-even win rate (TP = +1.5 R, SL about -1 R): about 40%, before timeouts.", ""]
    for side in SIDES:
        lines.append(f"{side.upper()}  all: {_rates(df, side)}   "
                     f"(TP and SL in the same minute, counted as SL: {int(df[f'{side}_same_minute'].sum()):,})")
        for h, g in df.groupby(entry_mut.dt.hour):
            lines.append(f"     entry {h:02d}:xx MUT ({(h - 4) % 24:02d} UTC): {_rates(g, side)}")
        lines.append("")
    lines.append("By year (buy | sell):")
    for y, g in df.groupby(pd.to_datetime(df["entry_utc"], utc=True).dt.year):
        lines.append(f"  {y}: {_rates(g, 'buy')}  |  {_rates(g, 'sell')}")
    lines += ["", "Entries skipped (inside the window):"]
    lines += [f"  {k}: {v:,}" for k, v in meta["skipped"].items()]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Triple-barrier labels.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("build", "make the labels (development period) and show the report"),
                       ("report", "show the report for the saved labels")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--instrument", default="EURUSD", choices=list(config.IG_SPREAD_PIPS))
        p.add_argument("--save", action="store_true", help="also save the report in data/reports/")
    args = parser.parse_args(argv)
    if args.command == "build":
        try:
            meta = build_and_save(args.instrument)
        except fetch_data.HistoryAccessError as exc:
            print(f"STOPPED: {exc}")
            return 1
        print(f"Saved {meta['rows']:,} labels to {label_file(args.instrument)}\n")
    text = "\n".join(report(args.instrument))
    print(text)
    if args.save:
        folder = config.DATA_DIR / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"labels_report_{args.instrument}_{timeutils.now_utc():%Y-%m-%d_%H%M}UTC.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
