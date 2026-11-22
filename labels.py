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
    python labels.py daily-build --instrument EURUSD   # Phase 2B daily-decision labels + report
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


# ==========================================================================
# Phase 2B: one decision per instrument per day (PROPOSAL_PHASE_2B.md, section 8)
# ==========================================================================
# Each weekday Monday-Thursday at 17:15 MUT: a buy and, separately, a sell,
# held to the next weekday's decision time with a protective stop at 1 x daily
# ATR(14) and a take profit at 3 x daily ATR(14). label (win) = 1 if the
# result in R after spread, slippage and funding is above 0. Exits are checked
# on 1-minute bars by backtest.simulate (night spreads and slippage, funding
# per rollover); a trade open at the close time exits at the first tradable
# price at or after it, so gaps count. Daily ATR uses completed trading days
# (17:00 New York) only.
DAILY_LABELS_VERSION = 1


def daily_label_file(instrument: str) -> Path:
    return config.DATASETS_DIR / f"{instrument}_daily_labels_v{DAILY_LABELS_VERSION}.parquet"


def daily_close_by(entries) -> pd.DatetimeIndex:
    """The next weekday's decision time (also on a holiday), as UTC."""
    mut = pd.DatetimeIndex(entries).tz_convert(config.TZ_MAURITIUS)
    d = mut.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
    nxt = np.busday_offset(d, 1, roll="forward")
    h, m = config.DAILY_DECISION_MUT
    return (pd.DatetimeIndex(nxt).tz_localize(config.TZ_MAURITIUS)
            + pd.Timedelta(hours=h, minutes=m)).tz_convert("UTC")


def daily_settings(instrument: str) -> dict:
    return {"daily_labels_version": DAILY_LABELS_VERSION, "instrument": instrument,
            "decision_mut": config.DAILY_DECISION_MUT, "entry_weekdays": config.DAILY_ENTRY_WEEKDAYS,
            "sl_atr": config.DAILY_SL_ATR, "tp_atr": config.DAILY_TP_ATR, "atr_bars": config.ATR_BARS,
            "max_gap_minutes": config.DAILY_MAX_GAP_MINUTES, "funding": config.FUNDING[instrument],
            "slippage": config.COST_SLIPPAGE[instrument]}


def build_daily_labels(b1: pd.DataFrame, instrument: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """(one row per decision with buy_/sell_ columns, skipped counts, spread table).
    b1: 1-minute bars indexed by start (features.to_indexed)."""
    import backtest as bt
    table = bt.spread_table(instrument, b1)
    spread_min = bt.spread_per_minute(b1, table)
    slip_min = bt.slippage_per_minute(instrument, b1)
    minutes = config.DECISION_BAR_MINUTES
    base = ft.resample_bars(b1, minutes)
    entry = base.index + pd.Timedelta(minutes=minutes)
    mut = entry.tz_convert(config.TZ_MAURITIUS)
    h, m = config.DAILY_DECISION_MUT
    at_time = np.asarray((mut.hour == h) & (mut.minute == m) & mut.day_of_week.isin(config.DAILY_ENTRY_WEEKDAYS))
    holiday = np.zeros(len(base), bool)
    holiday[at_time] = [timeutils.no_entry_reason(d) is not None for d in mut.date[at_time]]
    atr_d = bt.atr_at_decision(b1, base.index, "D")
    has_price = np.asarray(entry.isin(b1.index))
    keep, skipped = at_time.copy(), {}
    for name, bad in (("holiday or year-end", holiday), ("daily ATR not available yet", np.isnan(atr_d)),
                      ("no price at the decision minute", ~has_price)):
        skipped[name] = int((keep & bad).sum())
        keep &= ~bad
    e = entry[keep]
    a = atr_d[keep]
    pos = np.searchsorted(ft.ns(b1.index), ft.ns(e))
    cand = pd.DataFrame({"entry_utc": e, "close_by_utc": daily_close_by(e), "atr": a,
                         "sl_dist": config.DAILY_SL_ATR * a, "tp_dist": config.DAILY_TP_ATR * a,
                         "floor_applied": False, "spread_entry": spread_min[pos]})
    trades, dropped = bt.simulate(b1, cand, instrument, spread_min, slip_min, config.DAILY_MAX_GAP_MINUTES,
                                  exit_after_close=True)
    skipped["no price after the close time"] = dropped
    if len(trades) == 0:
        return pd.DataFrame(), skipped, table
    trades["win"] = (trades["result_r"] > 0).astype(np.int8)
    cols = ["outcome", "result_r", "win", "spread_r", "slippage_r", "funding_r", "nights", "hours_held"]
    wide = trades.pivot(index="entry_utc", columns="side", values=cols)
    wide.columns = [f"{side}_{c}" for c, side in wide.columns]
    wide = wide.reset_index()
    info = cand.set_index("entry_utc").loc[wide["entry_utc"]]
    wide["close_by_utc"] = info["close_by_utc"].to_numpy()
    wide["atr_d"] = info["atr"].to_numpy()
    wide["sl_dist"] = info["sl_dist"].to_numpy()
    wide["decision_bar_utc"] = pd.to_datetime(wide["entry_utc"], utc=True) - pd.Timedelta(minutes=minutes)
    wide["day"] = pd.to_datetime(wide["entry_utc"], utc=True).dt.tz_convert(None).dt.normalize()
    for side in SIDES:
        wide[f"{side}_win"] = wide[f"{side}_win"].astype(np.int8)
        for c in ("result_r", "spread_r", "slippage_r", "funding_r", "nights", "hours_held"):
            wide[f"{side}_{c}"] = wide[f"{side}_{c}"].astype(float)
    return wide.sort_values("entry_utc").reset_index(drop=True), skipped, table


def daily_build_and_save(instrument: str) -> dict:
    if not config.HISTORY_APPROVED_OVERNIGHT.get(instrument, False):
        raise fetch_data.HistoryAccessError(
            f"{instrument} is not cleared for overnight holding yet (config.HISTORY_APPROVED_OVERNIGHT): "
            "a week of overnight recording and a night-hours comparison come first")
    start, end = config.HISTORY_START_UTC, fetch_data.development_end_utc()
    bars = fetch_data.load_history_1m(instrument, start, end, fetch_data.PURPOSE_DEVELOPMENT)
    wide, skipped, _ = build_daily_labels(ft.to_indexed(bars), instrument)
    path = daily_label_file(instrument)
    fetch_data.write_parquet_atomic(wide, path, compression="zstd")
    meta = {"settings": daily_settings(instrument), "start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "rows": len(wide), "skipped": skipped, "created_utc": timeutils.now_utc().isoformat()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=list), encoding="utf-8")
    return meta


def daily_rounds():
    import train as tr
    return tr.schedule(pd.Timestamp(fetch_data.development_end_utc()), config.DAILY_WF_FIRST_TEST_START,
                       config.WF_PERIOD_MONTHS)


def daily_baselines(df: pd.DataFrame, rounds=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two bars the model must clear, per walk-forward test period, both
    fixed from the data before the test period (training + calibration, after
    the 1-day gaps), as the model sees it:
    - same direction: always trade the direction with the higher average R
      before the test period; its test results (periods, and the trades for the
      day-resampled interval);
    - log loss: a constant win probability per direction = the win rate before
      the test period; its test log loss.
    df: daily labels with a 'day' column (one instrument)."""
    import train as tr
    rows, parts = [], []
    for rnd in rounds or daily_rounds():
        train_rows, cal, test = tr.split(df, rnd)
        before = pd.concat([train_rows, cal])
        if len(test) == 0 or len(before) == 0:
            continue
        side = "buy" if before["buy_result_r"].mean() >= before["sell_result_r"].mean() else "sell"
        other = "sell" if side == "buy" else "buy"
        row = {"round": rnd.number, "period": rnd.label(), "side": side, "decisions": len(test),
               "avg_r": float(test[f"{side}_result_r"].mean()),
               "other_avg_r": float(test[f"{other}_result_r"].mean())}
        for s_ in SIDES:
            rate = float(before[f"{s_}_win"].mean())
            row[f"{s_}_base_rate"] = rate
            row[f"{s_}_base_log_loss"] = tr.log_loss(test[f"{s_}_win"].to_numpy(), np.full(len(test), rate))
        rows.append(row)
        parts.append(pd.DataFrame({"round": rnd.number, "day": test["day"].to_numpy(), "side": side,
                                   "result_r": test[f"{side}_result_r"].to_numpy()}))
    trades = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=["round", "day", "side", "result_r"]))
    return pd.DataFrame(rows), trades


def daily_report(instrument: str) -> list[str]:
    import backtest as bt
    import train as tr
    path = daily_label_file(instrument)
    if not path.exists():
        return [f"No daily labels yet: run  python labels.py daily-build --instrument {instrument}"]
    df = pd.read_parquet(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    unit = "pips" if instrument in config.PIP_SIZE else "points"
    p = config.PIP_SIZE.get(instrument, 1.0)
    days = pd.to_datetime(df["day"])
    h, m = config.DAILY_DECISION_MUT
    lines = [f"Daily labels: {instrument}, {days.min():%Y-%m-%d} to {days.max():%Y-%m-%d}, {len(df):,} decisions "
             f"(Monday-Thursday {h:02d}:{m:02d} MUT), made {meta['created_utc'][:16]} UTC",
             f"Held to the next weekday's {h:02d}:{m:02d} MUT; stop {config.DAILY_SL_ATR:g} x / take profit "
             f"{config.DAILY_TP_ATR:g} x daily ATR(14); win = result after spread, slippage and funding above 0.",
             f"Stop distance ({unit}): median {df['sl_dist'].median() / p:.1f}, 10% {df['sl_dist'].quantile(0.1) / p:.1f}, "
             f"90% {df['sl_dist'].quantile(0.9) / p:.1f}.  Decisions per year: "
             f"{days.dt.normalize().nunique() / max((days.max() - days.min()).days / 365.25, 1e-9):.0f}", ""]
    lines.append("No model, every decision taken (both directions):")
    for side in SIDES:
        t = pd.DataFrame({"day": df["day"], "result_r": df[f"{side}_result_r"],
                          "spread_r": df[f"{side}_spread_r"], "slippage_r": df[f"{side}_slippage_r"],
                          "funding_r": df[f"{side}_funding_r"]})
        lo, hi = tr.bootstrap_mean_r(t)
        mv = bt.cost_vs_move(t)
        o = df[f"{side}_outcome"].value_counts(normalize=True)
        lines.append(f"  {side.upper():<4} win {100 * df[f'{side}_win'].mean():5.1f}%  avg {t['result_r'].mean():+.3f} R "
                     f"[{lo:+.3f}, {hi:+.3f}]  TP/SL/timeout {100 * o.get('TP', 0):.0f}/{100 * o.get('SL', 0):.0f}/"
                     f"{100 * o.get('TIMEOUT', 0):.0f}%  held {df[f'{side}_hours_held'].mean():.1f} h  "
                     f"nights {df[f'{side}_nights'].mean():.2f}")
        lines.append(f"       costs per trade: spread {t['spread_r'].mean():.3f}  slippage {t['slippage_r'].mean():.3f}  "
                     f"funding {t['funding_r'].mean():.3f}  = {mv['cost_r']:.3f} R; sd {mv['sd_r']:.3f}; "
                     f"cost/sd {mv['cost_sd']:.3f}")
    lines += ["", "By year (decisions, buy avg R, sell avg R):"]
    for y, g in df.groupby(days.dt.year):
        lines.append(f"  {y}: {len(g):>4}  buy {g['buy_result_r'].mean():+.3f}  sell {g['sell_result_r'].mean():+.3f}")
    lines += ["", "By decision weekday (buy | sell avg R; Wednesday holds pay the triple-night funding for forex):"]
    wd = pd.to_datetime(df["entry_utc"], utc=True).dt.tz_convert(config.TZ_MAURITIUS).dt.day_of_week
    for d, g in df.groupby(wd):
        lines.append(f"  {('Mon', 'Tue', 'Wed', 'Thu', 'Fri')[d]}: {len(g):>4}  {g['buy_result_r'].mean():+.3f} | "
                     f"{g['sell_result_r'].mean():+.3f}")
    base, bt_trades = daily_baselines(df)
    lines += ["", "Baselines the model must beat, per walk-forward test period (fixed from the data before it:",
              "training + calibration, 1-day gaps). Same direction = always trade the direction with the higher",
              "average R before the period. Log loss = a constant win rate per direction from before the period."]
    if len(base):
        for _, r in base.iterrows():
            lines.append(f"  round {int(r['round']):>2} ({r['period']}): {r['side']:<4} {int(r['decisions']):>4} decisions  "
                         f"avg {r['avg_r']:+.3f} R (other {r['other_avg_r']:+.3f})   log loss buy "
                         f"{r['buy_base_log_loss']:.4f} (rate {r['buy_base_rate']:.3f}), sell "
                         f"{r['sell_base_log_loss']:.4f} (rate {r['sell_base_rate']:.3f})")
        lo, hi = tr.bootstrap_mean_r(bt_trades)
        pos = float((base["avg_r"] > 0).mean())
        lines.append(f"  All test periods: {len(bt_trades):,} decisions, avg {bt_trades['result_r'].mean():+.3f} R "
                     f"[{lo:+.3f}, {hi:+.3f}], positive in {100 * pos:.0f}% of periods")
    else:
        lines.append("  (no complete walk-forward round in the data yet)")
    lines += ["", "Decisions skipped: " + ", ".join(f"{k}: {v}" for k, v in meta["skipped"].items())]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Labels: intraday triple barrier and daily decisions.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("build", "intraday labels (development period) and their report"),
                       ("report", "report for the saved intraday labels"),
                       ("daily-build", "daily-decision labels (Phase 2B) and their report"),
                       ("daily-report", "report for the saved daily-decision labels")):
        p = sub.add_parser(name, help=text)
        choices = list(config.IG_SPREAD_PIPS) if name in ("build", "report") else list(config.FUNDING)
        p.add_argument("--instrument", default="EURUSD", choices=choices)
        p.add_argument("--save", action="store_true", help="also save the report in data/reports/")
    args = parser.parse_args(argv)
    daily = args.command.startswith("daily")
    try:
        if args.command == "build":
            meta = build_and_save(args.instrument)
            print(f"Saved {meta['rows']:,} labels to {label_file(args.instrument)}\n")
        elif args.command == "daily-build":
            meta = daily_build_and_save(args.instrument)
            print(f"Saved {meta['rows']:,} daily decisions to {daily_label_file(args.instrument)}\n")
    except (fetch_data.HistoryAccessError, ValueError) as exc:
        print(f"STOPPED: {exc}")
        return 1
    text = "\n".join(daily_report(args.instrument) if daily else report(args.instrument))
    print(text)
    if args.save:
        folder = config.DATA_DIR / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        kind = "daily_labels_report" if daily else "labels_report"
        path = folder / f"{kind}_{args.instrument}_{timeutils.now_utc():%Y-%m-%d_%H%M}UTC.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
