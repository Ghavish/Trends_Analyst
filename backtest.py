"""
backtest.py - simulates trades with variable spreads, slippage and overnight
funding, checking exits on 1-minute bars (plan Sections 7.6, 10.3).

So far: the COST REPORT (decided 16 Nov 2026). No model is involved: every
possible entry in your window is taken, as a buy and as a sell, under
several holding rules, to see how much each rule costs and how far the
break-even win rate moves.

    python backtest.py costs --instrument EURUSD --save

Rules (settings in config.py, section "Cost report"):
  * Entries: every 5-minute bar completing 17:00-20:00 MUT, entered at the
    next open; the same days and skips as the labels (holidays, first hour
    after a gap, no ATR yet, no price at entry).
  * TP / SL: SL = 1 x ATR(14) of completed 1-hour, 4-hour or daily bars
    (daily = trading days from 17:00 New York), at least 10 x the spread at
    entry; TP = 1.5 x SL.
  * Close-by: today's rule (earlier of 4 h and 22:00 MUT), or 22:00 MUT on
    the 1st / 2nd / 5th weekday after entry, never after Friday 22:00 MUT
    (so Friday entries are same-day trades).
  * Spreads per minute: inside your window (13:00-18:00 UTC) the IG spread
    you see (EUR/USD: config.IG_SPREAD_PIPS; indices: your recordings by
    hour); outside it, your recordings for that hour if there are enough,
    otherwise Dukascopy's spread for that hour x how much wider IG was than
    Dukascopy in your recorded hours (an estimate). A second set of results
    doubles every spread outside your window.
  * Stop slippage: config.COST_SLIPPAGE (more outside your window); a stop is
    filled at the minute's open if price gapped past it.
  * Overnight funding per 17:00 New York rollover crossed (config.FUNDING;
    EUR/USD Wednesday counts three nights).
  * Results in R (1 R = the stop distance), costs included.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
import features as ft
import fetch_data
import timeutils

SIDES = ("buy", "sell")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")


# --------------------------------------------------------------------------
# Price units
# --------------------------------------------------------------------------
def pip(instrument: str) -> float:
    """Price per pip / point (IG's 'one point'): 0.0001 for EUR/USD, 1 for the $1 indices."""
    return config.PIP_SIZE.get(instrument, 1.0)


# --------------------------------------------------------------------------
# ATR of the chosen timeframe, known when each 5-minute bar completes
# --------------------------------------------------------------------------
def ny_time_utc(dates, hour: int) -> pd.DatetimeIndex:
    """hh:00 New York on each plain (time-zone-free) date, as UTC."""
    local = (pd.DatetimeIndex(dates).normalize() + pd.Timedelta(hours=hour)).tz_localize(config.TZ_NEW_YORK)
    return local.tz_convert("UTC")


def trading_day_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The trading day (starting 17:00 New York) each time belongs to, as a plain date."""
    ny = index.tz_convert(config.TZ_NEW_YORK)
    return (ny - pd.Timedelta(hours=config.TRADING_DAY_RESET_HOUR)).tz_localize(None).normalize()


def daily_atr(b1: pd.DataFrame) -> pd.Series:
    """ATR(14) of trading-day bars (17:00 to 17:00 New York), indexed by when each
    day is COMPLETE. Daily bars use the previous day's close as usual."""
    m = ft.mid(b1)
    key = trading_day_key(m.index)
    g = m.groupby(key)
    d = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    prev = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / config.ATR_BARS, adjust=False, min_periods=config.ATR_BARS).mean()
    ready = ny_time_utc(d.index + pd.Timedelta(days=1), config.TRADING_DAY_RESET_HOUR)   # day complete
    return pd.Series(a.to_numpy(), index=ready)


def atr_at_decision(b1: pd.DataFrame, base_index: pd.DatetimeIndex, kind) -> np.ndarray:
    minutes = config.DECISION_BAR_MINUTES
    if kind == "D":
        a = daily_atr(b1)
        left = pd.DataFrame({"ready": base_index + pd.Timedelta(minutes=minutes)})
        right = pd.DataFrame({"ready": a.index, "v": a.to_numpy()}).dropna()
        return pd.merge_asof(left, right, on="ready", direction="backward")["v"].to_numpy()
    m = ft.mid(ft.resample_bars(b1, kind))
    return ft.align_completed(ft.atr(m, kind), kind, base_index, minutes).to_numpy()


# --------------------------------------------------------------------------
# Spreads and slippage per minute
# --------------------------------------------------------------------------
def in_window(index: pd.DatetimeIndex) -> np.ndarray:
    h0, h1 = config.COST_WINDOW_UTC_HOURS
    return np.asarray((index.hour >= h0) & (index.hour < h1))


def recorded_ig_spread_by_hour(instrument: str) -> pd.Series:
    """Median IG spread by UTC hour from your recordings (complete bars only);
    NaN for hours with fewer than COST_MIN_RECORDED_MINUTES minutes."""
    folder = config.BARS_1M_DIR / instrument
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.Series(np.nan, index=range(24))
    b = pd.concat(frames, ignore_index=True)
    b = b[~b["partial"].astype(bool)]
    hour = pd.to_datetime(b["time_utc"], utc=True).dt.hour
    g = b.groupby(hour)["spread_mean"]
    med = g.median().where(g.size() >= config.COST_MIN_RECORDED_MINUTES)
    return med.reindex(range(24))


def spread_table(instrument: str, b1: pd.DataFrame) -> pd.DataFrame:
    """Spread used for each UTC hour (price units), where it came from, and Dukascopy's for comparison."""
    duka = b1["spread_mean"].groupby(b1.index.hour).median().reindex(range(24))
    ig = recorded_ig_spread_by_hour(instrument)
    both = ig.notna() & duka.notna() & (duka > 0)
    if not both.any():
        raise ValueError(f"no IG recordings for {instrument} in {config.BARS_1M_DIR}: the cost report "
                         "needs your recorded IG spreads (run the engine during your window)")
    ratio = float((ig[both] / duka[both]).median())
    used = ig.where(ig.notna(), duka * ratio)
    source = np.where(ig.notna(), "IG recorded", "estimate (Dukascopy x ratio)")
    h0, h1 = config.COST_WINDOW_UTC_HOURS
    if instrument in config.IG_SPREAD_PIPS:                 # decided fixed spread inside your window
        for h in range(h0, h1):
            used[h] = config.IG_SPREAD_PIPS[instrument] * pip(instrument)
            source[h] = "fixed (config)"
    return pd.DataFrame({"used": used, "source": source, "ig_recorded": ig, "dukascopy": duka,
                         "ratio": ratio})


def spread_per_minute(b1: pd.DataFrame, table: pd.DataFrame, night_multiple: float = 1.0) -> np.ndarray:
    s = table["used"].to_numpy()[b1.index.hour]
    return np.where(in_window(b1.index), s, s * night_multiple)


def slippage_per_minute(instrument: str, b1: pd.DataFrame) -> np.ndarray:
    rule = config.COST_SLIPPAGE[instrument]
    inside = in_window(b1.index)
    if "inside_pips" in rule:
        return np.where(inside, rule["inside_pips"], rule["outside_pips"]) * pip(instrument)
    hourly = ft.mid(ft.resample_bars(b1, 60))
    a1 = ft.align_completed(ft.atr(hourly, 60), 60, b1.index, 1).to_numpy()   # completed hours only
    slip = rule["inside_atr_share"] * a1 * np.where(inside, 1.0, rule["outside_multiple"])
    open_us = ft.local_time_utc(b1.index, config.TZ_NEW_YORK, (9, 30))
    after_open = (b1.index >= open_us) & (b1.index < open_us + pd.Timedelta(minutes=rule["us_open_minutes"]))
    return np.where(np.asarray(after_open), slip * rule["us_open_multiple"], slip)


# --------------------------------------------------------------------------
# Entries and close-by times
# --------------------------------------------------------------------------
def close_by_days(entries: pd.DatetimeIndex, days: int) -> pd.DatetimeIndex:
    """22:00 MUT on the `days`-th weekday after the entry's (MUT) date, never after
    Friday 22:00 MUT of the entry week. days = 0: today's day-trade rule."""
    if days == 0:
        return ft.close_by_times(entries)
    mut = pd.DatetimeIndex(entries).tz_convert(config.TZ_MAURITIUS)
    d = mut.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
    target = np.busday_offset(d, days, roll="forward")
    friday = d + (4 - mut.day_of_week.to_numpy()).astype("timedelta64[D]")
    close_day = np.minimum(target, friday)
    local = pd.DatetimeIndex(close_day).tz_localize(config.TZ_MAURITIUS) + pd.Timedelta(
        hours=config.CLOSE_BY_MUT[0], minutes=config.CLOSE_BY_MUT[1])
    return local.tz_convert("UTC")


def candidates(b1: pd.DataFrame, instrument: str, atr_kind, days: int, spread_min: np.ndarray):
    """Entries in your window with their stop distance and close-by time, plus skip counts."""
    minutes = config.DECISION_BAR_MINUTES
    base = ft.resample_bars(b1, minutes)
    entry = base.index + pd.Timedelta(minutes=minutes)
    in_win = ft.in_entry_window(entry)
    a = atr_at_decision(b1, base.index, atr_kind)
    new_run = (~ft.follows_previous(base.index, minutes)).to_numpy()
    since_gap = pd.Series(np.ones(len(base), int)).groupby(np.cumsum(new_run)).cumsum().to_numpy() - 1
    days_mut = entry.tz_convert(config.TZ_MAURITIUS).date
    holiday = np.zeros(len(base), bool)
    holiday[in_win] = [timeutils.no_entry_reason(d) is not None for d in days_mut[in_win]]
    pos = np.searchsorted(ft.ns(b1.index), ft.ns(entry))
    has_price = np.asarray(entry.isin(b1.index))
    keep = in_win.copy()
    skipped = {}
    for name, bad in (("holiday or year-end", holiday), ("ATR not available yet", np.isnan(a)),
                      (f"within {config.BARS_AFTER_GAP_NO_TRADE} bars after a gap or weekend",
                       since_gap < config.BARS_AFTER_GAP_NO_TRADE),
                      ("no price at the entry minute", ~has_price)):
        skipped[name] = int((keep & bad).sum())
        keep &= ~bad
    e = entry[keep]
    s_entry = spread_min[pos[keep]]
    floor = config.STOP_FLOOR_SPREADS * s_entry
    sl = np.maximum(config.SL_ATR_MULTIPLE * a[keep], floor)
    cand = pd.DataFrame({"entry_utc": e, "close_by_utc": close_by_days(e, days), "atr": a[keep],
                         "sl_dist": sl, "tp_dist": sl * config.TP_ATR_MULTIPLE / config.SL_ATR_MULTIPLE,
                         "floor_applied": config.SL_ATR_MULTIPLE * a[keep] < floor, "spread_entry": s_entry})
    return cand.reset_index(drop=True), skipped


# --------------------------------------------------------------------------
# Funding
# --------------------------------------------------------------------------
def rollovers(start: pd.Timestamp, end: pd.Timestamp, wednesday_triple: bool):
    """17:00 New York rollover instants (UTC, ns) on weekdays between start and end,
    with their weight in nights (Wednesday = 3 for forex, T+2 settlement)."""
    days = pd.date_range(start.tz_convert(config.TZ_NEW_YORK).tz_localize(None).normalize() - pd.Timedelta(days=1),
                         end.tz_convert(config.TZ_NEW_YORK).tz_localize(None).normalize() + pd.Timedelta(days=1),
                         freq="D")
    days = days[days.day_of_week < 5]
    t = ny_time_utc(days, config.TRADING_DAY_RESET_HOUR)
    w = np.where((days.day_of_week == 2) & wednesday_triple, 3.0, 1.0)
    return ft.ns(t), w


def nights_held(entry_ns: np.ndarray, exit_ns: np.ndarray, roll_ns: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted count of rollovers r with entry < r <= exit."""
    cw = np.concatenate([[0.0], np.cumsum(w)])
    return cw[np.searchsorted(roll_ns, exit_ns, side="right")] - cw[np.searchsorted(roll_ns, entry_ns, side="right")]


def funding_cost(instrument: str, side: str, nights: np.ndarray, entry_price: np.ndarray) -> np.ndarray:
    """Funding paid, in price units (positive = cost)."""
    f = config.FUNDING[instrument]
    if f["kind"] == "pips_per_night":
        return nights * f[side] * pip(instrument)
    return nights * entry_price * f[side] / f["day_count"]


# --------------------------------------------------------------------------
# Simulation on 1-minute bars
# --------------------------------------------------------------------------
CHUNK = 300


def simulate(b1: pd.DataFrame, cand: pd.DataFrame, instrument: str, spread_min: np.ndarray,
             slip_min: np.ndarray, max_gap_minutes: int,
             exit_after_close: bool = False) -> tuple[pd.DataFrame, int]:
    """Every candidate as a buy and a sell. Returns one row per (entry, side) and
    the number of entries dropped because the data had a gap during the trade.
    exit_after_close=True (daily decisions): a trade still open at its close-by
    time exits at the OPEN of the first minute at or after it - if the market
    is shut then, the first tradable price after it reopens, so gaps count;
    gaps inside the trade are then allowed up to max_gap_minutes."""
    m = ft.mid(b1)
    t = ft.ns(b1.index)
    mo, mh, ml, mc = (m[c].to_numpy() for c in ("open", "high", "low", "close"))
    n = len(t)
    gap_ns = (1 + max_gap_minutes) * 60 * 10**9
    fund = config.FUNDING[instrument]
    roll_ns, roll_w = rollovers(b1.index[0], b1.index[-1], fund["wednesday_triple"])
    rows, dropped = [], 0
    for c0 in range(0, len(cand), CHUNK):
        c = cand.iloc[c0:c0 + CHUNK]
        e_ns, cb = ft.ns(c["entry_utc"]), ft.ns(c["close_by_utc"])
        pos = np.searchsorted(t, e_ns)
        end = np.searchsorted(t, cb)                         # first minute at/after close-by
        k = int(max(end - pos).max()) + 1 if len(c) else 1
        step = np.arange(k)
        raw = pos[:, None] + step
        idx = np.minimum(raw, n - 1)
        tt = t[idx]
        valid = (raw < n) & (tt < cb[:, None])
        dt = np.diff(np.where(valid, tt, np.nan), axis=1)
        broken = np.nanmax(np.where(np.isnan(dt), 0, dt), axis=1, initial=0) > gap_ns
        last_j = np.maximum(valid.sum(axis=1) - 1, 0)
        rws = np.arange(len(c))
        last_t = tt[rws, last_j]
        if exit_after_close:
            end_c = np.minimum(end, n - 1)
            broken |= (end >= n) | ((t[end_c] - cb) > gap_ns)   # no price soon enough after the close time
        else:
            broken |= (cb - last_t) > gap_ns
        sl_d, tp_d = c["sl_dist"].to_numpy(), c["tp_dist"].to_numpy()
        s_path = spread_min[idx]
        for side in SIDES:
            sign = 1.0 if side == "buy" else -1.0
            s_e = spread_min[pos]
            entry = mo[pos] + sign * s_e / 2
            tp, sl = entry + sign * tp_d, entry - sign * sl_d
            exit_hi = mh[idx] - sign * s_path / 2
            exit_lo = ml[idx] - sign * s_path / 2
            if side == "buy":
                hit_tp, hit_sl = valid & (exit_hi >= tp[:, None]), valid & (exit_lo <= sl[:, None])
            else:
                hit_tp, hit_sl = valid & (exit_lo <= tp[:, None]), valid & (exit_hi >= sl[:, None])
            j_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), k)
            j_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), k)
            is_sl = (j_sl < k) & (j_sl <= j_tp)
            is_tp = (j_tp < k) & ~is_sl
            j = np.where(is_sl, j_sl, np.where(is_tp, j_tp, last_j))
            ix = idx[rws, j]
            s_x = spread_min[ix]
            open_exit = mo[ix] - sign * s_x / 2
            gapped = (open_exit < sl) if side == "buy" else (open_exit > sl)
            sl_fill = np.where(gapped, open_exit, sl) - sign * slip_min[ix]
            if exit_after_close:
                s_end = spread_min[end_c]
                timeout_px = mo[end_c] - sign * s_end / 2      # first tradable price at/after close-by
                timeout_ns = t[end_c]
                # Market shut at the close time and reopening beyond a level: the order at IG
                # acts at the reopening price (stop: that price minus slippage; take profit: its level).
                open_sl = ~(is_sl | is_tp) & ((timeout_px <= sl) if side == "buy" else (timeout_px >= sl))
                open_tp = ~(is_sl | is_tp) & ((timeout_px >= tp) if side == "buy" else (timeout_px <= tp))
                sl_fill = np.where(open_sl, timeout_px - sign * slip_min[end_c], sl_fill)
                is_sl, is_tp = is_sl | open_sl, is_tp | open_tp
                ix = np.where(open_sl | open_tp, end_c, ix)
                s_x = np.where(open_sl | open_tp, s_end, s_x)
                s_x = np.where(is_sl | is_tp, s_x, s_end)
            else:
                timeout_px = mc[ix] - sign * s_x / 2
                timeout_ns = cb
            exit_px = np.where(is_sl, sl_fill, np.where(is_tp, tp, timeout_px))
            exit_ns = np.where(is_sl | is_tp, t[ix], timeout_ns)
            nights = nights_held(e_ns, exit_ns, roll_ns, roll_w)
            fcost = funding_cost(instrument, side, nights, mo[pos])
            gross_r = sign * (exit_px - entry) / sl_d
            rows.append(pd.DataFrame({
                "entry_utc": c["entry_utc"].to_numpy(), "side": side,
                "outcome": np.where(is_sl, "SL", np.where(is_tp, "TP", "TIMEOUT")),
                "result_r": gross_r - fcost / sl_d,
                "spread_r": (s_e / 2 + s_x / 2) / sl_d,
                "slippage_r": np.where(is_sl, (np.abs(sl - sl_fill)) / sl_d, 0.0),
                "funding_r": fcost / sl_d, "nights": nights,
                "hours_held": (exit_ns - e_ns) / 3.6e12, "sl_pips": sl_d / pip(instrument),
                "floor_applied": c["floor_applied"].to_numpy(), "broken": broken,
            }))
        dropped += int(broken.sum())
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out = out[~out["broken"]].drop(columns="broken").reset_index(drop=True)
        e = pd.to_datetime(out["entry_utc"], utc=True)
        out["day"] = e.dt.tz_convert(None).dt.normalize()
        out["weekday"] = e.dt.tz_convert(config.TZ_MAURITIUS).dt.day_of_week
    return out, dropped


# --------------------------------------------------------------------------
# Summary and report
# --------------------------------------------------------------------------
def summarize(trades: pd.DataFrame) -> dict:
    """Costs, outcomes, break-even TP share of decided trades and the no-model average R."""
    from train import bootstrap_mean_r
    if len(trades) == 0:
        return {"n": 0}
    o = trades["outcome"]
    tp, sl = trades[o == "TP"]["result_r"], trades[o == "SL"]["result_r"]
    avg_tp = tp.mean() if len(tp) else config.TP_ATR_MULTIPLE
    avg_sl = sl.mean() if len(sl) else -1.0
    decided = (o != "TIMEOUT").sum()
    lo, hi = bootstrap_mean_r(trades)
    return {"n": len(trades), "sl_pips": trades["sl_pips"].median(), "floor": trades["floor_applied"].mean(),
            "tp": (o == "TP").mean(), "sl": (o == "SL").mean(), "timeout": (o == "TIMEOUT").mean(),
            "spread_r": trades["spread_r"].mean(), "slippage_r": trades["slippage_r"].mean(),
            "funding_r": trades["funding_r"].mean(), "nights": trades["nights"].mean(),
            "hours": trades["hours_held"].mean(),
            "break_even": -avg_sl / (avg_tp - avg_sl) if avg_tp > avg_sl else np.nan,
            "tp_share": (o == "TP").sum() / decided if decided else np.nan,
            "avg_r": trades["result_r"].mean(), "ci": (lo, hi), **cost_vs_move(trades)}


def cost_vs_move(trades: pd.DataFrame) -> dict:
    """
    Cost relative to the typical move: the total cost per trade (spread +
    slippage + funding, in R) divided by the standard deviation of trade
    results (how much a trade's result typically varies). The smaller it is,
    the less predictive skill is needed to overcome the costs.
    'decisions to see' = how many INDEPENDENT decisions it takes before an
    edge as big as the costs stands 2 standard errors clear of zero:
    (2 / ratio) squared. Entries on the same evening share most of their price
    path, so one decision per instrument per trading day is counted, and
    'years' = decisions to see / decision days per year in the data.
    """
    cost = float((trades["spread_r"] + trades["slippage_r"] + trades["funding_r"]).mean())
    sd = float(trades["result_r"].std())
    ratio = cost / sd if sd > 0 else np.nan
    need = (2 / ratio) ** 2 if ratio and ratio > 0 else np.nan
    days = pd.to_datetime(trades["day"])
    span_years = (days.max() - days.min()).days / 365.25 if len(days) else 0
    per_year = days.nunique() / span_years if span_years > 0 else np.nan
    return {"cost_r": cost, "sd_r": sd, "cost_sd": ratio, "trades_to_see": need,
            "days_per_year": per_year, "years_to_see": need / per_year if per_year else np.nan}


def run_costs(b1: pd.DataFrame, instrument: str, night_multiple: float = 1.0, scenarios=None):
    table = spread_table(instrument, b1)
    spread_min = spread_per_minute(b1, table, night_multiple)
    slip_min = slippage_per_minute(instrument, b1)
    results = []
    for name, atr_kind, days in scenarios or config.COST_SCENARIOS:
        cand, skipped = candidates(b1, instrument, atr_kind, days, spread_min)
        gap = config.GAP_MINUTES if days == 0 else config.COST_MAX_GAP_MINUTES[instrument]
        trades, dropped = simulate(b1, cand, instrument, spread_min, slip_min, gap)
        skipped["gap in the data during the trade"] = dropped
        results.append({"name": name, "atr": atr_kind, "days": days, "trades": trades, "skipped": skipped})
    return table, results


def _row(label: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"  {label:<32} no trades"
    lo, hi = s["ci"]
    return (f"  {label:<32} {s['n']:>6} {s['sl_pips']:>7.1f} {100 * s['tp']:>4.0f}/{100 * s['sl']:>2.0f}/"
            f"{100 * s['timeout']:>2.0f} {s['hours']:>6.1f} {s['spread_r']:>6.3f} {s['slippage_r']:>6.3f} "
            f"{s['funding_r']:>6.3f} {100 * s['break_even']:>6.1f}% {100 * s['tp_share']:>6.1f}% "
            f"{s['avg_r']:>+7.3f} [{lo:+.3f}, {hi:+.3f}]")


def header(unit: str) -> str:
    return (f"  {'rule / side':<32} {'trades':>6} {'stop':>7} {'TP/SL/TO %':>10} {'hours':>6} "
            f"{'spread':>6} {'slip':>6} {'fund':>6} {'break-':>7} {'actual':>7} {'avg R':>7}  95% interval\n"
            f"  {'':<32} {'':>6} {'(' + unit + ')':>7} {'':>10} {'held':>6} {'(R)':>6} {'(R)':>6} {'(R)':>6} "
            f"{'even TP':>7} {'TP':>7} {'(no model)':>10}")


def _move_row(label: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"  {label:<32} no trades"
    return (f"  {label:<32} {s['cost_r']:>9.3f} {s['sd_r']:>9.3f} {s['cost_sd']:>9.3f} "
            f"{s['trades_to_see']:>12,.0f} {s['years_to_see']:>9.1f}")


def cost_report(instrument: str, b1: pd.DataFrame) -> list[str]:
    f = config.FUNDING[instrument]
    unit = "pips" if instrument in config.PIP_SIZE else "pts"
    lines = [f"Cost report: {instrument}, no model - every entry 17:00-20:00 MUT taken as a buy and as a sell.",
             "TP = 1.5 x SL; SL = 1 x ATR(14) of the timeframe shown, at least 10 x the spread at entry.",
             "'break-even TP' = share of finished trades (TP or SL) that must hit TP to break even, with costs;",
             "'actual TP' = the share that did. 'avg R' = average result per trade of ALL entries (no model),",
             "with a 95% interval resampling whole days. Costs in R: spread (entry + exit), stop slippage, funding.",
             f"Funding: buys {f['buy']} / sells {f['sell']} "
             + ("pips a night (Wednesday x3)" if f["kind"] == "pips_per_night" else
                f"a year of the price, /{f['day_count']}") + ("  [PROVISIONAL]" if f.get("provisional") else ""),
             ""]
    for mult in (1.0, config.COST_NIGHT_SPREAD_MULTIPLE):
        table, results = run_costs(b1, instrument, mult)
        if mult == 1.0:
            lines.append(f"Spread used by UTC hour ({'pips' if unit == 'pips' else 'points'}; "
                         "your window is 13-17 UTC = 17-22 MUT):")
            for h, r in table.iterrows():
                rec = f"{r['ig_recorded'] / pip(instrument):.2f}" if pd.notna(r["ig_recorded"]) else "-"
                lines.append(f"  {h:02d} UTC ({(h + 4) % 24:02d} MUT): {r['used'] / pip(instrument):6.2f}  "
                             f"[{r['source']}; IG recorded {rec}, Dukascopy {r['dukascopy'] / pip(instrument):.2f}]")
            lines.append(f"  (IG / Dukascopy ratio in recorded hours: {table['ratio'].iloc[0]:.2f})")
            lines.append("")
            lines.append("A. Spreads as above")
        else:
            lines.append(f"B. Spreads outside your window x{mult:g} (night spreads doubled)")
        lines.append(header(unit))
        sums = {}
        for res in results:
            for side in SIDES:
                t = res["trades"][res["trades"]["side"] == side] if len(res["trades"]) else res["trades"]
                sums[(res["name"], side)] = summarize(t)
                lines.append(_row(f"{res['name']} - {side}", sums[(res["name"], side)]))
        lines.append("")
        lines.append(f"{'A' if mult == 1.0 else 'B'}2. Cost relative to the typical move (smaller = less skill needed)")
        lines.append("   cost = spread + slippage + funding per trade; sd = standard deviation of trade results;")
        lines.append("   'decisions to see' = independent decisions needed before an edge as large as the costs")
        lines.append("   stands 2 standard errors clear of zero, (2 / (cost/sd)) squared; one decision per")
        lines.append("   instrument per trading day (same-evening entries share their price path), so")
        lines.append("   'years' = decisions / decision days per year (about 240 here), for ONE instrument.")
        lines.append(f"  {'rule / side':<32} {'cost (R)':>9} {'sd (R)':>9} {'cost/sd':>9} {'decisions':>12} {'years':>9}")
        for (name, side), sm in sums.items():
            lines.append(_move_row(f"{name} - {side}", sm))
        lines.append("")
        if mult == 1.0:
            lines.append("By entry weekday (average R, no model; Friday entries close the same day at 22:00 MUT):")
            lines.append(f"  {'rule / side':<32} " + " ".join(f"{d:>7}" for d in WEEKDAYS))
            for res in results:
                if res["days"] == 0:
                    continue
                for side in SIDES:
                    t = res["trades"][res["trades"]["side"] == side]
                    avg = t.groupby("weekday")["result_r"].mean().reindex(range(5))
                    lines.append(f"  {res['name'] + ' - ' + side:<32} " +
                                 " ".join(f"{v:>+7.3f}" if pd.notna(v) else f"{'-':>7}" for v in avg))
            lines.append("")
            lines.append("Entries skipped (inside the window), first rule shown: " +
                         ", ".join(f"{k}: {v}" for k, v in results[0]["skipped"].items()))
            lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtests and cost reports.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("costs", help="cost report: no model, several holding rules")
    p.add_argument("--instrument", default="EURUSD", choices=list(config.FUNDING))
    p.add_argument("--save", action="store_true", help="also save the report in data/reports/")
    args = parser.parse_args(argv)
    timeutils.configure_logging()
    try:
        bars = fetch_data.load_history_1m(args.instrument, config.HISTORY_START_UTC,
                                          fetch_data.development_end_utc(), fetch_data.PURPOSE_COST_ANALYSIS)
        if bars.empty:
            raise ValueError(f"no {args.instrument} history yet: run  python fetch_data.py download "
                             f"--instruments {args.instrument}")
        text = "\n".join(cost_report(args.instrument, ft.to_indexed(bars)))
    except (fetch_data.HistoryAccessError, ValueError) as exc:
        print(f"STOPPED: {exc}")
        return 1
    print(text)
    if args.save:
        folder = config.DATA_DIR / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"costs_{args.instrument}_{timeutils.now_utc():%Y-%m-%d_%H%M}UTC.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
