"""
train.py - LightGBM with probability calibration, judged by walk-forward
testing only (plan Sections 7.6, 7.9, 7.10). Settings in config.py
(decided 15 Nov 2026; fixed, not tuned).

    python train.py walkforward    # EUR/USD: all variants, all rounds, full report
    python train.py diagnose       # diagnostics on the saved test predictions (no retraining)

How one walk-forward round works:
  1. Train on everything before the calibration period (expanding window).
  2. Calibrate on the next 6 months (Platt / sigmoid; isotonic shown for
     information) and measure there the average R of winners and of
     non-winners for the expected value.
  3. Test on the 6 months after that, which the model has never seen.
  One trading day is left out between the periods. Then roll forward 6 months.

For every test entry and direction:
  p   = calibrated probability that TP is hit before SL
  EV  = p x (average R of winners) + (1 - p) x (average R of non-winners)
  A signal needs EV >= SIGNAL_MIN_EV_R; if buy and sell both qualify, the bar
  is skipped. Trades are taken one at a time (no new trade while one is open).

Variants A-D (config.VARIANTS) use different feature groups but exactly the
same rounds, data and rules. A group is kept only if adding it lowers the
test-period log loss in at least GROUP_KEEP_MIN_PERIODS periods.

Only the development period is used (labels and features are built from it);
the final-check period is never touched here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
import features as ft
import fetch_data
import labels as lb
import timeutils

SIDES = lb.SIDES
EPS = 1e-12


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def prepare(labels: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Labels joined with features on the signal bar; adds the entry's trading day (UTC date)."""
    f = feats.copy()
    if "signal_bar_utc" in f.columns:
        f = f.set_index("signal_bar_utc")
    f.index = pd.DatetimeIndex(pd.to_datetime(f.index, utc=True), name="signal_bar_utc")
    lab = labels.copy()
    lab.index = pd.DatetimeIndex(pd.to_datetime(lab["signal_bar_utc"], utc=True), name="signal_bar_utc")
    lab = lab.drop(columns=["signal_bar_utc"])
    df = lab.join(f, how="inner").sort_index()
    df["day"] = pd.to_datetime(df["entry_utc"], utc=True).dt.tz_convert(None).dt.normalize()
    return df


def load_dataset(instrument: str) -> tuple[pd.DataFrame, dict, dict]:
    """The saved labels and features, joined. Refuses data from the final-check period."""
    lpath, fpath = lb.label_file(instrument), ft.feature_file(instrument)
    for p, cmd in ((lpath, "python labels.py build"), (fpath, "python features.py build")):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found: run  {cmd}")
    lmeta = json.loads(lpath.with_suffix(".json").read_text(encoding="utf-8"))
    fmeta = json.loads(fpath.with_suffix(".json").read_text(encoding="utf-8"))
    df = prepare(pd.read_parquet(lpath), pd.read_parquet(fpath))
    last = pd.to_datetime(df["close_by_utc"], utc=True).max()
    if last > pd.Timestamp(fetch_data.development_end_utc()):
        raise fetch_data.HistoryAccessError(
            f"the data reaches {last}, past the end of the development period")
    return df, fmeta["groups"], {"labels": lmeta, "features": fmeta}


# --------------------------------------------------------------------------
# Walk-forward schedule
# --------------------------------------------------------------------------
@dataclass
class Round:
    number: int
    cal_start: pd.Timestamp       # training is everything before this (minus the gap)
    test_start: pd.Timestamp      # calibration is [cal_start, test_start) (minus the gap)
    test_end: pd.Timestamp        # test is [test_start, test_end)

    def label(self) -> str:
        last = self.test_end - pd.Timedelta(days=1)
        return f"{self.test_start:%Y-%m-%d} to {last:%Y-%m-%d}"


def _day(ts) -> pd.Timestamp:
    """A date/time as a plain (time-zone-free) UTC date, like the 'day' column."""
    t = pd.Timestamp(ts)
    return (t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t).normalize()


def schedule(data_end, first_test_start=None, months: int | None = None) -> list[Round]:
    """Rounds whose test periods start at first_test_start and roll forward `months`."""
    months = months or config.WF_PERIOD_MONTHS
    start = _day(first_test_start or config.WF_FIRST_TEST_START)
    end = _day(data_end)
    rounds, k = [], 0
    while True:
        test_start = start + pd.DateOffset(months=months * k)
        if test_start >= end:
            break
        rounds.append(Round(k + 1, test_start - pd.DateOffset(months=months), test_start,
                            min(test_start + pd.DateOffset(months=months), end)))
        k += 1
    return rounds


def split(df: pd.DataFrame, rnd: Round, gap_days: int | None = None) -> tuple[pd.DataFrame, ...]:
    """Train / calibration / test rows for one round, with gap_days trading days left out
    before the calibration period and before the test period."""
    gap = config.WF_GAP_TRADING_DAYS if gap_days is None else gap_days
    days = np.sort(df["day"].unique())

    def last_days_before(boundary):
        before = days[days < np.datetime64(boundary)]
        return set(before[-gap:]) if gap else set()

    drop = last_days_before(rnd.cal_start) | last_days_before(rnd.test_start)
    keep = ~df["day"].isin(drop)
    train = df[keep & (df["day"] < rnd.cal_start)]
    cal = df[keep & (df["day"] >= rnd.cal_start) & (df["day"] < rnd.test_start)]
    test = df[(df["day"] >= rnd.test_start) & (df["day"] < rnd.test_end)]
    return train, cal, test


# --------------------------------------------------------------------------
# One model: LightGBM + Platt calibration (+ isotonic for information)
# --------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class DirectionModel:
    side: str
    features: list[str]
    booster: object
    platt_a: float
    platt_b: float
    iso_x: list = field(default_factory=list)
    iso_y: list = field(default_factory=list)
    avg_win_r: float = np.nan
    avg_nonwin_r: float = np.nan
    cal_base_rate: float = np.nan

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.features], raw_score=True)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        r = self.raw(X)
        p = _sigmoid(self.platt_a * r + self.platt_b)
        p_iso = np.interp(_sigmoid(r), self.iso_x, self.iso_y) if self.iso_x else np.full(len(r), np.nan)
        ev = p * self.avg_win_r + (1 - p) * self.avg_nonwin_r
        return pd.DataFrame({"raw": r, "p": p, "p_iso": p_iso, "ev": ev}, index=X.index)

    def calibration_dict(self) -> dict:
        return {"side": self.side, "features": self.features, "platt_a": self.platt_a,
                "platt_b": self.platt_b, "iso_x": self.iso_x, "iso_y": self.iso_y,
                "avg_win_r": self.avg_win_r, "avg_nonwin_r": self.avg_nonwin_r,
                "cal_base_rate": self.cal_base_rate}


def fit_direction(side: str, features: list[str], train: pd.DataFrame, cal: pd.DataFrame,
                  params: dict | None = None, trees: int | None = None) -> DirectionModel:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    params = dict(params or config.LGBM_PARAMS)
    y_tr, y_cal = train[f"{side}_win"].to_numpy(), cal[f"{side}_win"].to_numpy()
    r_cal = cal[f"{side}_result_r"].to_numpy()
    if y_cal.sum() == 0 or y_cal.sum() == len(y_cal):
        raise ValueError(f"{side}: the calibration period has only one outcome; it can't be calibrated")
    booster = lgb.train(params, lgb.Dataset(train[features], y_tr), num_boost_round=trees or config.LGBM_TREES)
    raw_cal = booster.predict(cal[features], raw_score=True)
    platt = LogisticRegression(C=np.inf).fit(raw_cal.reshape(-1, 1), y_cal)       # Platt: no penalty
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(_sigmoid(raw_cal), y_cal)
    return DirectionModel(side, list(features), booster, float(platt.coef_[0, 0]), float(platt.intercept_[0]),
                          [float(x) for x in iso.X_thresholds_], [float(y) for y in iso.y_thresholds_],
                          float(r_cal[y_cal == 1].mean()), float(r_cal[y_cal == 0].mean()), float(y_cal.mean()))


# --------------------------------------------------------------------------
# Signals and trades
# --------------------------------------------------------------------------
def choose_signals(pred: pd.DataFrame, threshold: float) -> pd.Series:
    """'buy', 'sell' or '' per row: EV above the threshold; both qualifying = skip."""
    buy = pred["buy_ev"].to_numpy() >= threshold
    sell = pred["sell_ev"].to_numpy() >= threshold
    out = np.where(buy & ~sell, "buy", np.where(sell & ~buy, "sell", ""))
    return pd.Series(out, index=pred.index)


def take_trades(test: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """One trade at a time: a signal is taken only if no earlier trade is still open."""
    rows = []
    open_until = pd.Timestamp.min.tz_localize("UTC")
    entries = pd.to_datetime(test["entry_utc"], utc=True)
    for ts, side in signals[signals != ""].items():
        e = entries.loc[ts]
        if e < open_until:
            continue
        r = test.loc[ts]
        exit_t = pd.Timestamp(r[f"{side}_exit_utc"])
        rows.append({"signal_bar_utc": ts, "entry_utc": e, "day": r["day"], "side": side,
                     "outcome": r[f"{side}_outcome"], "win": int(r[f"{side}_win"]),
                     "result_r": float(r[f"{side}_result_r"]), "exit_utc": exit_t})
        open_until = exit_t
    return pd.DataFrame(rows, columns=["signal_bar_utc", "entry_utc", "day", "side", "outcome",
                                       "win", "result_r", "exit_utc"])


def max_drawdown(r: pd.Series) -> float:
    """Largest fall of the cumulative R from a previous peak (0 or negative)."""
    if len(r) == 0:
        return 0.0
    eq = np.concatenate([[0.0], np.cumsum(np.asarray(r, float))])
    return float((eq - np.maximum.accumulate(eq)).min())


def bootstrap_mean_r(trades: pd.DataFrame, resamples: int | None = None, seed: int | None = None):
    """95% interval for the average R per trade, resampling whole trading days
    (trades on the same day stay together)."""
    if len(trades) == 0:
        return (np.nan, np.nan)
    g = trades.groupby("day")["result_r"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(config.BOOTSTRAP_SEED if seed is None else seed)
    pick = rng.integers(0, len(g), size=(resamples or config.BOOTSTRAP_RESAMPLES, len(g)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def group_decision(period_losses: pd.DataFrame, min_periods: int | None = None) -> dict:
    """For each feature group: in how many periods adding it lowered the test log loss."""
    need = config.GROUP_KEEP_MIN_PERIODS if min_periods is None else min_periods
    out = {}
    for group, (with_g, without) in {"candles": ("B", "A"), "trend_1h": ("C", "A")}.items():
        if with_g in period_losses and without in period_losses:
            better = int((period_losses[with_g] < period_losses[without]).sum())
            out[group] = {"better_in": better, "periods": int(len(period_losses)), "keep": better >= need}
    return out


# --------------------------------------------------------------------------
# The whole walk-forward run
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    predictions: pd.DataFrame
    trades: pd.DataFrame
    rounds: list
    models: dict
    losses: pd.DataFrame
    baseline: pd.DataFrame
    importance: pd.DataFrame
    cal_rates: dict            # (round, side) -> share of wins in the calibration period


def run_walkforward(df: pd.DataFrame, groups: dict, variants: dict | None = None, rounds=None,
                    params: dict | None = None, trees: int | None = None, keep_models: bool = True) -> RunResult:
    variants = variants or config.VARIANTS
    rounds = rounds or schedule(pd.Timestamp(fetch_data.development_end_utc()))
    preds, trades, models, losses, base_rows, imp, cal_rates = [], [], {}, [], [], [], {}
    for rnd in rounds:
        train, cal, test = split(df, rnd)
        if len(test) == 0:
            continue
        for side in SIDES:                                   # the no-model baseline
            cal_rates[(rnd.number, side)] = float(cal[f"{side}_win"].mean())
            base_rows.append({"round": rnd.number, "period": rnd.label(), "side": side,
                              "entries": len(test), "avg_r": float(test[f"{side}_result_r"].mean())})
        loss_row = {"round": rnd.number}
        for name, gnames in variants.items():
            cols = [c for g in gnames for c in groups[g]]
            pv = pd.DataFrame(index=test.index)
            ys, ps = [], []
            for side in SIDES:
                m = fit_direction(side, cols, train, cal, params, trees)
                pr = m.predict(test)
                for c in pr.columns:
                    pv[f"{side}_{c}"] = pr[c]
                ys.append(test[f"{side}_win"].to_numpy())
                ps.append(pr["p"].to_numpy())
                if keep_models:
                    models[(rnd.number, name, side)] = m
                gain = m.booster.feature_importance(importance_type="gain")
                imp.append(pd.DataFrame({"variant": name, "side": side, "round": rnd.number,
                                         "feature": cols, "gain": gain}))
            loss_row[name] = log_loss(np.concatenate(ys), np.concatenate(ps))
            sig = choose_signals(pv, config.SIGNAL_MIN_EV_R)
            t = take_trades(test, sig)
            t.insert(0, "variant", name)
            t.insert(0, "round", rnd.number)
            trades.append(t)
            out = pv.copy()
            out["variant"], out["round"], out["signal"] = name, rnd.number, sig
            for side in SIDES:
                for c in ("win", "result_r", "outcome", "exit_utc"):
                    out[f"{side}_{c}"] = test[f"{side}_{c}"]
            out["entry_utc"], out["day"] = test["entry_utc"], test["day"]
            preds.append(out)
        losses.append(loss_row)
    return RunResult(pd.concat(preds) if preds else pd.DataFrame(),
                     pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
                     rounds, models, pd.DataFrame(losses).set_index("round") if losses else pd.DataFrame(),
                     pd.DataFrame(base_rows), pd.concat(imp, ignore_index=True) if imp else pd.DataFrame(),
                     cal_rates)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _trade_line(t: pd.DataFrame) -> str:
    if len(t) == 0:
        return "    0 trades"
    return (f"{len(t):>5} trades  win {100 * t['win'].mean():4.1f}%  avg {t['result_r'].mean():+.3f} R  "
            f"total {t['result_r'].sum():+7.1f} R  worst drawdown {max_drawdown(t['result_r']):+.1f} R")


def report(res: RunResult, instrument: str) -> list[str]:
    P, T = res.predictions, res.trades
    used = [k for k in config.VARIANTS if k in set(P["variant"])]
    names = ", ".join(k + "=" + "+".join(config.VARIANTS[k]) for k in used)
    lines = [f"Walk-forward: {instrument}, {len(res.losses)} rounds, variants {names}",
             f"Signal: EV >= {config.SIGNAL_MIN_EV_R:+.2f} R (EV = p x avg R of winners + (1-p) x avg R of "
             "non-winners, from the calibration period); both directions qualifying = skip; one trade at a time.",
             "Calibration: Platt (used); isotonic shown for information. All figures are TEST periods only.", ""]
    rounds = {r.number: r.label() for r in res.rounds}

    lines.append("1. Test-period log loss (lower is better; 'no model' = always the calibration base rate)")
    for rn, row in res.losses.iterrows():
        one = P[(P["round"] == rn) & (P["variant"] == used[0])]
        ys = np.concatenate([one[f"{s}_win"].to_numpy() for s in SIDES])
        base = np.concatenate([np.full(len(one), res.cal_rates[(rn, s)]) for s in SIDES])
        lines.append(f"  round {rn} ({rounds[rn]}): " + "  ".join(f"{v} {row[v]:.4f}" for v in res.losses.columns)
                     + f"  | no model {log_loss(ys, base):.4f}")
    dec = group_decision(res.losses)
    for g, d in dec.items():
        lines.append(f"  -> {g}: lower log loss in {d['better_in']} of {d['periods']} periods "
                     f"(keep needs {config.GROUP_KEEP_MIN_PERIODS}): {'KEEP' if d['keep'] else 'DROP'}")
    if "D" in res.losses and "A" in res.losses:
        lines.append(f"     (D, both groups, beat A in {int((res.losses['D'] < res.losses['A']).sum())} periods; information)")
    lines.append("")

    lines.append("2. Calibration over all test periods (Platt). Hit rate should match the predicted probability.")
    lines.append(f"  {'variant/side':<12} {'Brier':>7} {'log loss':>9} {'isotonic LL':>12}")
    for v in sorted(P["variant"].unique()):
        sub = P[P["variant"] == v]
        for s in SIDES:
            lines.append(f"  {v} {s:<10} {brier(sub[f'{s}_win'], sub[f'{s}_p']):>7.4f} "
                         f"{log_loss(sub[f'{s}_win'], sub[f'{s}_p']):>9.4f} "
                         f"{log_loss(sub[f'{s}_win'], sub[f'{s}_p_iso']):>12.4f}")
    band = config.CALIBRATION_BAND
    for v in sorted(P["variant"].unique()):
        sub = P[P["variant"] == v]
        lines.append(f"  Variant {v}: predicted band -> entries, average predicted, actual hit rate (buy | sell)")
        tabs = {}
        for s in SIDES:
            b = np.floor(sub[f"{s}_p"] / band) * band
            tabs[s] = sub.groupby(b).agg(n=(f"{s}_win", "size"), pred=(f"{s}_p", "mean"), hit=(f"{s}_win", "mean"))
        for k in sorted(set(tabs["buy"].index) | set(tabs["sell"].index)):
            cells = []
            for s in SIDES:
                if k in tabs[s].index:
                    r = tabs[s].loc[k]
                    cells.append(f"{int(r['n']):>6} {100 * r['pred']:5.1f}% {100 * r['hit']:5.1f}%")
                else:
                    cells.append(f"{'':>6} {'':>6} {'':>6}")
            lines.append(f"    {100 * k:4.0f}-{100 * (k + band):3.0f}%: {cells[0]}  | {cells[1]}")
    lines.append("")

    lines.append(f"3. Trades (EV >= {config.SIGNAL_MIN_EV_R:+.2f} R, one at a time), per period")
    for rn in res.losses.index:
        lines.append(f"  round {rn} ({rounds[rn]}):")
        for s in SIDES:
            b = res.baseline[(res.baseline["round"] == rn) & (res.baseline["side"] == s)]
            if len(b):
                lines.append(f"    no model, every {s} entry: {int(b['entries'].iloc[0]):>5} entries  "
                             f"avg {b['avg_r'].iloc[0]:+.3f} R")
        for v in sorted(T["variant"].unique()) if len(T) else []:
            lines.append(f"    {v}: {_trade_line(T[(T['round'] == rn) & (T['variant'] == v)])}")
    lines.append("")
    lines.append("4. All test periods together")
    for s in SIDES:
        lines.append(f"  no model, every {s} entry: avg {np.average(res.baseline[res.baseline['side'] == s]['avg_r'], weights=res.baseline[res.baseline['side'] == s]['entries']):+.3f} R")
    for v in sorted(P["variant"].unique()):
        t = T[T["variant"] == v] if len(T) else T
        lo, hi = bootstrap_mean_r(t)
        won = sum(_trade_avg(T, rn, v) > _trade_avg(T, rn, "A") for rn in res.losses.index) if v != "A" else None
        lines.append(f"  {v}: {_trade_line(t)}")
        lines.append(f"      95% interval for avg R (resampling whole days): {lo:+.3f} to {hi:+.3f}"
                     + (f"; higher avg R than A in {won} of {len(res.losses)} periods" if won is not None else ""))
        if len(t):
            by_side = ", ".join(f"{s} {len(t[t['side'] == s])} ({t[t['side'] == s]['result_r'].mean():+.3f} R)"
                                for s in SIDES if (t["side"] == s).any())
            lines.append(f"      by direction: {by_side}")
    lines.append("")
    lines.append("5. Other EV thresholds (information only; the threshold stays fixed)")
    for v in sorted(P["variant"].unique()):
        cells = []
        for thr in config.SIGNAL_EV_INFO_THRESHOLDS:
            all_t = []
            for rn in res.losses.index:
                sub = P[(P["variant"] == v) & (P["round"] == rn)]
                all_t.append(take_trades(sub, choose_signals(sub, thr)))
            tt = pd.concat(all_t) if all_t else pd.DataFrame()
            cells.append(f"EV>={thr:+.2f}: {len(tt)} trades {tt['result_r'].mean() if len(tt) else float('nan'):+.3f} R")
        lines.append(f"  {v}: " + " | ".join(cells))
    lines.append("")
    lines.append(f"6. Most used features (share of total gain, variant {used[-1]}, all rounds and both directions)")
    if len(res.importance):
        d = res.importance[res.importance["variant"] == used[-1]]
        share = d.groupby("feature")["gain"].sum()
        share = (share / share.sum()).sort_values(ascending=False).head(15)
        lines += [f"  {f:<24} {100 * v:5.1f}%" for f, v in share.items()]
    return lines


def _trade_avg(T: pd.DataFrame, rn: int, v: str) -> float:
    t = T[(T["round"] == rn) & (T["variant"] == v)] if len(T) else T
    return t["result_r"].mean() if len(t) else -np.inf


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def save_run(res: RunResult, instrument: str, meta: dict, text: str) -> Path:
    stamp = f"{timeutils.now_utc():%Y-%m-%d_%H%M}UTC"
    run_dir = config.MODELS_DIR / "walkforward" / f"{instrument}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cal = {}
    for (rn, v, s), m in res.models.items():
        m.booster.save_model(str(run_dir / f"round{rn}_{v}_{s}.txt"))
        cal[f"round{rn}_{v}_{s}"] = m.calibration_dict()
    (run_dir / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    out_dir = config.DATA_DIR / "walkforward"
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch_data.write_parquet_atomic(res.predictions.reset_index(), out_dir / f"{instrument}_{stamp}_predictions.parquet",
                                    compression="zstd")
    fetch_data.write_parquet_atomic(res.trades, out_dir / f"{instrument}_{stamp}_trades.parquet", compression="zstd")
    settings = {k.lower(): getattr(config, k) for k in (
        "WF_FIRST_TEST_START", "WF_PERIOD_MONTHS", "WF_GAP_TRADING_DAYS", "LGBM_TREES", "LGBM_PARAMS",
        "SIGNAL_MIN_EV_R", "VARIANTS", "GROUP_KEEP_MIN_PERIODS", "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED")}
    (run_dir / "run.json").write_text(json.dumps({"instrument": instrument, "settings": settings,
                                                  "data": meta, "created_utc": timeutils.now_utc().isoformat()},
                                                 indent=1, default=str), encoding="utf-8")
    reports = config.DATA_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"walkforward_{instrument}_{stamp}.txt").write_text(text + "\n", encoding="utf-8")
    return run_dir


# --------------------------------------------------------------------------
# Diagnostics on the SAVED test predictions (no retraining)
# --------------------------------------------------------------------------
# "Top bands": the probabilities where signals come from. Fixed on 15 Nov 2026
# BEFORE looking at these diagnostics: an EV of +0.10 R starts at p of about
# 0.35. The 40%+ bands are shown for information only.
DIAG_TOP_P = 0.35
DIAG_INFO_P = 0.40
EV_BANDS = (-np.inf, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, np.inf)


def _hour_mut(entry_utc) -> np.ndarray:
    return pd.to_datetime(entry_utc, utc=True).dt.tz_convert(config.TZ_MAURITIUS).dt.hour.to_numpy()


def stack_sides(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per (bar, direction): p, ev, outcome, win, result_r, entry hour."""
    parts = []
    for side in SIDES:
        parts.append(pd.DataFrame({
            "round": pred["round"].to_numpy(), "side": side, "hour": _hour_mut(pred["entry_utc"]),
            "p": pred[f"{side}_p"].to_numpy(), "ev": pred[f"{side}_ev"].to_numpy(),
            "outcome": pred[f"{side}_outcome"].to_numpy(), "win": pred[f"{side}_win"].to_numpy(),
            "result_r": pred[f"{side}_result_r"].to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def hour_baseline_losses(pred: pd.DataFrame, df: pd.DataFrame, rounds: list) -> pd.DataFrame:
    """Per round: log loss of the model vs 'no model' (calibration-period win
    rate) vs 'no model by entry hour' (calibration-period win rate per hour and direction)."""
    rows = []
    for rnd in rounds:
        pr = pred[pred["round"] == rnd.number]
        if len(pr) == 0:
            continue
        _, cal, _ = split(df, rnd)
        cal_h = _hour_mut(cal["entry_utc"])
        hours = _hour_mut(pr["entry_utc"])
        y, p_model, p_flat, p_hour = [], [], [], []
        for side in SIDES:
            rate = float(cal[f"{side}_win"].mean())
            by_h = pd.Series(cal[f"{side}_win"].to_numpy()).groupby(cal_h).mean()
            y.append(pr[f"{side}_win"].to_numpy())
            p_model.append(pr[f"{side}_p"].to_numpy())
            p_flat.append(np.full(len(pr), rate))
            p_hour.append(by_h.reindex(hours).fillna(rate).to_numpy())
        y = np.concatenate(y)
        rows.append({"round": rnd.number, "period": rnd.label(),
                     "model": log_loss(y, np.concatenate(p_model)),
                     "no_model": log_loss(y, np.concatenate(p_flat)),
                     "no_model_by_hour": log_loss(y, np.concatenate(p_hour))})
    return pd.DataFrame(rows)


def ev_vs_realised(stacked: pd.DataFrame) -> pd.DataFrame:
    """Predicted EV against the R actually obtained, in EV bands (both directions)."""
    band = pd.cut(stacked["ev"], EV_BANDS, right=False)
    return stacked.groupby(band, observed=True).agg(entries=("ev", "size"), predicted_ev=("ev", "mean"),
                                                    realised_r=("result_r", "mean"))


def direction_test(stacked: pd.DataFrame, top_p: float = DIAG_TOP_P) -> pd.DataFrame:
    """
    (a) Timeouts left out: the TP share of decided trades (TP or SL) among
    entries with p >= top_p, against the share that ALL decided entries at
    the same entry hours and direction had in the same period (no model).
    """
    rows = []
    for rn, g in stacked.groupby("round"):
        d = g[g["outcome"].isin(["TP", "SL"])]
        rate = d.groupby(["side", "hour"])["win"].mean()                  # no model, same period
        top = d[d["p"] >= top_p]
        n = len(top)
        if n == 0:
            rows.append({"round": rn, "decided_top": 0, "tp_rate": np.nan, "no_model_rate": np.nan,
                         "difference": np.nan, "z": np.nan})
            continue
        expected = rate.reindex(pd.MultiIndex.from_arrays([top["side"], top["hour"]])).to_numpy()
        e = float(expected.mean())
        r = float(top["win"].mean())
        se = np.sqrt(np.mean(expected * (1 - expected)) / n)
        rows.append({"round": rn, "decided_top": n, "tp_rate": r, "no_model_rate": e,
                     "difference": r - e, "z": (r - e) / se if se > 0 else np.nan})
    return pd.DataFrame(rows)


def band_table(stacked: pd.DataFrame, band: float = config.CALIBRATION_BAND) -> pd.DataFrame:
    """All periods pooled: decided TP share by probability band vs the hour-matched no-model share."""
    d = stacked[stacked["outcome"].isin(["TP", "SL"])].copy()
    rate = d.groupby(["round", "side", "hour"])["win"].transform("mean")   # no model, same period/hour/side
    d["expected"] = rate
    d["band"] = np.floor(d["p"] / band) * band
    return d.groupby("band").agg(decided=("win", "size"), tp_rate=("win", "mean"),
                                 no_model_rate=("expected", "mean"))


def side_agreement(pred: pd.DataFrame, top_p: float = DIAG_TOP_P) -> pd.DataFrame:
    """(b) How closely the buy and sell probabilities move together on the same bars."""
    rows = []
    hours = _hour_mut(pred["entry_utc"])
    for rn, idx in pd.Series(np.arange(len(pred))).groupby(pred["round"].to_numpy()):
        g = pred.iloc[idx.to_numpy()]
        h = hours[idx.to_numpy()]
        b, s_ = g["buy_p"], g["sell_p"]
        # the same after removing each entry hour's average (is it more than time of day?)
        rb = b - b.groupby(h).transform("mean").to_numpy()
        rs = s_ - s_.groupby(h).transform("mean").to_numpy()
        both = (b >= top_p) & (s_ >= top_p)
        one = (b >= top_p) ^ (s_ >= top_p)
        rows.append({"round": rn, "corr": b.corr(s_), "rank_corr": b.corr(s_, method="spearman"),
                     "corr_within_hour": rb.corr(rs), "both_top": float(both.mean()),
                     "one_top": float(one.mean())})
    return pd.DataFrame(rows)


def diagnose(pred: pd.DataFrame, df: pd.DataFrame, rounds: list) -> list[str]:
    variants = [v for v in config.VARIANTS if v in set(pred["variant"])]
    lines = [f"Diagnostics on saved test predictions (no retraining). 'Top' = p >= {100 * DIAG_TOP_P:.0f}% "
             f"(fixed in advance: where signals come from).", ""]
    lines.append("1. Log loss: model vs 'no model' (one rate) vs 'no model by entry hour' "
                 "(rates per hour and direction, calibration period)")
    for v in variants:
        hb = hour_baseline_losses(pred[pred["variant"] == v], df, rounds)
        beat = int((hb["model"] < hb["no_model_by_hour"]).sum())
        lines.append(f"  Variant {v}: beats 'no model by hour' in {beat} of {len(hb)} periods")
        for _, r in hb.iterrows():
            lines.append(f"    round {int(r['round'])} ({r['period']}): model {r['model']:.4f}  "
                         f"no model {r['no_model']:.4f}  no model by hour {r['no_model_by_hour']:.4f}")
    lines.append("")
    lines.append("2. Predicted EV vs R actually obtained (all entries, both directions, all test periods)")
    for v in variants:
        t = ev_vs_realised(stack_sides(pred[pred["variant"] == v]))
        lines.append(f"  Variant {v}:   EV band        entries  predicted  realised")
        for band, r in t.iterrows():
            lines.append(f"    {str(band):>14} {int(r['entries']):>9} {r['predicted_ev']:>+10.3f} {r['realised_r']:>+9.3f}")
    lines.append("")
    lines.append("3a. Direction test, timeouts left out: TP share of decided top-band entries vs the share of")
    lines.append("    ALL decided entries at the same hours and direction in the same period (no model)")
    for v in variants:
        st = stack_sides(pred[pred["variant"] == v])
        dt = direction_test(st)
        valid = dt.dropna(subset=["difference"])
        beat = int((valid["difference"] > 0).sum())
        clear = int((valid["z"] >= 2).sum())
        empty = len(dt) - len(valid)
        lines.append(f"  Variant {v}: top band above no model in {beat} of {len(dt)} periods "
                     f"({clear} by at least 2 standard errors)"
                     + (f"; {empty} period(s) had no decided top-band entries" if empty else ""))
        for _, r in dt.iterrows():
            if r["decided_top"] == 0:
                lines.append(f"    round {int(r['round'])}: no decided top-band entries")
                continue
            lines.append(f"    round {int(r['round'])}: {int(r['decided_top']):>5} decided  TP {100 * r['tp_rate']:5.1f}%  "
                         f"no model {100 * r['no_model_rate']:5.1f}%  difference {100 * r['difference']:+5.1f} points  "
                         f"(z {r['z']:+.1f})")
        info = direction_test(st, DIAG_INFO_P).dropna(subset=["difference"])
        lines.append(f"    (information: p >= {100 * DIAG_INFO_P:.0f}% above no model in "
                     f"{int((info['difference'] > 0).sum())} of {len(dt)} periods)")
        bt = band_table(st)
        lines.append(f"    all periods by band:  band     decided    TP   no model")
        for b, r in bt.iterrows():
            lines.append(f"      {100 * b:3.0f}-{100 * (b + config.CALIBRATION_BAND):3.0f}% {int(r['decided']):>10} "
                         f"{100 * r['tp_rate']:5.1f}% {100 * r['no_model_rate']:6.1f}%")
    lines.append("")
    lines.append("3b. Do buy and sell probabilities move together on the same bars?")
    lines.append("    (positive = both high at the same time: 'will price move enough', not 'which way')")
    for v in variants:
        sa = side_agreement(pred[pred["variant"] == v])
        lines.append(f"  Variant {v}: average correlation {sa['corr'].mean():+.2f}, within the same entry hour "
                     f"{sa['corr_within_hour'].mean():+.2f}; bars with BOTH >= {100 * DIAG_TOP_P:.0f}%: "
                     f"{100 * sa['both_top'].mean():.1f}%, only one: {100 * sa['one_top'].mean():.1f}%")
        for _, r in sa.iterrows():
            lines.append(f"    round {int(r['round'])}: corr {r['corr']:+.2f}  rank {r['rank_corr']:+.2f}  "
                         f"within hour {r['corr_within_hour']:+.2f}  both top {100 * r['both_top']:4.1f}%  "
                         f"one top {100 * r['one_top']:4.1f}%")
    return lines


def latest_predictions(instrument: str) -> Path:
    files = sorted((config.DATA_DIR / "walkforward").glob(f"{instrument}_*_predictions.parquet"))
    if not files:
        raise FileNotFoundError("no saved walk-forward predictions: run  python train.py walkforward")
    return files[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward training and testing.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("walkforward", help="all variants, all rounds, report (models and predictions saved)")
    p = sub.add_parser("diagnose", help="diagnostics on the saved test predictions (no retraining)")
    p.add_argument("--predictions", type=Path, default=None, help="predictions file (default: the latest)")
    for sp in sub.choices.values():
        sp.add_argument("--instrument", default="EURUSD", choices=list(config.IG_SPREAD_PIPS))
    args = parser.parse_args(argv)
    timeutils.configure_logging()
    try:
        df, groups, meta = load_dataset(args.instrument)
        if args.command == "diagnose":
            path = args.predictions or latest_predictions(args.instrument)
            pred = pd.read_parquet(path)
    except (FileNotFoundError, fetch_data.HistoryAccessError) as exc:
        print(f"STOPPED: {exc}")
        return 1
    if args.command == "diagnose":
        rounds = schedule(pd.Timestamp(fetch_data.development_end_utc()))
        text = "\n".join([f"Predictions: {path.name}"] + diagnose(pred, df, rounds))
        print(text)
        reports = config.DATA_DIR / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        out = reports / f"diagnose_{args.instrument}_{timeutils.now_utc():%Y-%m-%d_%H%M}UTC.txt"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved to {out}")
        return 0
    res = run_walkforward(df, groups)
    text = "\n".join(report(res, args.instrument))
    print(text)
    run_dir = save_run(res, args.instrument, meta, text)
    print(f"\nModels: {run_dir}\nPredictions, trades: {config.DATA_DIR / 'walkforward'}\n"
          f"Report: {config.DATA_DIR / 'reports'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
