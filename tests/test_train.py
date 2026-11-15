"""
Tests for train.py (walk-forward training, Phase 2 step 2c).
Synthetic random-walk prices (no real edge), short 1-month periods and few
trees so the tests are fast. Temporary folders only.
"""

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

import config
import features as ft
import fetch_data
import labels as lb
import train as tr
from tests.test_feature_set import market_minutes, random_bars

UTC = timezone.utc
SMALL = {**config.LGBM_PARAMS, "min_data_in_leaf": 50, "num_threads": 2}
TREES = 30


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "data" / "datasets")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "DUKASCOPY_BARS_1M_DIR", tmp_path / "data" / "dukascopy" / "bars_1m")
    yield tmp_path


HIST = random_bars(market_minutes("2025-01-06", 175), seed=3)       # ~6 months
LAB, _ = lb.build_labels(HIST, "EURUSD")
FEATS, GROUPS = ft.compute_features(ft.to_indexed(HIST))
DF = tr.prepare(LAB, ft.window_rows(FEATS).reset_index())
ROUNDS = tr.schedule(pd.Timestamp("2025-06-28"), first_test_start=datetime(2025, 4, 1, tzinfo=UTC), months=1)


@pytest.fixture(scope="module")
def run():
    return tr.run_walkforward(DF, GROUPS, rounds=ROUNDS, params=SMALL, trees=TREES)


# --------------------------------------------------------------------------
# Schedule and splits
# --------------------------------------------------------------------------
def test_real_schedule_has_eight_six_month_rounds():
    r = tr.schedule(pd.Timestamp(fetch_data.development_end_utc()))
    assert len(r) == 8
    assert (str(r[0].cal_start.date()), str(r[0].test_start.date()), str(r[0].test_end.date())) == \
        ("2022-01-01", "2022-07-01", "2023-01-01")
    assert r[0].label() == "2022-07-01 to 2022-12-31"
    assert (str(r[-1].test_start.date()), str(r[-1].test_end.date())) == ("2026-01-01", "2026-04-30")


def test_splits_are_ordered_disjoint_with_a_trading_day_gap():
    for rnd in ROUNDS:
        train, cal, test = tr.split(DF, rnd)
        assert train["day"].max() < cal["day"].min() < test["day"].min()
        days = np.sort(DF["day"].unique())
        # exactly one trading day left out before calibration and before test
        between_tc = days[(days > train["day"].max().to_datetime64()) & (days < cal["day"].min().to_datetime64())]
        between_ct = days[(days > cal["day"].max().to_datetime64()) & (days < test["day"].min().to_datetime64())]
        assert len(between_tc) == 1 and len(between_ct) == 1
        assert test["day"].max() < rnd.test_end


def test_refuses_data_past_the_development_period(monkeypatch):
    late = LAB.copy()
    late["close_by_utc"] = pd.Timestamp("2026-06-01 18:00", tz="UTC")
    lb_path, ft_path = lb.label_file("EURUSD"), ft.feature_file("EURUSD")
    lb_path.parent.mkdir(parents=True, exist_ok=True)
    late.to_parquet(lb_path)
    lb_path.with_suffix(".json").write_text(json.dumps({"x": 1}))
    ft.window_rows(FEATS).reset_index().to_parquet(ft_path)
    ft_path.with_suffix(".json").write_text(json.dumps({"groups": GROUPS}))
    with pytest.raises(fetch_data.HistoryAccessError):
        tr.load_dataset("EURUSD")


# --------------------------------------------------------------------------
# Model pieces
# --------------------------------------------------------------------------
def test_training_is_reproducible():
    rnd = ROUNDS[0]
    train, cal, test = tr.split(DF, rnd)
    cols = GROUPS["base"]
    a = tr.fit_direction("buy", cols, train, cal, SMALL, TREES).predict(test)
    b = tr.fit_direction("buy", cols, train, cal, SMALL, TREES).predict(test)
    pd.testing.assert_frame_equal(a, b)


def test_platt_recovers_a_known_calibration():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    raw = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.5 * raw - 1.0)))).astype(int)
    m = LogisticRegression(C=np.inf).fit(raw.reshape(-1, 1), y)
    assert m.coef_[0, 0] == pytest.approx(0.5, abs=0.05) and m.intercept_[0] == pytest.approx(-1.0, abs=0.05)


def test_expected_value_uses_calibration_averages():
    rnd = ROUNDS[0]
    train, cal, test = tr.split(DF, rnd)
    m = tr.fit_direction("sell", GROUPS["base"], train, cal, SMALL, TREES)
    r, w = cal["sell_result_r"], cal["sell_win"] == 1
    assert m.avg_win_r == pytest.approx(r[w].mean()) and m.avg_nonwin_r == pytest.approx(r[~w].mean())
    pr = m.predict(test)
    assert np.allclose(pr["ev"], pr["p"] * m.avg_win_r + (1 - pr["p"]) * m.avg_nonwin_r)
    assert ((pr["p"] > 0) & (pr["p"] < 1)).all()


def test_both_directions_qualifying_means_skip():
    pred = pd.DataFrame({"buy_ev": [0.2, 0.2, -0.1, -0.1], "sell_ev": [0.2, -0.1, 0.3, 0.05]})
    assert list(tr.choose_signals(pred, 0.10)) == ["", "buy", "sell", ""]


def test_one_trade_at_a_time():
    t0 = pd.Timestamp("2025-03-05 13:00", tz="UTC")
    idx = [t0 + pd.Timedelta(minutes=5 * k) for k in range(4)]
    test = pd.DataFrame({"entry_utc": [i + pd.Timedelta(minutes=5) for i in idx], "day": pd.Timestamp("2025-03-05"),
                         "buy_outcome": "TP", "buy_win": 1, "buy_result_r": 1.5,
                         "buy_exit_utc": [t0 + pd.Timedelta(minutes=17)] * 4,
                         "sell_outcome": "SL", "sell_win": 0, "sell_result_r": -1.0,
                         "sell_exit_utc": [t0 + pd.Timedelta(hours=2)] * 4}, index=idx)
    trades = tr.take_trades(test, pd.Series(["buy", "buy", "buy", "sell"], index=idx))
    # the first buy runs to 13:17: the 13:10 entry is blocked, 13:15 too; 13:20 entry (sell) is taken
    assert list(trades["entry_utc"].dt.strftime("%H:%M")) == ["13:05", "13:20"]
    assert list(trades["side"]) == ["buy", "sell"]


def test_max_drawdown():
    assert tr.max_drawdown(pd.Series([1, -1, -1, 2, -3, 1])) == -3
    assert tr.max_drawdown(pd.Series([], dtype=float)) == 0


def test_bootstrap_resamples_whole_days():
    # Day 1: ten trades at +1 R; day 2: one trade at -1 R. Resampling DAYS can give
    # -1 (both picks day 2) up to +1 (both day 1); resampling single trades could not reach -1.
    trades = pd.DataFrame({"day": [pd.Timestamp("2025-03-05")] * 10 + [pd.Timestamp("2025-03-06")],
                           "result_r": [1.0] * 10 + [-1.0]})
    lo, hi = tr.bootstrap_mean_r(trades, resamples=4000, seed=1)
    assert lo == pytest.approx(-1.0) and hi == pytest.approx(1.0)
    assert tr.bootstrap_mean_r(trades, 500, 7) == tr.bootstrap_mean_r(trades, 500, 7)    # fixed seed


def test_group_rule_needs_six_of_eight():
    losses = pd.DataFrame({"A": [0.60] * 8, "B": [0.59] * 6 + [0.61] * 2, "C": [0.59] * 5 + [0.61] * 3})
    d = tr.group_decision(losses)
    assert d["candles"] == {"better_in": 6, "periods": 8, "keep": True}
    assert d["trend_1h"] == {"better_in": 5, "periods": 8, "keep": False}


def test_log_loss_and_brier():
    assert tr.log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))
    assert tr.brier([1, 0], [1, 0]) == 0


# --------------------------------------------------------------------------
# The whole run
# --------------------------------------------------------------------------
def test_run_covers_every_round_variant_and_direction(run):
    assert set(run.losses.columns) == set(config.VARIANTS) and len(run.losses) == len(ROUNDS)
    assert set(run.predictions["variant"]) == set(config.VARIANTS)
    assert len(run.models) == len(ROUNDS) * len(config.VARIANTS) * 2
    # every test entry is predicted once per variant
    n_test = sum(len(tr.split(DF, r)[2]) for r in ROUNDS)
    assert (run.predictions.groupby("variant").size() == n_test).all()
    # variants differ only in their features
    a = run.models[(1, "A", "buy")].features
    d = run.models[(1, "D", "buy")].features
    assert set(a) == set(GROUPS["base"]) and set(d) == set(GROUPS["base"]) | set(GROUPS["candles"]) | set(GROUPS["trend_1h"])


def test_trades_follow_the_rules(run):
    t = run.trades
    if len(t):
        p = run.predictions.reset_index().set_index(["variant", "signal_bar_utc"])
        for _, row in t.iterrows():
            ev = p.loc[(row["variant"], row["signal_bar_utc"])]
            assert ev[f"{row['side']}_ev"] >= config.SIGNAL_MIN_EV_R
        for (_, _), g in t.groupby(["variant", "round"]):
            assert (g["entry_utc"].iloc[1:].to_numpy() >= g["exit_utc"].iloc[:-1].to_numpy()).all()


def test_baseline_is_the_average_of_all_entries(run):
    rnd = ROUNDS[0]
    test = tr.split(DF, rnd)[2]
    b = run.baseline[(run.baseline["round"] == 1) & (run.baseline["side"] == "buy")]
    assert b["avg_r"].iloc[0] == pytest.approx(test["buy_result_r"].mean())


def test_report_and_saving(run):
    text = "\n".join(tr.report(run, "EURUSD"))
    for part in ("1. Test-period log loss", "candles: lower log loss in", "2. Calibration", "isotonic",
                 "3. Trades", "no model, every buy entry", "4. All test periods together",
                 "95% interval for avg R (resampling whole days)", "5. Other EV thresholds", "6. Most used features"):
        assert part in text, part
    run_dir = tr.save_run(run, "EURUSD", {"x": 1}, text)
    assert (run_dir / "round1_A_buy.txt").exists() and (run_dir / "calibration.json").exists()
    assert len(list((config.DATA_DIR / "walkforward").glob("EURUSD_*_predictions.parquet"))) == 1
    assert len(list((config.DATA_DIR / "reports").glob("walkforward_EURUSD_*.txt"))) == 1
    # A saved model gives the same raw scores when loaded again.
    import lightgbm as lgb
    m = run.models[(1, "A", "buy")]
    test = tr.split(DF, ROUNDS[0])[2]
    loaded = lgb.Booster(model_file=str(run_dir / "round1_A_buy.txt"))
    assert np.allclose(loaded.predict(test[m.features], raw_score=True), m.raw(test))


def test_cli_end_to_end(monkeypatch, capsys):
    tt = pd.to_datetime(HIST["time_utc"], utc=True)
    for (y, mth), part in HIST.groupby([tt.dt.year, tt.dt.month]):
        fetch_data.write_parquet_atomic(part.reset_index(drop=True), fetch_data.history_bar_file("EURUSD", y, mth))
    lb.build_and_save("EURUSD")
    ft.build_and_save("EURUSD")
    monkeypatch.setattr(config, "WF_FIRST_TEST_START", datetime(2025, 5, 1, tzinfo=UTC))
    monkeypatch.setattr(config, "WF_PERIOD_MONTHS", 1)
    monkeypatch.setattr(config, "LGBM_TREES", TREES)
    monkeypatch.setattr(config, "LGBM_PARAMS", SMALL)
    monkeypatch.setattr(tr.timeutils, "configure_logging", lambda **kw: None)
    monkeypatch.setattr(tr.fetch_data, "development_end_utc", lambda: datetime(2025, 6, 28, tzinfo=UTC))
    assert tr.main(["walkforward"]) == 0
    out = capsys.readouterr().out
    assert "Walk-forward: EURUSD, 2 rounds" in out and "Models:" in out


# --------------------------------------------------------------------------
# Diagnostics on saved predictions (no retraining)
# --------------------------------------------------------------------------
def _entry(h_mut, k=0):
    """An entry time at hour h (Mauritius) on a fixed day; k shifts it by 5 minutes."""
    return pd.Timestamp("2025-03-05", tz="UTC") + pd.Timedelta(hours=h_mut - 4, minutes=5 * k)


def test_direction_test_compares_with_same_hours_and_ignores_timeouts():
    rows = []
    # Hour 17, buy: 10 decided entries, 4 TP (40%); the 4 with p >= 0.35 contain 3 TP (75%).
    for k in range(10):
        rows.append({"round": 1, "side": "buy", "hour": 17, "p": 0.5 if k < 4 else 0.2,
                     "outcome": "TP" if k in (0, 1, 2, 5) else "SL"})
    # Timeouts with high p must not count.
    rows += [{"round": 1, "side": "buy", "hour": 17, "p": 0.9, "outcome": "TIMEOUT"}] * 5
    st = pd.DataFrame(rows)
    st["win"] = (st["outcome"] == "TP").astype(int)
    dt = tr.direction_test(st, 0.35)
    r = dt.iloc[0]
    assert r["decided_top"] == 4 and r["tp_rate"] == pytest.approx(0.75)
    assert r["no_model_rate"] == pytest.approx(0.4) and r["difference"] == pytest.approx(0.35)
    assert r["z"] == pytest.approx(0.35 / np.sqrt(0.4 * 0.6 / 4))


def test_direction_test_weights_no_model_rate_by_the_top_bands_hours():
    rows = []
    for k in range(10):                      # hour 17: TP share 50%
        rows.append({"round": 1, "side": "sell", "hour": 17, "p": 0.1, "outcome": "TP" if k < 5 else "SL"})
    for k in range(10):                      # hour 20: TP share 10%; top entries only here
        rows.append({"round": 1, "side": "sell", "hour": 20, "p": 0.4 if k < 5 else 0.1,
                     "outcome": "TP" if k == 0 else "SL"})
    st = pd.DataFrame(rows)
    st["win"] = (st["outcome"] == "TP").astype(int)
    r = tr.direction_test(st, 0.35).iloc[0]
    assert r["no_model_rate"] == pytest.approx(0.1)        # the hour-20 rate, not the overall 30%
    assert r["tp_rate"] == pytest.approx(0.2)


def test_side_agreement():
    n = 200
    rng = np.random.default_rng(0)
    entries = [_entry(17 + (k % 4), k) for k in range(n)]
    x = rng.random(n)
    same = pd.DataFrame({"round": 1, "entry_utc": entries, "buy_p": x, "sell_p": x})
    opp = pd.DataFrame({"round": 1, "entry_utc": entries, "buy_p": x, "sell_p": 1 - x})
    assert tr.side_agreement(same)["corr"].iloc[0] == pytest.approx(1.0)
    assert tr.side_agreement(opp)["corr"].iloc[0] == pytest.approx(-1.0)
    both = pd.DataFrame({"round": 1, "entry_utc": entries[:4], "buy_p": [0.5, 0.5, 0.1, 0.1],
                         "sell_p": [0.5, 0.1, 0.5, 0.1]})
    r = tr.side_agreement(both, 0.35).iloc[0]
    assert r["both_top"] == pytest.approx(0.25) and r["one_top"] == pytest.approx(0.5)


def test_ev_vs_realised():
    st = pd.DataFrame({"ev": [0.15, 0.12, -0.05], "result_r": [-1.0, 1.5, 0.2]})
    t = tr.ev_vs_realised(st)
    top = t.loc[[i for i in t.index if i.left == 0.1][0]]
    assert top["entries"] == 2 and top["predicted_ev"] == pytest.approx(0.135) and top["realised_r"] == pytest.approx(0.25)


def test_hour_baseline_uses_calibration_rates_by_hour(run):
    pred = run.predictions[run.predictions["variant"] == "A"]
    hb = tr.hour_baseline_losses(pred, DF, ROUNDS)
    assert len(hb) == len(ROUNDS)
    # Recompute round 1 by hand.
    _, cal, _ = tr.split(DF, ROUNDS[0])
    pr = pred[pred["round"] == 1]
    ys, ps = [], []
    for side in ("buy", "sell"):
        by_h = cal.groupby(tr._hour_mut(cal["entry_utc"]))[f"{side}_win"].mean()
        ps.append(by_h.reindex(tr._hour_mut(pr["entry_utc"])).fillna(cal[f"{side}_win"].mean()).to_numpy())
        ys.append(pr[f"{side}_win"].to_numpy())
    assert hb.iloc[0]["no_model_by_hour"] == pytest.approx(tr.log_loss(np.concatenate(ys), np.concatenate(ps)))
    assert hb.iloc[0]["model"] == pytest.approx(run.losses.loc[1, "A"])


def test_diagnose_report_sections(run):
    text = "\n".join(tr.diagnose(run.predictions.reset_index(), DF, ROUNDS))
    for part in ("1. Log loss: model vs 'no model'", "beats 'no model by hour' in",
                 "2. Predicted EV vs R actually obtained", "3a. Direction test", "top band above no model in",
                 "all periods by band", "3b. Do buy and sell probabilities move together"):
        assert part in text, part


def test_cli_diagnose_after_walkforward(monkeypatch, capsys):
    test_cli_end_to_end(monkeypatch, capsys)
    assert tr.main(["diagnose"]) == 0
    out = capsys.readouterr().out
    assert "3a. Direction test" in out and "Saved to" in out
    assert len(list((config.DATA_DIR / "reports").glob("diagnose_EURUSD_*.txt"))) == 1
