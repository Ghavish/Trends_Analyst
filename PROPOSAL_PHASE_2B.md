# Proposal: Phase 2B - one decision per instrument per day, held about a day

Status: **adopted as an experiment on 22 November 2026** (your decisions in
section 8, fixed before any results). Adopting it changes decisions in plan
Section 2 (trading style, decisions per bar), so the plan PDF should be
updated; every change is logged in `METHOD_CHANGES.md`.

Not financial advice; personal use only.

---

## 1. Why consider it (evidence so far)

The intraday model (Phase 2, steps 2a-2c) found no edge after IG's costs
(method-change log #1-#3). The cost reports of 22 Nov 2026 then compared
holding rules **without any model**, every entry in your window taken.

Cost relative to the typical move (cost per trade / standard deviation of
trade results; smaller = less skill needed to beat the costs):

| Instrument | Today's rule (1-hour ATR, same day) | Daily-ATR stop, held 1 day: buy | sell |
|---|---|---|---|
| EUR/USD | 0.073 | 0.059 | 0.035 |
| US 500 | 0.051 | 0.033 | 0.023 |
| Germany 40 | 0.053 | 0.037 | 0.027 |

- Holding about a day lowers the cost hurdle by about **20-55%**: about half
  for sells, less for buys because buys pay more overnight funding
  (EUR/USD 1.36 vs 0.47 pips a night; the index figures are a provisional
  worst case until your demo positions confirm them).
- 4-hour-ATR stops do not help (funding eats the saving); 5-day holds are
  worse than 1 day for buys (more funding).
- With daily-ATR stops held 1 day, **82-84% of trades end at the time limit**,
  so the triple-barrier TP/SL hardly decides anything: it is really a
  "hold for about a day" trade with a wide protective stop.
- The indices are cheaper than EUR/USD under every rule.

## 2. The catch: how much data a test needs

Entries on the same evening share most of their price path, so the honest
unit is **one independent decision per instrument per trading day**, about
**240 a year** after holidays (about 190 if Fridays are left out).

To show that a model really earns more than its costs, its edge after costs
must stand about 2 standard errors clear of zero. With N independent
decisions, the smallest edge that can be confirmed is about **2 / sqrt(N)**
standard deviations of a trade's result.

| Test data available | Decisions per instrument | Smallest confirmable edge after costs (one instrument) | Pooled over 5 instruments* |
|---|---|---|---|
| Walk-forward tests Jul 2022 - Apr 2026 (3.8 years, history from 2019) | ~920 | 0.066 sd | ~0.042 sd |
| Same, with history from 2015 (tests from Jul 2018, 7.8 years) | ~1,890 | 0.046 sd | ~0.029 sd |
| Final-check period alone (6 months) | ~120 | 0.18 sd | ~0.12 sd |

\*EUR/USD and GBP/USD move together, and so do US 500 and Germany 40, so five
instruments count as roughly 2-3 independent ones; the pooled figures assume 2.5.

So a model would need a **gross** skill of about cost/sd + the confirmable
edge, e.g. for EUR/USD sells 0.035 + 0.066 = about **0.10 sd** on one
instrument with today's history, or about 0.06-0.08 sd pooled. That is a
lower bar than the intraday rule (0.073 + 0.066 = 0.14 sd), but still a
demanding one for predicting next-day direction. Two consequences:

- **Pool the instruments in one model** (plan Section 4.4) and judge results
  pooled, with per-instrument results shown for information.
- **Extending the history to 2015** roughly doubles the test record; it is
  the cheapest way to make a modest edge detectable (about 40 hours of
  downloading for all five instruments, plus the full 2019+ history still
  missing for GBP/USD and USD/JPY, about 35 hours).
- The 6-month final check can only catch large failures; the walk-forward
  record carries the evidence.

## 3. What would change (plan Section 2 and 7)

| Area | Now | Proposed |
|---|---|---|
| Trading style | Day trading, no overnight positions | Positions held about a day, overnight; closed before the weekend |
| Decisions | Every 5-minute bar 17:00-20:00 MUT | **One per instrument per day**, at a fixed time in your window |
| Exit | TP / SL / 22:00 MUT same day | At the **next day's decision time** (or kept, if the new decision is the same direction: no new spread); protective stop at IG; Friday: closed by 22:00 MUT |
| Label | TP before SL (triple barrier) | See section 4 |
| Bars | 5-minute decisions, 1-minute exit checks | Daily decisions; 1-minute exit checks kept (stops at night) |
| Costs | Spread, stop slippage | Spread, stop slippage (more at night), **overnight funding** |
| Unchanged | IG, the five instruments, LightGBM + calibration, walk-forward only, final-check lock, engine/dashboard split, risk limits, kill switch |

Every action stays inside your window: decisions, entries and time-based
exits happen at the decision time; at night only the protective stop at IG
can act.

## 4. The label (recommendation and alternatives)

- **A. Next-day trade result sign (recommended).** For each decision, a buy
  held to the next decision time with the protective stop: label = 1 if its
  result in R (after spread, slippage and funding) is above 0. A mirror label
  for a sell. Same calibrated-probability approach as now (Platt), EV =
  p x average R of winners + (1-p) x average R of losers from the
  calibration period. With a fixed holding period the earlier EV problem
  (non-winners' average depending on time left) largely disappears.
- **B. Expected R directly (regression).** Uses the size of moves, not just
  the sign; but no calibrated probability, and noisier to judge.
- **C. Keep the triple barrier with daily-ATR stops.** Not recommended:
  82-84% timeouts make the TP-first label mostly about running out of time.

Protective stop: 1 x daily ATR(14) (hit on 12-16% of 1-day holds in the
cost report), so the stop protects against large moves without deciding most
trades. No take profit, or a wide one (to decide with you).

## 5. Features (daily and multi-day; all from completed data)

Distances in daily-ATR units; groups switchable and judged by the same rule
(lower test log loss in at least 6 of 8 periods).

- **base_daily:** price change over 1, 2, 5, 10, 20 days; distance from 20-,
  50- and 100-day EMAs; position in the 20-day range; daily ATR vs its
  60-day average; yesterday's range and body; distance from yesterday's high
  and low; today's move up to the decision time; weekday.
- **cross_market:** the other instruments' 1- and 5-day changes (broad
  dollar moves from EUR/USD, GBP/USD, USD/JPY; risk appetite from US 500 and
  Germany 40). Needs GBP/USD and USD/JPY full history.
- **rates (needs your approval: FRED):** daily changes in US and German
  2-year yields and their difference, using only values published before the
  decision.
- **daily candles:** the daily candle shapes (replacing the 5-minute group).
- **news filter (not a feature yet):** skip or flag days with high-impact
  releases before the next decision, once a free calendar is chosen; plus
  IG's spread around 17:30 and 19:00 MUT from your recordings.

The intraday time features drop out (the decision time is fixed).

## 6. Testing and pass criteria (to fix before any results)

- Walk-forward as now: expanding training, 6-month calibration, 6-month
  test, 1 trading day gap; one pooled model with an instrument feature.
- Portfolio rules applied in the backtest: one position per instrument, at
  most 2 open, none in the same correlated group, 1% risk, 3% daily loss limit.
- Proposed pass criteria (your decision): pooled average R after costs above
  0 with the 95% day-resampled interval above 0; positive in at least 5 of 8
  test periods; no single instrument carrying the result alone.

## 7. Before any code: your decisions

1. Adopt holding about a day (plan Section 2 update)? Or keep day trading
   and stop here with forex/indices intraday?
2. Decision time: one fixed time for all (e.g. **17:15 MUT**, early in your
   window, before the 17:30 MUT US data in winter), or per instrument?
3. Fridays: no new positions on Fridays (recommended: they would be held only
   until 22:00 MUT), or same-day Friday trades?
4. Label A (recommended), B or C; protective stop 1 x daily ATR; take profit
   none or wide?
5. Data: extend all five instruments back to 2015, and download the full
   GBP/USD and USD/JPY history (about 75 hours in total)?
6. Feature groups: base_daily, cross_market, daily candles; rates (FRED)
   yes or no?
7. Pass criteria as in section 6?

Prerequisites already under way: the overnight recording week (needed for
the indices' overnight clearance and real night spreads), the demo funding
figures, the futures spreads, and the 5 December confirmation comparison.

---

## 8. Decisions of 22 November 2026 (fixed before any results)

**Stopping rule.** This is an experiment. If it fails the fixed criteria
below, this line of work stops; no variants are tried afterwards.

1. **Holding about a day**, as an experiment.
2. **One decision per instrument per weekday at 17:15 MUT** (13:15 UTC all
   year). It is before the 08:30 New York US data releases only in winter
   (17:30 MUT); in summer those releases come 45 minutes before it
   (16:30 MUT).
3. **No new positions on Fridays.** Thursday's positions close at Friday's
   decision time.
4. **Label A**: a buy (and, separately, a sell) entered at the decision time
   and closed at the next decision time, with a **protective stop at 1 x
   daily ATR(14)** and a **take profit at 3 x daily ATR(14)** (Phase 1 rule:
   every order carries both); label = 1 if its result in R after spread,
   slippage and funding is above 0. The labels use the same 3 x level. The
   report keeps the **predicted EV vs realised R table by EV band**.
5. **EUR/USD first**, with two rules:
   - (a) The EUR/USD-only test **can stop the project but cannot pass it**;
     the pass verdict comes only from the full pooled test. "Stop" means:
     the model does not beat the same-direction baseline on test log loss in
     most periods, **or** its average-R interval lies entirely below that
     baseline.
   - (b) **History extended to 2015 for all five instruments**, decided now,
     not after results; the only exception is Dukascopy lacking the data.
6. **Feature groups:** base_daily and cross_market first; rates approved but
   tested last (FRED has no daily German 2-year yield; the Bundesbank
   publishes one daily, series BBSSY.D.REN.EUR.A610.000000WT0202.A, to be
   checked then); daily candles skipped.
7. **Pass criteria** (full pooled test), section 6 plus:
   - (a) beat a baseline that always trades each instrument in the
     direction that was most profitable in its training data, with the same
     day-resampled 95% interval (the indices rose over 2015-2026);
   - (b) choose the feature set by the log-loss rule first, then apply the
     profit test **once**, to that one model;
   - (c) period rules stated as shares (with 2015 history there are about 16
     test periods);
   - (d) "no single instrument carries it" = the pooled average R stays
     above 0 with any one instrument removed;
   - (e) when the portfolio limits allow fewer positions than signals, the
     highest-EV signals are taken (fixed in advance).
