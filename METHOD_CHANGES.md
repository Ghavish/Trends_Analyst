# Method-change log

Every change to how the model is built, judged or traded after test results
have been seen, with the reason. Changing the method after looking at test
periods makes it easier to fool ourselves (plan Section 11.1), so the number
of attempts is kept visible here. The final-check period (May-October 2026)
stays locked throughout and remains the honest judge.

| # | Date | Change | Why | Results seen before? |
|---|---|---|---|---|
| 0 | 15 Nov 2026 | Baseline method: labels (1-hour ATR barriers, entries 17:00-20:00 MUT, closed by 22:00 MUT), 46 features (features v2: 5 redundant time features dropped after the feature report), walk-forward 8 x 6 months, LightGBM per direction, Platt calibration, EV = p x avg R of winners + (1-p) x avg R of non-winners, signal at EV >= +0.10 R | Decided in advance (design questions of step 2) | No model results yet |
| 1 | 15 Nov 2026 | No method change. First real run (walkforward_EURUSD_2026-11-15_1748UTC): every variant loses (-0.10 to -0.13 R per trade, 95% intervals below 0, worse than taking every entry); by the agreed rule both groups are kept (candles 8/8, 1-hour trend 6/8), so variant D | - | Yes |
| 2 | 15 Nov 2026 | No method change: diagnostics added on the saved predictions (`python train.py diagnose`): log loss vs a no-model rate per entry hour, predicted EV vs realised R, direction test (timeouts left out, top band p >= 35% fixed in advance, against the no-model rate at the same hours), buy/sell probability agreement | To find out whether the model predicts direction at all before changing the EV formula | Yes (run 1) |
| 3 | 15 Nov 2026 | No method change. Diagnostics result (diagnose_EURUSD_2026-11-15_1805UTC): the model beats a per-hour win rate in only 4 of 8 periods (all variants); predicted EV has almost no link to realised R; direction test not met under the definition fixed in advance ("clearly" = z >= 2, "most" = at least 5 of 8 periods): A 0, B 1, C 3, D 3 periods with z >= 2. Pooled D top band 37.8% TP vs 34.8% for the same hours (about +3 points, z overstated by overlapping entries, volatility not controlled), below the ~41% break-even. Buy/sell probabilities correlate about +0.5 (+0.1 to +0.15 within the hour). Decision: stop changing the EV formula (option 3 not pursued) | Rule agreed in advance | Yes (runs 1-2) |
| 4 | 16 Nov 2026 | No method change: cost report added (`python backtest.py costs`): no model, every entry under today's rule and under longer holding (4-hour / daily ATR stops, 1, 2, 5 weekdays, closed 22:00 MUT, never after Friday), with spreads by hour (recorded IG; night hours estimated, and doubled as a second case), stop slippage (more outside your window) and overnight funding (EUR/USD from the IG platform; indices provisional worst case). Indices may be read for cost analysis only (no model) | To check whether costs, not features, are the main obstacle, before adding features | Yes (runs 1-3) |

Decision agreed before running the diagnostics (15 Nov 2026): if the top
bands clearly beat the no-model rate in most periods, design a three-outcome
model (TP / SL / timeout, plan Section 7.8) together; if not, stop changing
the EV formula and discuss what new information could predict direction.

Next (decided 15 Nov 2026, after #3): check costs before adding features -
(a) EUR/USD with longer holding (stops from 4-hour / daily ATR, trades of a
day or more, overnight funding included); (b) US 500 and Germany 40 with the
same report. Continue where the break-even rate falls meaningfully; then test
other markets as a feature group; the economic calendar as a news filter
(and IG's spread around the 17:30 and 19:00 MUT releases). London fix
skipped. FRED and longer-term context kept for the longer-holding test.
