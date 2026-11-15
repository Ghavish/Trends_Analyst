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

Decision agreed before running the diagnostics (15 Nov 2026): if the top
bands clearly beat the no-model rate in most periods, design a three-outcome
model (TP / SL / timeout, plan Section 7.8) together; if not, stop changing
the EV formula and discuss what new information could predict direction.
