# Roadmap: pending ideas

Source: *AI Trend Analyst: Three Borrowed Ideas* (15 November 2026), a
companion to the project plan. The project plan stays the single source of
truth; where they disagree, the plan wins until it is updated.

**Status: candidates, not decisions. None is built yet.** Each is built only
when its trigger is reached and you have confirmed, exactly as its section in
that document describes. Each is kept only if it passes the same tests as
the rest of the system (walk-forward / shadow mode). An adopted idea is then
added to the project plan (decisions table and relevant section).

**At the start of every step, check the triggers below.** When one is reached:
stop, tell you, confirm, then build.

| Idea | Trigger | Reached? |
|---|---|---|
| 2 · Opening-range features (indices) | Phase 2: the baseline walk-forward run works **and** the indices are cleared for training by the Dukascopy-vs-IG comparison | No (no walk-forward run yet; indices not cleared) |
| 1 · Risk checks (`risk.py`) | End of Phase 2: the engine's filter step (plan Section 8, step 4) is about to be built, before the first live signal | No |
| 3 · Reality check on trade cards | Start of Phase 3: trade cards are live on demo in signals-only mode | No |

Last checked: 15 November 2026, at Phase 2 step 2c (walk-forward training built, not yet run on real data). No trigger reached: the indices are not cleared and there is no real walk-forward result yet.

---

## Idea 2 · Opening-range features for US 500 and Germany 40 (document Section 5)

- US 500 range from the 09:30 New York cash open; Germany 40 from the 09:00
  Frankfurt open; found by converting UTC to local time (daylight saving handled).
- Range length configurable: default 15 minutes; test only 5, 15 and 30.
  Mid prices, completed bars only (a 15-minute range is usable from 09:45).
- Features, in ATR units: `or_complete`, `minutes_since_open`, `or_width_atr`,
  `dist_or_high_atr`, `dist_or_low_atr`, `or_position`, `or_break_state`,
  `minutes_since_break`, `or_reentries`. Left empty before the range is
  complete, on cash-market holidays and for forex.
- Holidays from a market calendar (a library such as pandas_market_calendars,
  to be checked, or a holiday list in `config.py`).
- Tested as a "baseline + opening range" variant in identical walk-forward
  rounds, judged per index; kept only if calibration and expected value
  improve consistently across rounds.
- Optional trade-card line, e.g. "Opening range: broke above 12 minutes ago, price 0.6 ATR over the high".

To raise when the trigger comes:
- The comparison should look specifically at the opening period. The US 500
  open (18:30 / 17:30 MUT) is inside your usual engine hours (17:00-22:00 MUT);
  the Frankfurt open (12:00 / 11:00 MUT) is not, so Germany 40 needs
  recordings at that time first.
- Holiday calendar: the `holidays` library (NYSE and London calendars,
  NYSE early closes) was adopted in step 2a; it also has other exchanges'
  calendars, to be checked for the Frankfurt (Xetra) holidays.

## Idea 1 · Risk checks in one place (document Section 4)

- New `risk.py` with one function (e.g. `check_trade(proposal, account_state, limits)`)
  that runs **every** check and returns PASS/BLOCK with value, limit and reason
  for each. Called by the engine's filter step and, in Phase 3, by `broker.py`
  just before an order. Shadow signals are checked too.
- Existing plan limits: kill switch / halt, session and news, spread, 1% risk
  per trade, per-instrument position cap, open trades (2; 1 per instrument;
  none in the same correlated group), 3% daily loss limit.
- New checks: trades per day, reward-to-risk after spread, entry window,
  IG minimum stop distance, IG deal size (round down; block if below the
  minimum), available funds.
- Limits in `config.py`; results saved through `db.py` in a `risk_checks`
  table; blocked reasons on the dashboard and in Telegram alerts
  ("Risk checks: 13/13 passed").
- pytest: each check just under / at / over its limit; several failures at
  once; the 17:00 New York reset in US standard and daylight time; deal-size
  rounding (round-down and block cases).

Values to decide (left blank so far): daily trade cap = ?, minimum
reward-to-risk after spread = ?, entry window = ? minutes before the
close-by time.

To raise when the trigger comes:
- The entry window overlaps the step-2a rule: entries 17:00-20:00 MUT, every
  trade closed by 22:00 MUT (config.ENTRY_LAST_MUT / CLOSE_BY_MUT); one
  setting should serve both.
- With a 10x-spread stop floor and TP 1.5 / SL 1, reward-to-risk after spread
  is at least about 1.27, so a 1.2-1.3 minimum would rarely bind.
- Still open from Phase 1: the per-instrument maximum lots; the news check
  needs the economic calendar (not chosen yet); IG's deal-size step and
  available funds must be confirmed in IG's API documentation.

## Idea 3 · Reality check on trade cards (document Section 6)

- Part A, session replay: 10,000 paths from the signal to its close-by time,
  sampling blocks of whole 5-minute bars (e.g. 6 bars; high, low, close
  relative to the open, in ATR units) from the same instrument at the same
  time of day over the previous 60 sessions; fixed, stored seed; the same TP,
  SL, time-limit and spread rules as the backtest. Result: % TP first,
  % SL first, % timed out.
- Part B, similar past setups: same instrument and direction, time of day
  within 30 minutes, same ATR third, same 5-point model-probability bucket;
  walk-forward test predictions only, data before the signal only, never the
  final-check period; count and uncertainty range; "too few to judge" under 30.
- One trade-card line, e.g. "Reality check: replay 41% · similar past setups
  44% (86 cases) · model 52%", with a flag when the model is more than 10
  points (configurable) above both. Informational only, never blocks.
- Results and seed saved with each signal through `db.py`; dashboard view
  comparing outcomes of agreeing vs disagreeing signals (check after about
  100 signals).

To raise when the trigger comes:
- Same-bar rule: the document says TP and SL in one 5-minute bar = SL, while
  the labels and backtest check 1-minute bars first (step 2, question 6).
  Replaying 5-minute blocks can't do that, so the replay would be slightly
  more cautious than the backtest unless it samples 1-minute bars.
- Apply the 0.25-pip stop slippage in the replay too (same cost rules)?
- Part B needs the walk-forward test predictions; step 2c will save them
  anyway (they also feed the calibration chart).
