# Trends_Analyst

A personal AI day-trading assistant for forex and index CFDs on IG (demo
account). It proposes trades with entry, take profit, stop loss, calibrated
confidence and position size, shown on a laptop dashboard.

The project plan (*AI Trend Analyst: Project Plan*) is the single source of
truth. Personal use only; nothing here is financial advice.

## Method-change log

Any change to how the model is built, judged or traded after test results
have been seen is recorded in [METHOD_CHANGES.md](METHOD_CHANGES.md), with
the reason, so the number of attempts stays visible.

## Pending ideas

Three candidate ideas (opening-range features, risk checks in one place, a
reality check on trade cards) wait for their triggers; see
[ROADMAP.md](ROADMAP.md). Nothing there is built until its trigger is
reached and confirmed.

## Files

| File | Role |
|---|---|
| `config.py` | Non-secret settings: folders, time zones, instruments/epics, hard risk limits |
| `timeutils.py` | All time-zone conversions: UTC storage, market-local sessions, MUT + UTC display, trading day (17:00 New York) |
| `broker.py` | The only file that talks to IG (login, streaming, reconnection; orders later) |
| `db.py` | The only file that talks to SQLite |
| `fetch_data.py` | Records the live IG feed; downloads Dukascopy history; the final-check lock |
| `features.py` | Bars, ATR, no-lookahead alignment, the ATR report, and the model's 46 features in three groups (base, candles, 1-hour trend) |
| `labels.py` | Triple-barrier labels (buy and sell) for every entry in your window; label report |
| `train.py` | Walk-forward training and testing: LightGBM per direction, Platt calibration, variants A-D, full report |
| `backtest.py` | Cost report (no model): every entry under several holding rules, with spreads, slippage and overnight funding; full backtest later |
| `signals.py` | The engine (background process) |
| `app.py` | The dashboard (Streamlit) |
| `display.py` | Plain wording and price/spread formats used on screen (and later in alerts) |
| `test_connection.py` | Checks the IG demo connection, shows live EUR/USD |
| `killswitch.py` | Command-line kill switch |
| `compare_sources.py` | Compares Dukascopy history with your IG recordings (read only) |
| `backup.py` | Weekly backup of the IG recordings and the database to another drive |
| `tests/` | Automated tests (`python -m pytest`) |

`data/`, `models/`, `logs/`, the SQLite database and `.env` stay on your
laptop and are never committed.

## Setup on Windows 11 (PowerShell in VS Code)

Open the `Trends_Analyst` folder in VS Code, then open a terminal
(**Terminal > New Terminal**, PowerShell).

1. Create a virtual environment with Python 3.14 (the `py` launcher comes
   with the python.org installer):

   ```powershell
   py -3.14 -m venv .venv
   ```

2. Activate it (your prompt then starts with `(.venv)`):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell says running scripts is disabled, run this once, then try
   again:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

3. Install the pinned libraries:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Create your `.env` from the template and fill in your IG **demo** details
   in VS Code (never commit or share this file):

   ```powershell
   Copy-Item .env.example .env
   ```

5. Run the tests:

   ```powershell
   python -m pytest
   ```

In VS Code, choose the `.venv` interpreter (**Ctrl+Shift+P > Python: Select
Interpreter**) so new terminals activate it automatically.

## Checking the IG connection

```powershell
python test_connection.py                  # login, EUR/USD details, 60 s of live prices
python test_connection.py --simulate-drop  # also tests automatic reconnection
python test_connection.py --search         # API allowances + epic search
```

IG allows only one streaming connection per account at a time, so don't run
`test_connection.py` while the engine (`signals.py`) is running.

## Kill switch (works without the dashboard)

```powershell
python killswitch.py status                       # flags, whether trading is blocked, recent log
python killswitch.py on --reason "news spike"     # stop trading; stays on until cleared
python killswitch.py off --reason "checked"       # resume (type YES to confirm)
python killswitch.py clear-halt --reason "fixed"  # clear an automatic halt (type YES)
python killswitch.py log                          # every activation: time, reason, source
```

## Recording prices (the engine)

```powershell
python signals.py      # runs until Ctrl+C; the laptop must stay on
```

It records all five instruments to `data/ticks/<INSTRUMENT>/<YYYY-MM-DD>.parquet`
(every price update) and `data/bars_1m/...` (complete 1-minute bars), split at
UTC midnight, logs gaps and market status to the database, and writes its log
to `logs/engine.log`. Make sure Windows syncs the clock automatically
(Settings > Time & language > Date & time > Set time automatically).

## Historical data (Dukascopy)

History comes from Dukascopy through the `dukascopy-python` library (pinned to
6.0.0). It uses Dukascopy's web-chart data feeds, which are **not an official
API**: they may change or block scripted downloads. Requests are throttled to
one per second, and a run stops by itself if Dukascopy refuses them.

```powershell
python fetch_data.py pilot      # 1) every hour of 2-8 Mar 2026, all five instruments, then a report
python fetch_data.py download   # 2) everything since 2019-01-01 (after the pilot is checked)
python fetch_data.py status     # what has been downloaded, and where
```

* Stored in UTC: ticks in `data/dukascopy/ticks/<INSTRUMENT>/<YYYY-MM-DD>.parquet`,
  1-minute bars (same columns as the IG bars, labelled by start time) in
  `data/dukascopy/bars_1m/<INSTRUMENT>/<YYYY-MM>.parquet`.
* Ticks on another drive (they are ~14 GB from 2019): add a line to `.env`
  **without quotes** (in double quotes `\t` becomes a tab), create the folder
  once, then move any ticks already downloaded:

  ```powershell
  # in .env:  DUKASCOPY_TICKS_DIR=D:\TrendsAnalyst\dukascopy\ticks
  New-Item -ItemType Directory "D:\TrendsAnalyst\dukascopy\ticks"
  python fetch_data.py move-ticks
  ```

  The drive is only needed while downloading (or rebuilding bars); the bars
  stay on the laptop. If it's unplugged or has another letter, downloads stop
  with a message instead of writing elsewhere.
* Resumable: a day counts as done only when all its hours downloaded; Ctrl+C
  is safe and the same command continues. Finished days are never fetched again.
* After a hard stop (window closed, power cut, second Ctrl+C) a month's bar
  file may be out of date; `status` says so and the next `download` rebuilds
  it first. `python fetch_data.py repair-bars` does only that.
* Saturday and Sunday before 20:00 UTC are not requested (markets closed);
  the pilot downloads those hours too, to check the rule.
* The newest 3 days are left out (Dukascopy may still be completing them).
* Options: `--instruments EURUSD GBPUSD`, `--from 2026-11-01`, `--to 2026-11-06`
  (day after the last), `--feed 1` (the library's other feed, if feed 2 fails).
  Log: `logs/history.log`.

**Final-check period** (Section 7.9): 1 May 2026 to 31 Oct 2026, fixed in
`config.py`. The model pipeline reads history only through
`fetch_data.load_history_1m()`, which refuses that period (and the day before
it) unless the purpose is the final evaluation; every such read is logged.
The indices (US 500, Germany 40) are downloaded but kept out of the model
until the comparison with IG's own prices is done
(`config.HISTORY_APPROVED_FOR_MODEL`).

## Comparing Dukascopy with IG

```powershell
python compare_sources.py                       # all instruments, every day both sources have
python compare_sources.py --instruments US500 GER40 --save   # also saves to data\reports\
```

For each instrument, on minutes both sources have (IG partial bars left out):
time alignment (should match best at 0 min shift), price-level gap (IG minus
Dukascopy, overall, by hour and by day), 5-minute movement agreement and the
IG / Dukascopy ATR ratio, spreads by hour, and bars per hour in each source.
Times are shown in UTC and Mauritius time. The script decides nothing: the
numbers are for you to judge before approving the indices for the model.

Dukascopy days arrive 3 days late, and the history download goes from 2019
forwards, so the recent days need their own (short) download first, e.g.
`python fetch_data.py download --from 2026-11-09` (don't run two downloads at once).

## ATR report (choosing TP and SL distances)

```powershell
python features.py atr-report --save        # EUR/USD, 2019 to the end of the development period
```

Typical ATR(14) of 5-minute, 15-minute and 1-hour bars by hour of day (in
pips, UTC and Mauritius time), how much of a 1x ATR stop IG's spread would
cost, how often 1x ATR is below the 10x-spread floor, how far price travels
in the next 4 hours, ATR by year, and unusually quiet weekdays (holidays).
Longer bars are only used once complete (no lookahead); ATR doesn't reach
across gaps such as weekends.

## Labels (triple barrier)

```powershell
python labels.py build --save     # EUR/USD, 2019 to the end of the development period
python labels.py report           # the report again, from the saved labels
```

For every 5-minute bar completing 17:00-20:00 MUT (entry at the next open),
a buy and a sell are followed minute by minute: SL = 1 x 1-hour ATR(14)
(at least 9 pips), TP = 1.5 x SL, closed by the earlier of 4 hours and
22:00 MUT. Dukascopy mid + IG's 0.9-pip spread, 0.25-pip stop slippage;
TP and SL in the same minute counts as SL. No entries on 24 Dec - 1 Jan,
NYSE / London holidays or NYSE early-close days, in the first hour after a
gap, or across a data gap. Saved to `data\datasets\EURUSD_labels_v1.parquet`
with its settings in a `.json` file next to it. All settings are in `config.py`.

## Features

```powershell
python features.py build --save   # EUR/USD, 2019 to the end of the development period (version 2: 46 features)
python features.py describe       # summary again, from the saved file
```

One row per 5-minute bar completing in your entry window (after 14 days of
warm-up), saved to `data\datasets\EURUSD_features_v1.parquet` (settings and
feature groups in the `.json` next to it). Labels are a separate file; the
two join on `signal_bar_utc`. Every feature uses only bars complete at the
decision time: a test recomputes them on data cut off at random times and
checks they are identical to the full-data values.

## Walk-forward training and testing

```powershell
python labels.py build              # once (or after label settings change)
python features.py build            # once (or after feature settings change)
python train.py walkforward         # about 1-3 minutes
python train.py diagnose            # diagnostics on the saved test predictions (no retraining)
```

Eight rounds: train on everything before, calibrate on the next 6 months
(Platt; isotonic for information), test on the 6 months after that; 1
trading day left out between periods. One LightGBM model per direction;
variants A = base, B = + candles, C = + 1-hour trend, D = both, all in the
same rounds. A signal needs EV >= +0.10 R; trades one at a time. The report
shows test log loss per period (a group is kept only if it helps in 6 of 8),
calibration by 5-point bands, trades per period against a no-model baseline,
a 95% interval for average R that resamples whole days, other thresholds
(information only) and the most used features. Models are saved in
`models\walkforward\`, predictions and trades in `data\walkforward\`, the
report in `data\reports\`. Only the development period is used.

## Cost report (no model)

```powershell
python backtest.py costs --instrument EURUSD --save   # about 3 minutes
```

Every entry 17:00-20:00 MUT taken as a buy and as a sell, under today's rule
(1-hour ATR, same day) and with stops from the 4-hour or daily ATR held for
1, 2 or 5 weekdays (closed at 22:00 MUT, never after Friday). For each rule:
stop size, outcomes, costs in R (spread, stop slippage, overnight funding),
the break-even TP share and the actual one, and the no-model average R with
a 95% interval; then the same with night spreads doubled, and results by
entry weekday, and the cost relative to the typical move (cost per trade / spread of results, and how many trades it takes to see an edge that size). Spreads come from your IG recordings (night hours estimated
until the engine has recorded them). Funding and slippage settings are in
`config.py` ("Cost report"). US 500 and Germany 40 work the same way once
downloaded (`--instrument US500` / `GER40`); their funding is provisional.
The indices are cleared for the model for your 17:00-22:00 MUT window only;
not overnight until the night hours have been compared.

## Weekly backup (IG recordings and database)

The IG recordings can't be downloaded again. Once, add the backup folder to
`.env` (**no quotes**) and create it:

```powershell
# in .env:  BACKUP_DIR=D:\TrendsAnalyst\backups
New-Item -ItemType Directory "D:\TrendsAnalyst\backups"
```

Then weekly (the engine can keep running):

```powershell
python backup.py          # database snapshot + new/changed IG recording files
python backup.py --list   # what is in the backup folder
```

* The database is copied with SQLite's backup API (a consistent snapshot even
  while the engine writes), checked for integrity, and saved as
  `database\trends_analyst_<date_time>UTC.db`; the newest 8 are kept.
* IG recordings are mirrored to `ig_recordings\ticks` and `ig_recordings\bars_1m`.
  Nothing is ever deleted from the backup, and a finished day that changed on
  the laptop is saved next to the backup copy, never over it.
* If the drive is missing, it stops with a message instead of writing elsewhere.
* Not included: Dukascopy history (can be downloaded again) and models (can be retrained).

## Dashboard

Start the engine in one terminal and the dashboard in a second one:

```powershell
streamlit run app.py   # opens http://localhost:8501
```

It shows engine/stream status, live prices, gaps, the latest complete 1-minute
bars and the kill switch. It only reads data and sends requests to the engine.
