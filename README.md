# Trends_Analyst

A personal AI day-trading assistant for forex and index CFDs on IG (demo
account). It proposes trades with entry, take profit, stop loss, calibrated
confidence and position size, shown on a laptop dashboard.

The project plan (*AI Trend Analyst: Project Plan*) is the single source of
truth. Personal use only; nothing here is financial advice.

## Files

| File | Role |
|---|---|
| `config.py` | Non-secret settings: folders, time zones, instruments/epics, hard risk limits |
| `timeutils.py` | All time-zone conversions: UTC storage, market-local sessions, MUT + UTC display, trading day (17:00 New York) |
| `broker.py` | The only file that talks to IG (login, streaming, reconnection; orders later) |
| `db.py` | The only file that talks to SQLite |
| `fetch_data.py` | Records the live IG feed; downloads Dukascopy history; the final-check lock |
| `features.py`, `labels.py`, `train.py`, `backtest.py` | Model pipeline (Phase 2) |
| `signals.py` | The engine (background process) |
| `app.py` | The dashboard (Streamlit) |
| `display.py` | Plain wording and price/spread formats used on screen (and later in alerts) |
| `test_connection.py` | Checks the IG demo connection, shows live EUR/USD |
| `killswitch.py` | Command-line kill switch |
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
