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
| `fetch_data.py` | Records the live IG feed; downloads history later |
| `features.py`, `labels.py`, `train.py`, `backtest.py` | Model pipeline (Phase 2) |
| `signals.py` | The engine (background process) |
| `app.py` | The dashboard (Streamlit) |
| `test_connection.py` | Checks the IG demo connection, shows live EUR/USD |
| `killswitch.py` | Command-line kill switch |
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

## Dashboard

Start the engine in one terminal and the dashboard in a second one:

```powershell
streamlit run app.py   # opens http://localhost:8501
```

It shows engine/stream status, live prices, gaps, the latest complete 1-minute
bars and the kill switch. It only reads data and sends requests to the engine.
