"""
app.py - the dashboard (Streamlit). First version (Phase 1, step 6).

    streamlit run app.py          (opens http://localhost:8501 in your browser)

Rules (Section 10.2): the dashboard never talks to IG and never does the
engine's work. It only READS the database / Parquet files and WRITES
requests into the jobs table (e.g. the kill switch), which the engine
(signals.py) carries out within a few seconds.

Shows: engine and stream status, live prices for the five instruments, gaps,
market status changes, the latest complete 1-minute bars, and the kill
switch / halt controls. Every time is shown in UTC and Mauritius time.
Trade cards and the other pages (Section 10.4) come in later phases.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import config
import db
import fetch_data
import timeutils

REFRESH_SECONDS = 5          # live sections refresh themselves this often
GAP_HOURS = 24               # gaps shown from the last N hours

st.set_page_config(page_title="Trends Analyst", layout="wide")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def dual(ts, seconds: bool = True) -> str:
    """UTC and MUT; adds the date when it isn't today (UTC)."""
    if ts is None or pd.isna(ts):
        return "-"
    ts = pd.Timestamp(ts).to_pydatetime()
    today = timeutils.now_utc().date()
    return timeutils.fmt_dual(ts, with_date=ts.date() != today, seconds=seconds)


def age_text(ts) -> str:
    if ts is None:
        return "-"
    secs = (timeutils.now_utc() - ts).total_seconds()
    if secs < 120:
        return f"{secs:.0f} s"
    if secs < 7200:
        return f"{secs / 60:.0f} min"
    return f"{secs / 3600:.1f} h"


def duration_text(start, end) -> str:
    end = end or timeutils.now_utc()
    secs = (end - start).total_seconds()
    return f"{secs:.0f} s" if secs < 120 else (f"{secs / 60:.1f} min" if secs < 7200 else f"{secs / 3600:.1f} h")


def database_ready() -> bool:
    return Path(config.DB_PATH).exists()


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("Trends Analyst")
st.caption("IG demo account · data status (Phase 1). Times in UTC and Mauritius time (MUT, UTC+4). "
           "Not financial advice.")

if not database_ready():
    st.info("No database yet. Start the engine first:  `python signals.py`")
    st.stop()


@st.fragment(run_every=REFRESH_SECONDS)
def live_status() -> None:
    now = timeutils.now_utc()
    st.caption(f"Updated {timeutils.fmt_dual(now, with_date=True, seconds=True)} · refreshes every "
               f"{REFRESH_SECONDS} s")

    # ---- engine and stream ----
    status = db.get_engine_status()
    running = db.engine_is_running()
    heartbeat = status.get("heartbeat")
    c1, c2 = st.columns(2)
    c1.metric("Engine", "Running" if running else "Not running",
              help="Based on the engine's heartbeat (written every minute).")
    c2.metric("Stream connections", status.get("connections_opened", "-") if running else "-",
              help="1 = no reconnection needed since the engine started.")
    st.markdown(
        f"**Price stream:** {status.get('stream_status', '-') if running else '-'} · "
        f"**last heartbeat:** {dual(heartbeat) if heartbeat else '-'}"
        f"{f' ({age_text(heartbeat)} ago)' if heartbeat else ''}")
    if not running:
        st.warning("The engine is not running, so the prices below are NOT live. "
                   "Start it with `python signals.py`.")
    if status.get("market_status_error"):
        st.warning(f"Market status unavailable: {status['market_status_error']}")

    blocked = db.trading_block_reasons()
    if blocked:
        st.error("Trading BLOCKED: " + " · ".join(blocked))
    else:
        st.success("Trading allowed (no kill switch, no halt). Signals start in a later phase.")

    # ---- live prices ----
    st.subheader("Live prices")
    feed = {f["instrument"]: f for f in db.get_feed_status()}
    rows = []
    for key, inst in config.INSTRUMENTS.items():
        f = feed.get(key, {})
        bid, offer = f.get("bid"), f.get("offer")
        rows.append({
            "Instrument": inst["name"],
            "Bid": bid,
            "Offer": offer,
            "Spread": (offer - bid) if bid is not None and offer is not None else None,
            "Market": f.get("market_status") or "unknown",
            "Last price (UTC / MUT)": dual(f.get("last_tick_utc")),
            "Age": age_text(f.get("last_tick_utc")),
            "Updates today": f.get("ticks_today"),
            "Epic": inst["epic"],
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True,
        column_config={
            "Bid": st.column_config.NumberColumn(format="%.6g"),
            "Offer": st.column_config.NumberColumn(format="%.6g"),
            "Spread": st.column_config.NumberColumn(format="%.4g", help="Offer minus bid, in price units"),
        },
    )
    st.caption("Market = IG's dealing status (DEAL means open). IG only sends a price when it changes, "
               "so a quiet market can show an older last price.")

    # ---- gaps ----
    st.subheader(f"Gaps (last {GAP_HOURS} h)")
    gaps = db.list_gaps(since_utc=now - timedelta(hours=GAP_HOURS))
    if not gaps:
        st.write("No gaps recorded.")
    else:
        gaps.sort(key=lambda g: (g["end_utc"] is not None, -g["start_utc"].timestamp()))
        st.dataframe(pd.DataFrame([{
            "Instrument": config.INSTRUMENTS.get(g["instrument"], {}).get("name", g["instrument"]),
            "Kind": g["kind"],
            "Start (UTC / MUT)": dual(g["start_utc"]),
            "End (UTC / MUT)": dual(g["end_utc"]) if g["end_utc"] else "still open",
            "Duration": duration_text(g["start_utc"], g["end_utc"]),
            "Detail": g["detail"],
        } for g in gaps]), hide_index=True)
        st.caption("OUTAGE = stream not connected · QUIET = no price while IG reported the market open "
                   "(may be a lull) · ENGINE_OFF = engine not running")


@st.fragment(run_every=60)
def recent_bars() -> None:
    st.subheader("Latest complete 1-minute bars")
    key = st.selectbox("Instrument", list(config.INSTRUMENTS),
                       format_func=lambda k: config.INSTRUMENTS[k]["name"])
    today = timeutils.now_utc().date()
    try:
        bars = fetch_data.read_bars(key, today)
    except Exception as exc:          # e.g. the engine is swapping the file at this instant
        st.warning(f"Could not read the bar file just now ({exc}); it will retry.")
        return
    if bars.empty:
        st.write("No complete bars for today (UTC) yet.")
        return
    last = bars.tail(15).iloc[::-1]
    st.dataframe(pd.DataFrame({
        "Bar start (UTC / MUT)": [dual(t, seconds=False) for t in last["time_utc"]],
        "Bid open": last["bid_open"], "Bid high": last["bid_high"],
        "Bid low": last["bid_low"], "Bid close": last["bid_close"],
        "Offer close": last["offer_close"],
        "Updates": last["tick_count"],
        "Spread avg": last["spread_mean"], "Spread max": last["spread_max"],
        "Partial": ["yes" if p else "" for p in last.get("partial", [False] * len(last))],
    }), hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="%.6g")
        for c in ("Bid open", "Bid high", "Bid low", "Bid close", "Offer close", "Spread avg", "Spread max")
    })
    st.caption(f"{len(bars)} bars today (UTC). A bar labelled 10:00 covers 10:00-10:01 and only "
               "appears once it is complete (no unfinished bars). Partial = prices were not being "
               "received for part of that minute (outage, engine off or start-up).")


def safety_controls() -> None:
    st.subheader("Kill switch and halt")
    st.caption("Buttons send a request to the engine (it acts within a few seconds). A kill request "
               "blocks trading immediately, even before the engine has processed it. "
               "Command-line alternative: `python killswitch.py`.")
    flags = db.get_flags()
    for name, label in ((db.KILL_SWITCH, "Kill switch"), (db.HALT, "Automatic halt")):
        f = flags[name]
        state = "ON" if f["is_on"] else "off"
        detail = f" - by {f['source']}: {f['reason']} (since {dual(f['changed_utc'])})" if f["changed_utc"] else ""
        st.write(f"**{label}: {state}**{detail}")

    left, mid, right = st.columns(3)
    with left.form("kill_on", clear_on_submit=True):
        reason = st.text_input("Reason", value="manual stop from the dashboard")
        if st.form_submit_button("Kill switch ON", type="primary"):
            job = db.request_job(db.JOB_KILL_SWITCH_ON, {"reason": reason}, "dashboard")
            st.success(f"Kill request sent (job {job}).")
    with mid.form("kill_off", clear_on_submit=True):
        reason = st.text_input("Reason for resuming", value="")
        sure = st.checkbox("I have checked and want to allow trading again")
        if st.form_submit_button("Kill switch OFF"):
            if not reason.strip() or not sure:
                st.error("Give a reason and tick the box to confirm.")
            else:
                job = db.request_job(db.JOB_KILL_SWITCH_OFF, {"reason": reason}, "dashboard")
                st.success(f"Request sent (job {job}).")
    with right.form("clear_halt", clear_on_submit=True):
        reason = st.text_input("Reason for clearing the halt", value="")
        sure = st.checkbox("I have checked the cause of the halt")
        if st.form_submit_button("Clear automatic halt"):
            if not reason.strip() or not sure:
                st.error("Give a reason and tick the box to confirm.")
            else:
                job = db.request_job(db.JOB_CLEAR_HALT, {"reason": reason}, "dashboard")
                st.success(f"Request sent (job {job}).")

    with st.expander("Activation log and recent requests"):
        events = db.list_flag_events(20)
        st.dataframe(pd.DataFrame([{
            "Time (UTC / MUT)": dual(e["time_utc"]), "Flag": e["flag"], "Action": e["action"],
            "By": e["source"], "Reason": e["reason"],
        } for e in events]) if events else pd.DataFrame(), hide_index=True)
        jobs = db.list_jobs(10)
        st.dataframe(pd.DataFrame([{
            "Job": j["id"], "Kind": j["kind"], "Status": j["status"],
            "Requested (UTC / MUT)": dual(j["requested_utc"]), "Message": j["message"],
        } for j in jobs]) if jobs else pd.DataFrame(), hide_index=True)
        statuses = db.list_market_status_log(20)
        st.dataframe(pd.DataFrame([{
            "Time (UTC / MUT)": dual(m["time_utc"]),
            "Instrument": config.INSTRUMENTS.get(m["instrument"], {}).get("name", m["instrument"]),
            "Market status": m["status"],
        } for m in statuses]) if statuses else pd.DataFrame(), hide_index=True)


live_status()
recent_bars()
safety_controls()
