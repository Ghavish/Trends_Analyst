"""
app.py - the dashboard (Streamlit).

    streamlit run app.py          (opens http://localhost:8501 in your browser)

Rules (Section 10.2): the dashboard never talks to IG and never does the
engine's work. It only READS the database / Parquet files and WRITES
requests into the jobs table (e.g. the kill switch), which the engine
(signals.py) carries out within a few seconds.

Pages (Section 10.4) are added as the project grows; for now there is one,
"Data and controls". Streamlit hides the page menu while there is only one.
Plain wording and number formats come from display.py. Every time is shown in
UTC and Mauritius time (MUT, UTC+4).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import config
import db
import display
import fetch_data
import timeutils

REFRESH_SECONDS = 5          # live parts of the page refresh themselves this often
GAP_HOURS = 24               # gaps listed from the last N hours

st.set_page_config(page_title="Trends Analyst", layout="wide")


# --------------------------------------------------------------------------
# Data and controls page
# --------------------------------------------------------------------------
def data_and_controls() -> None:
    st.title("Data and controls")
    st.caption("Trends Analyst · IG demo account · times in UTC and Mauritius time (MUT) · "
               "not financial advice")

    if not Path(config.DB_PATH).exists():
        st.info("Nothing recorded yet. Start the engine with `python signals.py`.")
        return

    live_section()
    bars_section()
    safety_section()


@st.fragment(run_every=REFRESH_SECONDS)
def live_section() -> None:
    now = timeutils.now_utc()
    status = db.get_engine_status()
    running = db.engine_is_running(now=now)
    feed = db.get_feed_status()
    open_gaps = db.list_gaps(open_only=True)
    info = db.get_market_info()

    # ---- one line: is everything OK? ----
    level, message = display.overall_status(
        running, status.get("stream_status"), feed, open_gaps, status.get("market_status_error"), now)
    {"ok": st.success, "warning": st.warning, "error": st.error}[level](message)

    blocked = db.trading_block_reasons(now=now)
    if blocked:
        st.error("Trading stopped: " + " · ".join(blocked))
    else:
        st.info("Trading: allowed. No signals yet - they start in a later phase.")

    # ---- prices ----
    st.subheader("Prices")
    feed_by_inst = {f["instrument"]: f for f in feed}
    rows = []
    for key in config.INSTRUMENTS:
        f = feed_by_inst.get(key, {})
        mi = info.get(key, {})
        decimals = mi.get("decimals")
        last = f.get("last_tick_utc")
        rows.append({
            "Market": display.instrument_name(key),
            "Status": display.market_status_text(f.get("market_status")),
            "Bid": display.format_price(f.get("bid"), decimals),
            "Offer": display.format_price(f.get("offer"), decimals),
            "Spread": display.format_spread(f.get("bid"), f.get("offer"), mi.get("pip_size"),
                                            display.spread_unit(key)),
            "Last price (UTC / MUT)": display.time_text(last, now),
            "Age": display.age_text((now - last).total_seconds()) if last else "-",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Refreshes every {REFRESH_SECONDS} s. IG sends a price only when it changes, "
               "so a calm market can show an older price.")

    # ---- gaps (summary line, details on click) ----
    gaps = db.list_gaps(since_utc=now - timedelta(hours=GAP_HOURS))
    ongoing = sum(1 for g in gaps if g["end_utc"] is None)
    label = (f"Gaps in the last {GAP_HOURS} h: none" if not gaps else
             f"Gaps in the last {GAP_HOURS} h: {len(gaps)}" + (f" ({ongoing} ongoing)" if ongoing else ""))
    with st.expander(label, key="gaps_expander"):
        if gaps:
            gaps.sort(key=lambda g: (g["end_utc"] is not None, -g["start_utc"].timestamp()))
            st.dataframe(pd.DataFrame([{
                "Market": display.instrument_name(g["instrument"]),
                "What happened": display.gap_kind_text(g["kind"]),
                "From (UTC / MUT)": display.time_text(g["start_utc"], now),
                "To (UTC / MUT)": display.time_text(g["end_utc"], now) if g["end_utc"] else "ongoing",
                "Length": display.age_text(((g["end_utc"] or now) - g["start_utc"]).total_seconds()),
            } for g in gaps]), hide_index=True)
        else:
            st.write("No missing data.")

    # ---- technical details (hidden by default) ----
    with st.expander("Technical details", key="details_expander"):
        hb = status.get("heartbeat")
        st.markdown(
            f"- Engine heartbeat: {display.time_text(hb, now)}"
            f"{' (' + display.age_text((now - hb).total_seconds()) + ' ago)' if hb else ''}\n"
            f"- IG stream status: `{status.get('stream_status', '-')}` · connections opened since "
            f"start: {status.get('connections_opened', '-')} (1 = no reconnection needed)")
        if status.get("market_status_error"):
            st.markdown(f"- Market status problem: {status['market_status_error']}")
        st.dataframe(pd.DataFrame([{
            "Market": display.instrument_name(key),
            "IG epic": inst["epic"],
            "IG market code": feed_by_inst.get(key, {}).get("market_status") or "-",
            "Price updates today": feed_by_inst.get(key, {}).get("ticks_today"),
            "One pip means": info.get(key, {}).get("one_pip_means") or "-",
            "Min deal size": info.get(key, {}).get("min_deal_size"),
            "Min stop distance": info.get(key, {}).get("min_stop_distance"),
        } for key, inst in config.INSTRUMENTS.items()]), hide_index=True)


@st.fragment(run_every=60)
def bars_section() -> None:
    with st.expander("1-minute bars (latest 15)", key="bars_expander"):
        key = st.selectbox("Market", list(config.INSTRUMENTS), format_func=display.instrument_name)
        decimals = db.get_market_info().get(key, {}).get("decimals")
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
        partial = last["partial"] if "partial" in last else pd.Series(False, index=last.index)
        fmt = lambda col: [display.format_price(v, decimals) for v in last[col]]  # noqa: E731
        st.dataframe(pd.DataFrame({
            "Minute (UTC / MUT)": [display.time_text(t.to_pydatetime(), seconds=False) for t in last["time_utc"]],
            "Open": fmt("bid_open"), "High": fmt("bid_high"), "Low": fmt("bid_low"), "Close": fmt("bid_close"),
            "Price updates": last["tick_count"].astype(int).tolist(),
            "Complete?": ["no - some prices missing" if p else "yes" for p in partial],
        }), hide_index=True)
        st.caption("Bid prices. The 10:00 bar covers 10:00-10:01 and appears only after it ends. "
                   "'Some prices missing' = the connection or engine was down for part of that minute.")


def safety_section() -> None:
    st.subheader("Kill switch")
    flags = db.get_flags()
    kill, halt = flags[db.KILL_SWITCH], flags[db.HALT]
    c1, c2 = st.columns(2)
    with c1:
        st.badge("Kill switch: ON" if kill["is_on"] else "Kill switch: off",
                 color="red" if kill["is_on"] else "green")
        if kill["is_on"]:
            st.caption(f"{kill['reason']} (by {kill['source']}, {display.time_text(kill['changed_utc'])})")
    with c2:
        st.badge("Automatic halt: ON" if halt["is_on"] else "Automatic halt: off",
                 color="red" if halt["is_on"] else "green")
        if halt["is_on"]:
            st.caption(f"{halt['reason']} ({display.time_text(halt['changed_utc'])})")

    with st.form("kill_on", clear_on_submit=True):
        reason = st.text_input("Reason", value="manual stop from the dashboard")
        if st.form_submit_button("Stop trading (kill switch ON)", type="primary"):
            job = db.request_job(db.JOB_KILL_SWITCH_ON, {"reason": reason}, "dashboard")
            st.success(f"Stop request sent (request {job}). Trading is blocked from now on.")
    st.caption("The engine carries out requests within a few seconds. "
               "If the dashboard is unavailable: `python killswitch.py on`.")

    with st.expander("Resume trading or clear an automatic halt"):
        left, right = st.columns(2)
        with left.form("kill_off", clear_on_submit=True):
            reason = st.text_input("Why is it safe to resume?")
            sure = st.checkbox("I have checked and want to allow trading again")
            if st.form_submit_button("Resume trading (kill switch OFF)"):
                if not reason.strip() or not sure:
                    st.error("Give a reason and tick the box to confirm.")
                else:
                    job = db.request_job(db.JOB_KILL_SWITCH_OFF, {"reason": reason}, "dashboard")
                    st.success(f"Request sent (request {job}).")
        with right.form("clear_halt", clear_on_submit=True):
            reason = st.text_input("What caused the halt, and is it fixed?")
            sure = st.checkbox("I have checked the cause of the halt")
            if st.form_submit_button("Clear automatic halt"):
                if not reason.strip() or not sure:
                    st.error("Give a reason and tick the box to confirm.")
                else:
                    job = db.request_job(db.JOB_CLEAR_HALT, {"reason": reason}, "dashboard")
                    st.success(f"Request sent (request {job}).")

    with st.expander("History: kill switch, requests, market open/close"):
        events = db.list_flag_events(20)
        st.markdown("**Kill switch and halt changes**")
        st.dataframe(pd.DataFrame([{
            "When (UTC / MUT)": display.time_text(e["time_utc"]),
            "What": f"{'Kill switch' if e['flag'] == db.KILL_SWITCH else 'Automatic halt'} {e['action']}",
            "By": e["source"], "Reason": e["reason"],
        } for e in events]) if events else pd.DataFrame({"": ["none yet"]}), hide_index=True)
        jobs = db.list_jobs(10)
        st.markdown("**Requests from the dashboard**")
        st.dataframe(pd.DataFrame([{
            "Request": j["id"], "What": j["kind"].replace("_", " ").lower(), "Status": j["status"].lower(),
            "Sent (UTC / MUT)": display.time_text(j["requested_utc"]), "Result": j["message"] or "",
        } for j in jobs]) if jobs else pd.DataFrame({"": ["none yet"]}), hide_index=True)
        statuses = db.list_market_status_log(20)
        st.markdown("**Market open/close changes reported by IG**")
        st.dataframe(pd.DataFrame([{
            "When (UTC / MUT)": display.time_text(m["time_utc"]),
            "Market": display.instrument_name(m["instrument"]),
            "Now": display.market_status_text(m["status"]),
        } for m in statuses]) if statuses else pd.DataFrame({"": ["none yet"]}), hide_index=True)


# --------------------------------------------------------------------------
# Navigation (more pages are added in later phases, Section 10.4)
# --------------------------------------------------------------------------
page = st.navigation([st.Page(data_and_controls, title="Data and controls", default=True)],
                     position="top")
page.run()
