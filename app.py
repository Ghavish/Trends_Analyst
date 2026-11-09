"""
app.py - the dashboard (Streamlit + Plotly). Reads from the database and
writes requests to the jobs table only; never talks to IG directly.

First version (after the connection works): data status (connected, last
price time, gaps) and live prices for the five instruments, with times in
MUT and UTC.

Status: placeholder - built in step 6.
"""
