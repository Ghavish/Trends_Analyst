"""
db.py - the ONLY file that talks to SQLite (Section 10.2, design rule).

Will hold the tables shared by the engine and the dashboard: variants, jobs,
predictions/trades, the kill-switch / halt flags with an activation log, and
price-feed status (connected, last price time, gaps).

The dashboard only reads, plus writes requests into the jobs table.

Status: placeholder - built in step 4.
"""
