"""
signals.py - THE ENGINE. A background process (APScheduler) that collects
prices, builds 5-minute bars, computes features, runs every model variant
that is not Off, applies the filters and hard risk limits (Section 8, step 4),
writes everything to the database and picks up jobs from the dashboard.

Status: placeholder - the price-recording part starts in step 5; the model
loop comes in Phase 2/3.
"""
