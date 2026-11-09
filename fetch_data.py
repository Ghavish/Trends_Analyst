"""
fetch_data.py - downloads history and records the live broker feed.

Phase 1: record IG live prices (UTC timestamp, bid, offer) and 1-minute bars
as daily Parquet files per instrument in data/, and log any gaps.
Phase 2: download Dukascopy history.

Status: placeholder - built in step 5.
"""
