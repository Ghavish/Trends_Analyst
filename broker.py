"""
broker.py - the ONLY file that talks to IG (Section 5, design rule).

Will provide (Phase 1):
  * login to the IG DEMO account using credentials from .env (never in code);
    refuses to run unless IG_ACC_TYPE=DEMO
  * account and market look-ups (balance, epic search, market details)
  * live price streaming (Lightstreamer, via trading-ig)
  * automatic re-login and re-subscription when the session expires or the
    stream drops

Later (Phase 3): get_positions(), place_order() with SL and TP always attached,
and a final hard-limit check before any order. No live-trading path.

Status: placeholder - built in steps 2 and 3.
"""
