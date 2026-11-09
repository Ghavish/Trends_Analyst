"""
killswitch.py - command-line kill switch, for when the dashboard is down.

It sets the same KILL_SWITCH flag the dashboard button asks for, directly in
the database (through db.py), so it works even if the engine or dashboard is
stuck. The engine checks the flag every bar and before every order.

    python killswitch.py status                     # flags + recent activations
    python killswitch.py on  --reason "news spike"  # stop trading (stays on)
    python killswitch.py off --reason "checked"     # resume (asks you to confirm)
    python killswitch.py clear-halt --reason "..."  # clear an automatic HALT (asks to confirm)
    python killswitch.py log                        # full activation log

Stopping trading and closing positions are separate actions. "Close all
positions" will be added in Phase 3, when the system can place orders; until
then there are no positions opened by the system. Orders will always carry
SL and TP at IG, so positions stay protected even if the engine stops.
"""

import argparse
import sys

import db
import timeutils


def show_status() -> None:
    flags = db.get_flags()
    for name in (db.KILL_SWITCH, db.HALT):
        f = flags[name]
        state = "ON " if f["is_on"] else "off"
        when = timeutils.fmt_dual(f["changed_utc"], with_date=True) if f["changed_utc"] else "never changed"
        print(f"  {name:<12} {state}  since {when}")
        if f["reason"]:
            print(f"               last change by {f['source']}: {f['reason']}")
        if f["is_on"] and f["expires_utc"]:
            print(f"               ends by itself at {timeutils.fmt_dual(f['expires_utc'], with_date=True)}")
    reasons = db.trading_block_reasons()
    print()
    if reasons:
        print("  Trading is BLOCKED:")
        for r in reasons:
            print(f"    - {r}")
    else:
        print("  Trading is allowed (no kill switch, no halt).")


def show_log(limit: int) -> None:
    events = db.list_flag_events(limit)
    if not events:
        print("  (no activations yet)")
    for e in events:
        extra = f"  [ends {timeutils.fmt_dual(e['expires_utc'], with_date=True)}]" if e["expires_utc"] else ""
        print(f"  {timeutils.fmt_dual(e['time_utc'], with_date=True, seconds=True)}  "
              f"{e['flag']:<12} {e['action']:<3} by {e['source']:<9} {e['reason']}{extra}")


def confirm(what: str) -> bool:
    answer = input(f"  Type YES to {what}: ").strip()
    return answer == "YES"


def main() -> int:
    parser = argparse.ArgumentParser(description="Kill switch (works without the dashboard).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the flags and whether trading is blocked")
    p_on = sub.add_parser("on", help="turn the kill switch ON (stops trading until cleared)")
    p_on.add_argument("--reason", default="manual stop from the command line")
    p_off = sub.add_parser("off", help="turn the kill switch OFF (asks to confirm)")
    p_off.add_argument("--reason", required=True)
    p_off.add_argument("--yes", action="store_true", help="skip the confirmation question")
    p_halt = sub.add_parser("clear-halt", help="clear an automatic HALT (asks to confirm)")
    p_halt.add_argument("--reason", required=True)
    p_halt.add_argument("--yes", action="store_true", help="skip the confirmation question")
    p_log = sub.add_parser("log", help="show the activation log")
    p_log.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    try:
        if args.command == "status":
            show_status()
            print("\n  Recent activations:")
            show_log(10)
        elif args.command == "on":
            db.set_kill_switch(args.reason, "cli")
            print("  Kill switch is ON. Trading is stopped until you run 'killswitch.py off'.")
            show_status()
        elif args.command == "off":
            if not db.get_flags()[db.KILL_SWITCH]["is_on"]:
                print("  The kill switch is already off.")
            elif args.yes or confirm("turn the kill switch OFF and allow trading again"):
                db.clear_kill_switch(args.reason, "cli")
                print("  Kill switch is OFF.")
                show_status()
            else:
                print("  Not changed.")
        elif args.command == "clear-halt":
            if not db.get_flags()[db.HALT]["is_on"]:
                print("  There is no HALT to clear.")
            elif args.yes or confirm("clear the HALT"):
                db.clear_halt(args.reason, "cli")
                print("  HALT cleared.")
                show_status()
            else:
                print("  Not changed.")
        elif args.command == "log":
            show_log(args.limit)
        return 0
    except db.DBError as exc:
        print(f"  ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
