"""
test_connection.py - checks the IG demo connection. Display only: nothing
is saved, and no orders are placed.

Usage (inside the activated .venv, from the project folder):

    python test_connection.py
        1. reads .env and confirms it is set to DEMO
        2. logs in and shows your demo account (balance, currency)
        3. shows the request allowances IG reports for your API key
        4. searches IG for the five instruments so you can choose their epics
        5. shows contract details for any epics already filled in config.py

    python test_connection.py --market EPIC
        shows contract details and a price snapshot for one epic

    python test_connection.py --market EPIC1 EPIC2 ...
        compares several epics side by side (spread, contract size, minimum deal)

Step 3 will add live streaming EUR/USD prices.
"""

import argparse
import logging
import sys

import broker
import config
import timeutils

MAX_SEARCH_ROWS = 12   # rows shown per instrument search


def line(char="-"):
    print(char * 78)


def show_account(b: broker.Broker) -> None:
    info = b.session_info
    acc = b.get_account()
    env = info["environment"] or "not reported by IG"
    print(f"  Environment     : {env}  (server: {info['server']})")
    print(f"  Account         : {broker.mask(acc['account_id'])}  ({acc['account_type']}, {acc['status']})")
    print(f"  Currency        : {acc['currency']}")
    print(f"  Balance         : {acc['balance']}")
    print(f"  Available       : {acc['available']}")
    print(f"  Open P/L        : {acc['profit_loss']}")
    print(f"  Dealing enabled : {info['dealing_enabled']}")


def show_allowances(b: broker.Broker) -> None:
    allow = b.get_api_allowances()
    if allow is None:
        print("  (IG did not list this API key among your applications)")
        return
    print(f"  Application name               : {allow['name']} ({allow['status']})")
    print(f"  Account requests / minute      : {allow['account_overall_per_min']}")
    print(f"  Account trading requests / min : {allow['account_trading_per_min']}")
    print(f"  Application requests / minute  : {allow['application_overall_per_min']}")
    print(f"  Historical data points         : {allow['historical_data_per_min']}"
          "  (IG reference says per minute, IG FAQ says per week; we treat it as per week)")
    print(f"  Streaming subscriptions        : {allow['concurrent_subscriptions']}")
    print(f"  (our own cap in config.py      : {config.IG_REST_REQUESTS_PER_MINUTE} requests / minute)")


def show_search(b: broker.Broker) -> None:
    for key, inst in config.INSTRUMENTS.items():
        results = b.search_markets(inst["search"])
        print(f"\n  {inst['name']}  (search '{inst['search']}', {len(results)} results"
              f"{', first ' + str(MAX_SEARCH_ROWS) + ' shown' if len(results) > MAX_SEARCH_ROWS else ''})"
              f"   config epic: {inst['epic'] or 'not set'}")
        print(f"    {'EPIC':<28} {'NAME':<30} {'TYPE':<11} {'EXPIRY':<7} {'STATUS':<9}")
        for r in results[:MAX_SEARCH_ROWS]:
            print(f"    {str(r['epic']):<28} {str(r['name'])[:30]:<30} {str(r['type'])[:11]:<11} "
                  f"{str(r['expiry'])[:7]:<7} {str(r['status'])[:9]:<9}")


def show_market(b: broker.Broker, epic: str) -> None:
    m = b.get_market(epic)

    def rule(r):
        return f"{r['value']} {r['unit']}" if r else "-"

    print(f"\n  {m['epic']}  -  {m['name']}")
    print(f"    Type / expiry          : {m['type']} / {m['expiry']}")
    print(f"    Status                 : {m['market_status']}")
    print(f"    Bid / offer / spread   : {m['bid']} / {m['offer']} / {m['spread']}")
    print(f"    Snapshot received      : {timeutils.fmt_dual(m['received_utc'], with_date=True, seconds=True)}")
    print(f"    IG 'updateTime' (raw)  : {m['ig_update_time_raw']}   (time zone not stated in IG docs)")
    print(f"    Contract size / unit   : {m['contract_size']} / {m['unit']}")
    print(f"    Lot size               : {m['lot_size']}")
    print(f"    One pip means          : {m['one_pip_means']}")
    print(f"    Value of one pip       : {m['value_of_one_pip']}")
    print(f"    Currencies             : {', '.join(c['code'] + (' (default)' if c['is_default'] else '') for c in m['currencies'])}")
    print(f"    Margin factor          : {m['margin_factor']} {m['margin_factor_unit']}")
    print(f"    Min deal size          : {rule(m['min_deal_size'])}")
    print(f"    Min stop/limit distance: {rule(m['min_stop_or_limit_distance'])}")
    print(f"    Min guaranteed stop    : {rule(m['min_guaranteed_stop_distance'])}")
    print(f"    Stops/limits allowed   : {m['stops_limits_allowed']}")
    print(f"    Scaling factor         : {m['scaling_factor']}   decimal places factor: {m['decimal_places_factor']}")


def show_comparison(b: broker.Broker, epics: list[str]) -> None:
    """Two lines per epic, to compare candidate markets side by side."""
    def rule(r):
        return f"{r['value']} {r['unit']}" if r else "-"

    received = None
    print(f"  {'EPIC':<22} {'NAME':<28} {'STATUS':<10} {'BID':>11} {'OFFER':>11} {'SPREAD':>9}")
    print(f"  {'':<22} {'contract size / unit':<28} {'pip value':<18} {'min deal':<12} {'min stop':<12} {'ccy'}")
    print("  " + "-" * 92)
    for epic in epics:
        try:
            m = b.get_market(epic)
        except broker.BrokerError as exc:
            print(f"  {epic:<22} ERROR: {exc}")
            continue
        received = m["received_utc"]
        ccy = ",".join(c["code"] for c in m["currencies"] if c["is_default"]) or \
            ",".join(c["code"] for c in m["currencies"])
        print(f"  {m['epic']:<22} {str(m['name'])[:28]:<28} {str(m['market_status'])[:10]:<10} "
              f"{str(m['bid']):>11} {str(m['offer']):>11} {str(m['spread']):>9}")
        print(f"  {'':<22} {(str(m['contract_size']) + ' / ' + str(m['unit']))[:28]:<28} "
              f"{str(m['value_of_one_pip'])[:18]:<18} {rule(m['min_deal_size'])[:12]:<12} "
              f"{rule(m['min_stop_or_limit_distance'])[:12]:<12} {ccy}")
    if received:
        print(f"  (snapshots taken around {timeutils.fmt_dual(received, with_date=True)})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the IG demo connection.")
    parser.add_argument("--market", nargs="+", metavar="EPIC", help="show details for these epics only")
    parser.add_argument("--debug", action="store_true", help="show detailed log messages")
    args = parser.parse_args()

    timeutils.configure_logging(logging.DEBUG if args.debug else logging.WARNING)

    line("=")
    print("IG DEMO connection test   " + timeutils.fmt_dual(timeutils.now_utc(), with_date=True, seconds=True))
    line("=")

    b = None
    try:
        print("[1] Reading .env ...")
        creds = broker.load_credentials()
        print(f"  OK: IG_ACC_TYPE={creds['acc_type']}, API key {broker.mask(creds['api_key'], 4)}")

        print("[2] Logging in ...")
        b = broker.Broker(creds)
        b.connect()
        print("  OK: logged in")
        show_account(b)

        if args.market and len(args.market) > 1:
            print("[3] Market comparison ...")
            show_comparison(b, args.market)
        elif args.market:
            print("[3] Market details ...")
            show_market(b, args.market[0])
        else:
            print("[3] API allowances for this key ...")
            show_allowances(b)

            print("[4] Searching for the five instruments ...")
            show_search(b)

            configured = [i["epic"] for i in config.INSTRUMENTS.values() if i["epic"]]
            print("\n[5] Details for epics set in config.py ...")
            if not configured:
                print("  None set yet. Pick the epics from the lists above, check them with")
                print("  'python test_connection.py --market EPIC', then fill them in config.py.")
            for epic in configured:
                show_market(b, epic)

        line()
        print("Connection test finished OK.")
        return 0

    except broker.BrokerError as exc:
        line()
        print(f"FAILED ({type(exc).__name__}): {exc}")
        return 1
    finally:
        if b is not None:
            b.disconnect()


if __name__ == "__main__":
    sys.exit(main())
