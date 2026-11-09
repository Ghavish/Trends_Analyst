"""
broker.py - the ONLY file that talks to IG (Section 5, design rule).

The rest of the project calls the small functions here and never imports
trading-ig itself, so if the library breaks or falls behind an IG change, only
this file needs fixing.

What it provides so far (Phase 1):
  * load_credentials()   - reads .env; refuses anything other than DEMO
  * Broker.connect()     - logs in (REST, session v2) and checks it really is
                           the demo server and your CFD account
  * Broker.get_account() - balance, available funds, currency
  * Broker.search_markets(), Broker.get_market() - epic look-up and contract details
  * Broker.get_api_allowances() - the request limits IG reports for your API key
  * automatic re-login when IG rejects an expired/invalid session token
  * a simple limit on REST requests per minute

  * PriceStream          - live prices with automatic reconnection (step 3)
  * halt_reason()        - on Broker and PriceStream: the automatic-halt rules
Later (Phase 3): positions and orders, always with SL and TP attached, plus a
final hard-limit check. There is no live-trading path.

Facts checked on 9 Nov 2026 (IG Labs docs and trading-ig 0.0.25 source):
  * POST /session v2 returns CST and X-SECURITY-TOKEN headers, plus
    accountType, currentAccountId, reroutingEnvironment (DEMO/LIVE/...),
    lightstreamerEndpoint and accountInfo in the body.
  * IG's own pages disagree on token life (REST guide: 6 h, extended up to
    72 h while in use; FAQ: 12 h, reset by each call, all tokens invalidated
    during weekend maintenance). So we don't rely on either figure: when IG
    rejects a token we log in again and retry once.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import requests
from dotenv import load_dotenv

# trading-ig prints a notice at import time that the optional "munch" package
# is not installed. We don't use munch, so hide only that one message.
logging.getLogger("trading_ig.utils").addFilter(
    lambda record: "munch is not present" not in record.getMessage()
)
from trading_ig import IGService  # noqa: E402  (must come after the filter above)
from trading_ig.rest import ApiExceededException, TokenInvalidException  # noqa: E402
from trading_ig.streamer.ticker import TickerSubscription  # noqa: E402
# The Lightstreamer client library that trading-ig itself uses for streaming.
from lightstreamer.client import ClientListener, LightstreamerClient, SubscriptionListener  # noqa: E402

import config
import timeutils

log = logging.getLogger("broker")

# trading-ig logs the username at INFO level when it logs in. Keep its logger
# at WARNING so usernames never end up in our logs.
logging.getLogger("trading_ig").setLevel(logging.WARNING)

# Seconds before an HTTP request to IG is abandoned. trading-ig sets no
# timeout itself, which could leave the engine hanging forever on a dead
# connection.
HTTP_TIMEOUT_SECONDS = 20

REQUIRED_ENV_VARS = ("IG_USERNAME", "IG_PASSWORD", "IG_API_KEY", "IG_ACC_TYPE")

# Parts of IG error codes (from the IG REST reference) meaning "logging in
# again will not help": wrong details, disabled key, account problems...
# For these we stop instead of retrying, partly because IG locks the login
# after too many failed attempts (error.security.too-many-failed-attempts).
_FATAL_LOGIN_ERRORS = (
    "invalid-details",
    "too-many-failed-attempts",
    "api-key-",
    "kyc.required",
    "pending.agreements",
    "suspended",
    "access-denied",
    "not-yet-activated",
    "preferred.account",
    "missing.credentials",
    "invalid-application",
    "not-a-client-account",
)

# Parts of IG error codes meaning "the session token is no longer valid":
# log in again and retry.
_TOKEN_ERRORS = ("token-invalid", "token-missing")


# --------------------------------------------------------------------------
# Errors raised by this module (the rest of the project only sees these)
# --------------------------------------------------------------------------
class BrokerError(Exception):
    """Any problem talking to IG."""


class BrokerConfigError(BrokerError):
    """Something is wrong in .env or config.py (e.g. not DEMO)."""


class BrokerAuthError(BrokerError):
    """IG refused the login; retrying will not help (check .env / My IG)."""


class BrokerRateLimitError(BrokerError):
    """IG says a request allowance has been exceeded."""


class BrokerConnectionError(BrokerError):
    """Network problem or IG server error; may succeed later."""


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------
def load_credentials() -> dict:
    """
    Read the IG demo credentials from the .env file in the project folder.

    Never prints or logs the values. Raises BrokerConfigError if something is
    missing or if IG_ACC_TYPE is not DEMO.
    """
    env_path = config.PROJECT_ROOT / ".env"
    if not env_path.exists():
        raise BrokerConfigError(
            f"No .env file found at {env_path}. Copy .env.example to .env and fill it in."
        )
    load_dotenv(env_path)

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
    if missing:
        # Only the NAMES of missing settings are reported, never values.
        raise BrokerConfigError(f"Missing in .env: {', '.join(missing)}")

    acc_type = os.environ["IG_ACC_TYPE"].strip().upper()
    if acc_type != config.ALLOWED_ACC_TYPE:
        raise BrokerConfigError(
            f"IG_ACC_TYPE is '{acc_type}', but this project only runs on "
            f"{config.ALLOWED_ACC_TYPE}. There is no live-trading path."
        )

    return {
        "username": os.environ["IG_USERNAME"].strip(),
        "password": os.environ["IG_PASSWORD"],          # not stripped: spaces may be part of it
        "api_key": os.environ["IG_API_KEY"].strip(),
        "acc_type": acc_type,
        "acc_number": os.getenv("IG_ACC_NUMBER", "").strip() or None,
    }


def mask(value: str | None, keep: int = 2) -> str:
    """Show only the last few characters, e.g. 'ABC123' -> '****23'."""
    if not value:
        return "(none)"
    return "*" * max(len(value) - keep, 2) + value[-keep:]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class _TimeoutSession(requests.Session):
    """A requests Session that always applies a timeout."""

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", HTTP_TIMEOUT_SECONDS)
        return super().request(*args, **kwargs)


class RateLimiter:
    """
    Allow at most `max_per_minute` calls in any 60-second window; wait
    (sleep) when the limit is reached. Thread-safe.
    """

    def __init__(self, max_per_minute: int, clock=time.monotonic, sleep=time.sleep):
        self.max_per_minute = max_per_minute
        self._times: deque[float] = deque()
        self._lock = threading.Lock()
        self._clock = clock
        self._sleep = sleep

    def wait(self) -> None:
        with self._lock:
            while True:
                now = self._clock()
                while self._times and now - self._times[0] >= 60:
                    self._times.popleft()
                if len(self._times) < self.max_per_minute:
                    self._times.append(now)
                    return
                pause = 60 - (now - self._times[0])
                log.info("REST request limit reached; waiting %.1f s", pause)
                self._sleep(pause)


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_fatal_login_error(exc: Exception) -> bool:
    text = str(exc)
    return any(code in text for code in _FATAL_LOGIN_ERRORS)


def _is_token_error(exc: Exception) -> bool:
    if isinstance(exc, TokenInvalidException):
        return True
    text = str(exc)
    return any(code in text for code in _TOKEN_ERRORS)


def _num(value):
    """IG sends some numbers as strings (e.g. contractSize). Convert if possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


# --------------------------------------------------------------------------
# The broker
# --------------------------------------------------------------------------
class Broker:
    """
    One object per program. Usage:

        broker = Broker()
        broker.connect()
        print(broker.get_account())
        broker.disconnect()
    """

    def __init__(self, credentials: dict | None = None, ig_service_factory=IGService):
        self._creds = credentials or load_credentials()
        self._ig_service_factory = ig_service_factory   # replaceable in tests
        self._ig: IGService | None = None
        self._lock = threading.RLock()                    # one IG call at a time
        self._limiter = RateLimiter(config.IG_REST_REQUESTS_PER_MINUTE)
        self.session_info: dict = {}                      # safe details of the current login
        self.login_count = 0                              # how many times we have logged in
        self.consecutive_api_errors = 0                   # failed IG requests in a row (halt rule)
        self.last_api_error: str | None = None

    # ---------------- error counting (for the automatic halt) ----------------
    def _note_result(self, error: Exception | None) -> None:
        with self._lock:
            if error is None:
                self.consecutive_api_errors = 0
            else:
                self.consecutive_api_errors += 1
                self.last_api_error = _error_text(error)

    def halt_reason(self) -> str | None:
        """A reason to halt trading if IG requests keep failing, else None."""
        if self.consecutive_api_errors >= config.HALT_AFTER_API_ERRORS:
            return (f"{self.consecutive_api_errors} IG requests failed in a row "
                    f"(last: {self.last_api_error})")
        return None

    # ---------------- login / logout ----------------
    def connect(self) -> dict:
        """
        Log in to the IG demo account (session v2) and check what we got.

        Returns a dict with account id, type, currency, balance and the time
        of login. Raises BrokerAuthError / BrokerConfigError / BrokerConnectionError.
        """
        try:
            info = self._connect()
        except BrokerError as exc:
            self._note_result(exc)
            raise
        self._note_result(None)
        return info

    def _connect(self) -> dict:
        with self._lock:
            self._limiter.wait()
            ig = self._ig_service_factory(
                self._creds["username"],
                self._creds["password"],
                self._creds["api_key"],
                acc_type=self._creds["acc_type"],
                acc_number=self._creds["acc_number"],
                session=_TimeoutSession(),
                return_dataframe=False,   # plain dicts: no dependence on pandas formats
                return_munch=False,
                use_rate_limiter=False,   # we use our own RateLimiter (trading-ig's starts
                                          # background threads that can block program exit)
            )

            # Belt and braces: the library must be pointing at the DEMO server.
            if getattr(ig, "BASE_URL", None) != config.IG_DEMO_BASE_URL:
                raise BrokerConfigError(
                    f"Refusing to log in: server is {getattr(ig, 'BASE_URL', None)!r}, "
                    f"expected the demo server {config.IG_DEMO_BASE_URL!r}."
                )

            try:
                resp = ig.create_session(version="2")
            except requests.RequestException as exc:
                raise BrokerConnectionError(f"Could not reach IG: {_error_text(exc)}") from exc
            except ApiExceededException as exc:
                raise BrokerRateLimitError("IG request allowance exceeded during login") from exc
            except Exception as exc:
                if _is_fatal_login_error(exc):
                    raise BrokerAuthError(
                        f"IG refused the login ({exc}). Check IG_USERNAME, IG_PASSWORD and "
                        "IG_API_KEY in .env (demo API login, not your live one). Not retrying, "
                        "because IG locks the login after too many failed attempts."
                    ) from exc
                raise BrokerConnectionError(f"Login failed: {_error_text(exc)}") from exc

            self._check_login_response(resp)
            self._ig = ig
            self.login_count += 1
            self.session_info = {
                "account_id": resp.get("currentAccountId"),
                "account_type": resp.get("accountType"),
                "environment": resp.get("reroutingEnvironment"),   # may be None (see above)
                "server": ig.BASE_URL,
                "currency": resp.get("currencyIsoCode"),
                "lightstreamer_endpoint": resp.get("lightstreamerEndpoint"),
                "dealing_enabled": resp.get("dealingEnabled"),
                "logged_in_utc": timeutils.now_utc(),
            }
            log.info(
                "Logged in to IG DEMO, account %s (%s), login #%d",
                mask(self.session_info["account_id"]),
                self.session_info["account_type"],
                self.login_count,
            )
            return dict(self.session_info)

    def _check_login_response(self, resp: dict) -> None:
        """Stop unless the login is the DEMO environment and the expected account type."""
        env = resp.get("reroutingEnvironment")
        if env is not None and str(env).upper() != config.ALLOWED_ACC_TYPE:
            raise BrokerConfigError(f"IG reports environment {env!r}, not DEMO. Stopping.")
        if env is None:
            # Seen on the real demo login (9 Nov 2026): IG documents this field
            # but did not send it. The demo-URL check in connect() is the guard.
            log.info("IG login response has no reroutingEnvironment; demo server confirmed by URL.")

        acc_type = resp.get("accountType")
        if acc_type != config.EXPECTED_ACCOUNT_TYPE:
            raise BrokerConfigError(
                f"IG reports account type {acc_type!r}, but config.EXPECTED_ACCOUNT_TYPE is "
                f"{config.EXPECTED_ACCOUNT_TYPE!r}."
            )

        wanted = self._creds["acc_number"]
        current = resp.get("currentAccountId")
        if wanted and current != wanted:
            raise BrokerConfigError(
                f"Logged into account {mask(current)}, but IG_ACC_NUMBER in .env is {mask(wanted)}."
            )

    def disconnect(self) -> None:
        """Log out (best effort)."""
        with self._lock:
            if self._ig is None:
                return
            try:
                self._ig.logout()
                log.info("Logged out of IG")
            except Exception as exc:  # logging out must never crash the program
                log.warning("Logout failed (ignored): %s", _error_text(exc))
            finally:
                self._ig = None

    @property
    def is_connected(self) -> bool:
        return self._ig is not None

    def stream_login_details(self) -> tuple[str, str, str]:
        """
        (Lightstreamer endpoint, account id, password) for the price stream,
        built from the current login as IG's streaming guide describes:
        password = "CST-<CST token>|XST-<X-SECURITY-TOKEN>".
        """
        with self._lock:
            if self._ig is None:
                raise BrokerError("Not connected: call connect() first.")
            headers = self._ig.session.headers
            cst, xst = headers.get("CST"), headers.get("X-SECURITY-TOKEN")
            endpoint = self.session_info.get("lightstreamer_endpoint")
            if not (cst and xst and endpoint):
                raise BrokerError("Login did not provide streaming tokens/endpoint.")
            return endpoint, self.session_info["account_id"], f"CST-{cst}|XST-{xst}"

    # ---------------- the one place every REST call goes through ----------------
    def _call(self, method_name: str, *args, **kwargs):
        """
        Call a trading-ig method with rate limiting and error translation.
        If IG rejects the session token, log in again and retry ONCE.
        Successes and failures feed the consecutive-error count (halt rule).
        """
        try:
            result = self._call_once(method_name, *args, **kwargs)
        except BrokerError as exc:
            self._note_result(exc)
            raise
        self._note_result(None)
        return result

    def _call_once(self, method_name: str, *args, **kwargs):
        with self._lock:
            if self._ig is None:
                raise BrokerError("Not connected: call connect() first.")
            for attempt in (1, 2):
                self._limiter.wait()
                try:
                    return getattr(self._ig, method_name)(*args, **kwargs)
                except requests.RequestException as exc:
                    raise BrokerConnectionError(
                        f"{method_name}: network problem: {_error_text(exc)}"
                    ) from exc
                except ApiExceededException as exc:
                    raise BrokerRateLimitError(f"{method_name}: IG allowance exceeded") from exc
                except Exception as exc:
                    if _is_token_error(exc) and attempt == 1:
                        log.warning("%s: session token rejected by IG; logging in again", method_name)
                        self.connect()
                        continue
                    if "Server problem" in str(exc):   # trading-ig's wording for HTTP 5xx
                        raise BrokerConnectionError(f"{method_name}: IG server error: {exc}") from exc
                    raise BrokerError(f"{method_name} failed: {_error_text(exc)}") from exc

    # ---------------- account ----------------
    def get_account(self) -> dict:
        """Balance etc. of the account we are logged into."""
        data = self._call("fetch_accounts")
        current = self.session_info.get("account_id")
        for acc in data.get("accounts", []):
            if acc.get("accountId") == current:
                bal = acc.get("balance") or {}
                return {
                    "account_id": acc.get("accountId"),
                    "account_name": acc.get("accountName"),
                    "account_type": acc.get("accountType"),
                    "status": acc.get("status"),
                    "currency": acc.get("currency"),
                    "balance": bal.get("balance"),
                    "available": bal.get("available"),
                    "deposit": bal.get("deposit"),
                    "profit_loss": bal.get("profitLoss"),
                    "fetched_utc": timeutils.now_utc(),
                }
        raise BrokerError("Current account not found in IG's account list.")

    def get_api_allowances(self) -> dict | None:
        """The per-minute request allowances IG reports for this API key."""
        apps = self._call("get_client_apps")
        for app in apps or []:
            if app.get("apiKey") == self._creds["api_key"]:
                return {
                    "name": app.get("name"),
                    "status": app.get("status"),
                    "account_overall_per_min": app.get("allowanceAccountOverall"),
                    "account_trading_per_min": app.get("allowanceAccountTrading"),
                    "application_overall_per_min": app.get("allowanceApplicationOverall"),
                    "historical_data_per_min": app.get("allowanceAccountHistoricalData"),
                    "concurrent_subscriptions": app.get("concurrentSubscriptionsLimit"),
                }
        return None

    # ---------------- markets ----------------
    def search_markets(self, term: str) -> list[dict]:
        """Markets matching a search term (used once to find epics)."""
        data = self._call("search_markets", term)
        out = []
        for m in data.get("markets", []):
            out.append({
                "epic": m.get("epic"),
                "name": m.get("instrumentName"),
                "type": m.get("instrumentType"),
                "expiry": m.get("expiry"),
                "status": m.get("marketStatus"),
                "bid": m.get("bid"),
                "offer": m.get("offer"),
                "streaming": m.get("streamingPricesAvailable"),
                # IG documents this field as the time of the last price update in UTC.
                "update_time_utc_raw": m.get("updateTimeUTC"),
            })
        return out

    def get_market(self, epic: str) -> dict:
        """
        Contract details and a price snapshot for one epic (GET /markets/{epic} v3).
        Needed later for position sizing (contract size, value of one pip, minimum deal size).
        """
        received = timeutils.now_utc()
        data = self._call("fetch_market_by_epic", epic)
        inst = data.get("instrument") or {}
        rules = data.get("dealingRules") or {}
        snap = data.get("snapshot") or {}

        def rule(name):
            r = rules.get(name) or {}
            return {"value": r.get("value"), "unit": r.get("unit")} if r else None

        bid, offer = snap.get("bid"), snap.get("offer")
        return {
            "epic": inst.get("epic", epic),
            "name": inst.get("name"),
            "type": inst.get("type"),
            "expiry": inst.get("expiry"),
            "unit": inst.get("unit"),
            "contract_size": _num(inst.get("contractSize")),
            "lot_size": inst.get("lotSize"),
            "one_pip_means": inst.get("onePipMeans"),
            "value_of_one_pip": inst.get("valueOfOnePip"),
            "currencies": [
                {"code": c.get("code"), "is_default": c.get("isDefault")}
                for c in inst.get("currencies") or []
            ],
            "margin_factor": inst.get("marginFactor"),
            "margin_factor_unit": inst.get("marginFactorUnit"),
            "stops_limits_allowed": inst.get("stopsLimitsAllowed"),
            "min_deal_size": rule("minDealSize"),
            "min_stop_or_limit_distance": rule("minNormalStopOrLimitDistance"),
            "min_guaranteed_stop_distance": rule("minControlledRiskStopDistance"),
            "max_stop_or_limit_distance": rule("maxStopOrLimitDistance"),
            "market_status": snap.get("marketStatus"),
            "bid": bid,
            "offer": offer,
            # rounded to hide floating-point noise like 9.999999999998899e-05
            "spread": round(offer - bid, 6) if isinstance(bid, (int, float)) and isinstance(offer, (int, float)) else None,
            "scaling_factor": snap.get("scalingFactor"),
            "decimal_places_factor": snap.get("decimalPlacesFactor"),
            # IG's docs do not state the time zone of snapshot.updateTime, so it
            # is kept as raw text and never used for calculations. Live prices
            # (step 3) carry a proper UTC timestamp.
            "ig_update_time_raw": snap.get("updateTime"),
            "received_utc": received,
        }


# ==========================================================================
# Live prices: streaming with automatic reconnection (Phase 1, step 3)
# ==========================================================================
#
# How it works (IG Streaming API guide and Lightstreamer client docs, checked
# 9 Nov 2026):
#   * One Lightstreamer connection, logged in with the account id and
#     "CST-<token>|XST-<token>" from the REST login. IG's terms forbid more
#     than one connection, so the old one is always closed before a new one.
#   * One subscription per epic to CHART:<epic>:TICK (DISTINCT mode), using
#     trading-ig's TickerSubscription. Fields used: BID, OFR and UTM (update
#     time in milliseconds since the epoch, i.e. UTC).
#   * Network blips: the Lightstreamer library recovers by itself (statuses
#     STALLED / DISCONNECTED:WILL-RETRY / DISCONNECTED:TRYING-RECOVERY) and
#     keeps the subscriptions.
#   * If IG closes the session (status DISCONNECTED + server error, e.g. after
#     tokens expire), or the library has not recovered within
#     STREAM_RECOVERY_GRACE_SECONDS, we log in again for fresh tokens, open a
#     new connection and subscribe again, waiting longer after each failure.
#
# trading-ig's IGStreamService.create_session() is not used, because it calls
# sys.exit() if the connection fails (which would kill the engine) and logs in
# a second time without our DEMO checks.

TICK_FIELDS_USED = ("BID", "OFR", "UTM")


@dataclass(frozen=True)
class Tick:
    """One price update from IG."""
    epic: str
    time_utc: datetime        # IG's update time (UTM), UTC
    bid: float | None
    offer: float | None
    received_utc: datetime    # when our program received it (for checking delays)
    is_snapshot: bool         # True if part of the initial snapshot sent on subscribing


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _ClientEvents(ClientListener):
    """Passes connection events to the PriceStream, tagged with its connection number."""

    def __init__(self, stream: "PriceStream", generation: int):
        self._stream, self._gen = stream, generation

    def onStatusChange(self, status):
        self._stream._on_status(self._gen, status)

    def onServerError(self, code, message):
        self._stream._on_server_error(self._gen, code, message)


class _TickEvents(SubscriptionListener):
    """Passes price updates for one epic to the PriceStream."""

    def __init__(self, stream: "PriceStream", generation: int, epic: str):
        self._stream, self._gen, self._epic = stream, generation, epic

    def onItemUpdate(self, update):
        self._stream._on_update(self._gen, self._epic, update)

    def onSubscriptionError(self, code, message):
        self._stream._on_subscription_error(self._gen, self._epic, code, message)

    def onItemLostUpdates(self, itemName, itemPos, lostUpdates):
        self._stream._on_lost_updates(self._gen, self._epic, lostUpdates)


class PriceStream:
    """
    Live bid/offer prices for a list of epics, with automatic reconnection.

        stream = PriceStream(broker, ["CS.D.EURUSD.MINI.IP"], on_tick=print)
        stream.start()
        ...
        stream.stop()

    on_tick(tick) is called for every price update, on Lightstreamer's own
    thread, so it must be quick (e.g. put the tick on a queue).
    on_event(kind, detail) reports connection events for logging/display.
    """

    def __init__(self, broker: "Broker", epics, on_tick, on_event=None,
                 client_factory=None, subscription_factory=None, clock=time.monotonic):
        self._broker = broker
        self._epics = list(epics)
        self._on_tick = on_tick
        self._on_event_cb = on_event
        # Replaceable in tests:
        self._client_factory = client_factory or (lambda endpoint: LightstreamerClient(endpoint, None))
        self._subscription_factory = subscription_factory or TickerSubscription
        self._clock = clock

        self._lock = threading.RLock()
        self._client = None
        self._generation = 0              # bumped for each connection; stale callbacks are ignored
        self._status = "NOT STARTED"
        self._connected = False
        self._down_since: float | None = None      # when streaming stopped (None while streaming)
        self._drop_counted = False
        self._attempt_started: float | None = None  # our (re)connection not yet streaming
        self._need_reconnect = False
        self._reconnect_reason = ""
        self._next_attempt_at = 0.0
        self._failed_reconnects = 0                 # in a row
        self._disconnect_times: deque[float] = deque()
        self._fatal_error: str | None = None        # e.g. login refused: stop retrying
        self._stopping = False
        self._thread: threading.Thread | None = None

        self.connections_opened = 0
        self.tick_count = 0
        self.last_tick: dict[str, Tick] = {}
        self.last_error: str | None = None

    # ---------------- public ----------------
    def start(self, watchdog: bool = True) -> None:
        """
        Open the stream (the broker must already be logged in) and start the
        watchdog thread that handles reconnection. (watchdog=False is for tests.)
        """
        with self._lock:
            if self._thread is not None:
                raise BrokerError("PriceStream already started")
            self._stopping = False
            self._open()
            if watchdog:
                self._thread = threading.Thread(target=self._watchdog, name="PriceStreamWatchdog",
                                                daemon=True)
                self._thread.start()

    def stop(self) -> None:
        """Close the stream for good."""
        with self._lock:
            self._stopping = True
            self._close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._emit("stopped", "stream closed")

    @property
    def is_streaming(self) -> bool:
        return self._connected

    def health(self) -> dict:
        """A snapshot of the stream's state (for logs and, later, the dashboard)."""
        with self._lock:
            self._trim_disconnects()
            return {
                "status": self._status,
                "streaming": self._connected,
                "connections_opened": self.connections_opened,
                "failed_reconnects_in_row": self._failed_reconnects,
                "disconnects_in_window": len(self._disconnect_times),
                "tick_count": self.tick_count,
                "last_tick_utc": {e: t.time_utc for e, t in self.last_tick.items()},
                "last_error": self.last_error,
                "fatal_error": self._fatal_error,
            }

    def halt_reason(self) -> str | None:
        """A reason to halt trading under your automatic-halt rules, else None."""
        with self._lock:
            if self._fatal_error:
                return f"stream cannot log in: {self._fatal_error}"
            if self._failed_reconnects >= config.HALT_AFTER_FAILED_RECONNECTS:
                return f"{self._failed_reconnects} stream reconnection attempts failed in a row"
            self._trim_disconnects()
            if len(self._disconnect_times) >= config.HALT_AFTER_DISCONNECTS:
                return (f"{len(self._disconnect_times)} stream disconnections in the last "
                        f"{config.HALT_DISCONNECT_WINDOW_MINUTES} minutes")
            return None

    def simulate_drop(self) -> None:
        """
        TEST AID: close the current connection as if IG had dropped it, so the
        automatic reconnection can be watched. Used by test_connection.py.
        """
        with self._lock:
            if self._client is not None:
                self._emit("test", "simulating a dropped connection")
                self._client.disconnect()   # generation NOT bumped: treated like a real drop

    # ---------------- connection management ----------------
    def _open(self) -> None:
        """Create a new Lightstreamer connection and subscribe to every epic."""
        endpoint, user, password = self._broker.stream_login_details()
        self._generation += 1
        gen = self._generation
        client = self._client_factory(endpoint)
        client.connectionDetails.setUser(user)
        client.connectionDetails.setPassword(password)
        client.addListener(_ClientEvents(self, gen))
        for epic in self._epics:
            sub = self._subscription_factory(epic)
            sub.addListener(_TickEvents(self, gen, epic))
            client.subscribe(sub)
        self._client = client
        self._need_reconnect = False
        self._attempt_started = self._clock()
        self.connections_opened += 1
        self._emit("connecting", f"connection #{self.connections_opened} to the IG price stream")
        client.connect()

    def _close(self) -> None:
        """Close the current connection; its late callbacks will be ignored."""
        client, self._client = self._client, None
        self._generation += 1
        self._connected = False
        if client is not None:
            try:
                client.disconnect()
            except Exception as exc:
                log.warning("Error while closing the stream (ignored): %s", _error_text(exc))

    def _request_reconnect(self, reason: str) -> None:
        if self._need_reconnect or self._stopping:
            return
        self._need_reconnect = True
        self._reconnect_reason = reason
        if self._attempt_started is not None:
            # Our own (re)connection never reached streaming: count it as failed.
            self._failed_reconnects += 1
            self._next_attempt_at = self._clock() + self._backoff()
        else:
            self._next_attempt_at = self._clock()   # first attempt straight away
        self._emit("reconnect_needed", reason)

    def _backoff(self) -> float:
        delays = config.STREAM_RECONNECT_DELAYS_SECONDS
        return delays[min(max(self._failed_reconnects - 1, 0), len(delays) - 1)]

    def _reconnect(self) -> None:
        """Fresh login + new connection. Called by the watchdog when due."""
        self._emit("reconnecting", f"attempt after: {self._reconnect_reason}")
        self._close()
        try:
            self._broker.connect()                  # new CST / X-SECURITY-TOKEN
            self._open()
        except BrokerAuthError as exc:
            # Wrong credentials etc.: retrying could lock the IG login. Stop trying.
            self._fatal_error = str(exc)
            self.last_error = str(exc)
            self._need_reconnect = False
            self._emit("fatal", str(exc))
        except BrokerError as exc:
            self._failed_reconnects += 1
            self.last_error = str(exc)
            self._next_attempt_at = self._clock() + self._backoff()
            self._emit("reconnect_failed",
                       f"{exc} (failed {self._failed_reconnects} in a row; next try in {self._backoff():.0f} s)")

    def _watchdog(self) -> None:
        while not self._stopping:
            try:
                self._supervise_once()
            except Exception:
                log.exception("PriceStream watchdog error (continuing)")
            time.sleep(1)

    def _supervise_once(self) -> None:
        """One check: start a reconnection if one is needed and due."""
        with self._lock:
            if self._stopping or self._fatal_error:
                return
            now = self._clock()
            if not self._need_reconnect:
                if (self._attempt_started is not None
                        and now - self._attempt_started > config.STREAM_CONNECT_TIMEOUT_SECONDS):
                    self._request_reconnect(
                        f"not streaming {config.STREAM_CONNECT_TIMEOUT_SECONDS} s after connecting")
                elif (self._attempt_started is None and self._down_since is not None
                        and now - self._down_since > config.STREAM_RECOVERY_GRACE_SECONDS):
                    self._request_reconnect(
                        f"not recovered by itself within {config.STREAM_RECOVERY_GRACE_SECONDS} s "
                        f"(status {self._status})")
            if self._need_reconnect and now >= self._next_attempt_at:
                self._reconnect()

    def _trim_disconnects(self) -> None:
        cutoff = self._clock() - config.HALT_DISCONNECT_WINDOW_MINUTES * 60
        while self._disconnect_times and self._disconnect_times[0] < cutoff:
            self._disconnect_times.popleft()

    # ---------------- callbacks from Lightstreamer (its own thread) ----------------
    def _on_status(self, gen: int, status: str) -> None:
        with self._lock:
            if gen != self._generation or self._stopping:
                return
            self._status = status
            now = self._clock()
            streaming = status.startswith("CONNECTED:") and status != "CONNECTED:STREAM-SENSING"
            if streaming:
                if self._failed_reconnects or self.connections_opened > 1:
                    self._emit("reconnected", f"streaming again ({status})")
                self._connected = True
                self._down_since = None
                self._drop_counted = False
                self._attempt_started = None
                self._failed_reconnects = 0
            elif status.startswith("DISCONNECTED") or status == "STALLED":
                if self._connected:                  # just stopped streaming
                    self._down_since = now
                    self._connected = False
                # A short STALLED that recovers is not counted; a DISCONNECTED* is,
                # once per drop (for the "disconnections per hour" halt rule).
                if (status.startswith("DISCONNECTED") and self._down_since is not None
                        and not self._drop_counted):
                    self._disconnect_times.append(now)
                    self._drop_counted = True
            if status == "DISCONNECTED":
                # Final: Lightstreamer will not retry by itself.
                self._request_reconnect("connection closed")
        self._emit("status", status)

    def _on_server_error(self, gen: int, code, message) -> None:
        with self._lock:
            if gen != self._generation or self._stopping:
                return
            self.last_error = f"server error {code}: {message}"
            self._request_reconnect(self.last_error)
        self._emit("server_error", f"{code}: {message}")

    def _on_subscription_error(self, gen: int, epic: str, code, message) -> None:
        if gen != self._generation:
            return
        self.last_error = f"subscription to {epic} refused ({code}: {message})"
        self._emit("subscription_error", self.last_error)

    def _on_lost_updates(self, gen: int, epic: str, lost) -> None:
        if gen != self._generation:
            return
        self._emit("lost_updates", f"{epic}: IG dropped {lost} update(s)")

    def _on_update(self, gen: int, epic: str, update) -> None:
        if gen != self._generation:
            return
        received = timeutils.now_utc()
        try:
            time_utc = timeutils.from_epoch_ms(update.getValue("UTM"))
        except (TypeError, ValueError):
            self._emit("bad_update", f"{epic}: update without a valid UTM time, skipped")
            return
        tick = Tick(
            epic=epic,
            time_utc=time_utc,
            bid=_to_float(update.getValue("BID")),
            offer=_to_float(update.getValue("OFR")),
            received_utc=received,
            is_snapshot=bool(update.isSnapshot()),
        )
        with self._lock:
            self.tick_count += 1
            self.last_tick[epic] = tick
        try:
            self._on_tick(tick)
        except Exception:
            log.exception("on_tick handler failed (tick dropped by handler, stream continues)")

    def _emit(self, kind: str, detail: str) -> None:
        level = logging.WARNING if kind in ("server_error", "reconnect_failed", "fatal",
                                            "subscription_error", "lost_updates") else logging.INFO
        log.log(level, "stream %s: %s", kind, detail)
        if self._on_event_cb is not None:
            try:
                self._on_event_cb(kind, detail)
            except Exception:
                log.exception("on_event handler failed (ignored)")
