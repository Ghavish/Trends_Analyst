"""
broker.py - the ONLY file that talks to IG (Section 5, design rule).

The rest of the project calls the small functions here and never imports
trading-ig itself, so if the library breaks or falls behind an IG change, only
this file needs fixing.

What this step (Phase 1, step 2) provides:
  * load_credentials()   - reads .env; refuses anything other than DEMO
  * Broker.connect()     - logs in (REST, session v2) and checks it really is
                           the demo server and your CFD account
  * Broker.get_account() - balance, available funds, currency
  * Broker.search_markets(), Broker.get_market() - epic look-up and contract details
  * Broker.get_api_allowances() - the request limits IG reports for your API key
  * automatic re-login when IG rejects an expired/invalid session token
  * a simple limit on REST requests per minute

Coming next (step 3): live price streaming with automatic reconnection.
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

import requests
from dotenv import load_dotenv

# trading-ig prints a notice at import time that the optional "munch" package
# is not installed. We don't use munch, so hide only that one message.
logging.getLogger("trading_ig.utils").addFilter(
    lambda record: "munch is not present" not in record.getMessage()
)
from trading_ig import IGService  # noqa: E402  (must come after the filter above)
from trading_ig.rest import ApiExceededException, TokenInvalidException  # noqa: E402

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

    # ---------------- login / logout ----------------
    def connect(self) -> dict:
        """
        Log in to the IG demo account (session v2) and check what we got.

        Returns a dict with account id, type, currency, balance and the time
        of login. Raises BrokerAuthError / BrokerConfigError / BrokerConnectionError.
        """
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

    # ---------------- the one place every REST call goes through ----------------
    def _call(self, method_name: str, *args, **kwargs):
        """
        Call a trading-ig method with rate limiting and error translation.
        If IG rejects the session token, log in again and retry ONCE.
        """
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
