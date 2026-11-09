"""
Tests for broker.py that need NO network and NO real credentials: a fake
IGService stands in for trading-ig.

They check the safety rules (DEMO only, credentials never echoed), the
re-login on an expired token, that bad logins are not retried, and the
request rate limiter.
"""

import os
import types

import pytest
from trading_ig.rest import IGException, TokenInvalidException

import broker
import config

ENV_NAMES = ("IG_USERNAME", "IG_PASSWORD", "IG_API_KEY", "IG_ACC_TYPE", "IG_ACC_NUMBER")

DEMO_CREDS = {
    "username": "demo_user", "password": "secret-pw", "api_key": "KEY123456",
    "acc_type": "DEMO", "acc_number": None,
}

GOOD_LOGIN = {
    "accountType": "CFD",
    "currentAccountId": "ABC12",
    "reroutingEnvironment": "DEMO",
    "currencyIsoCode": "USD",
    "lightstreamerEndpoint": "https://demo-apd.marketdatasystems.com",
    "dealingEnabled": True,
}


# --------------------------------------------------------------------------
# A fake trading-ig IGService
# --------------------------------------------------------------------------
class FakeIG:
    instances = []

    def __init__(self, username, password, api_key, acc_type="demo", acc_number=None, **kwargs):
        self.BASE_URL = {"demo": config.IG_DEMO_BASE_URL, "live": "https://api.ig.com/gateway/deal"}[acc_type.lower()]
        self.login_response = dict(GOOD_LOGIN)
        self.login_error = None
        self.accounts_errors = []   # exceptions to raise on successive fetch_accounts calls
        self.create_calls = 0
        self.session = types.SimpleNamespace(headers={})
        FakeIG.instances.append(self)

    def create_session(self, version="2"):
        self.create_calls += 1
        if self.login_error:
            raise self.login_error
        self.session.headers.update({"CST": "cst-token", "X-SECURITY-TOKEN": "xst-token"})
        return self.login_response

    def fetch_accounts(self):
        if self.accounts_errors:
            raise self.accounts_errors.pop(0)
        return {"accounts": [{
            "accountId": "ABC12", "accountName": "Demo CFD", "accountType": "CFD",
            "status": "ENABLED", "currency": "USD",
            "balance": {"balance": 10000.0, "available": 9000.0, "deposit": 0.0, "profitLoss": 0.0},
        }]}

    def logout(self):
        pass


@pytest.fixture(autouse=True)
def reset_fake():
    FakeIG.instances = []
    yield


def make_broker(**cred_overrides):
    creds = dict(DEMO_CREDS, **cred_overrides)
    return broker.Broker(creds, ig_service_factory=FakeIG)


# --------------------------------------------------------------------------
# Credentials (.env)
# --------------------------------------------------------------------------
@pytest.fixture
def env_dir(tmp_path, monkeypatch):
    """Point config.PROJECT_ROOT at an empty temp folder and clear IG_* variables."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    saved = {n: os.environ.pop(n) for n in ENV_NAMES if n in os.environ}
    yield tmp_path
    for n in ENV_NAMES:
        os.environ.pop(n, None)
    os.environ.update(saved)


def write_env(folder, **values):
    (folder / ".env").write_text("".join(f"{k}={v}\n" for k, v in values.items()))


def test_missing_env_file(env_dir):
    with pytest.raises(broker.BrokerConfigError, match="No .env file"):
        broker.load_credentials()


def test_live_account_type_refused(env_dir):
    write_env(env_dir, IG_USERNAME="u", IG_PASSWORD="p", IG_API_KEY="k", IG_ACC_TYPE="LIVE")
    with pytest.raises(broker.BrokerConfigError, match="only runs on DEMO"):
        broker.load_credentials()


def test_missing_values_reported_by_name_only(env_dir):
    write_env(env_dir, IG_USERNAME="someone", IG_PASSWORD="", IG_API_KEY="", IG_ACC_TYPE="DEMO")
    with pytest.raises(broker.BrokerConfigError) as err:
        broker.load_credentials()
    msg = str(err.value)
    assert "IG_PASSWORD" in msg and "IG_API_KEY" in msg
    assert "someone" not in msg   # values are never echoed


def test_valid_env(env_dir):
    write_env(env_dir, IG_USERNAME="u", IG_PASSWORD="p w", IG_API_KEY="k", IG_ACC_TYPE="demo")
    creds = broker.load_credentials()
    assert creds["acc_type"] == "DEMO"
    assert creds["password"] == "p w"
    assert creds["acc_number"] is None


def test_mask():
    assert broker.mask("ABC123") == "****23"
    assert broker.mask(None) == "(none)"


# --------------------------------------------------------------------------
# Login checks
# --------------------------------------------------------------------------
def test_connect_ok():
    b = make_broker()
    info = b.connect()
    assert info["environment"] == "DEMO"
    assert info["account_type"] == "CFD"
    assert b.login_count == 1


def test_non_demo_server_refused():
    # Even if something upstream asked for LIVE, broker.py refuses the non-demo URL.
    b = make_broker(acc_type="LIVE")
    with pytest.raises(broker.BrokerConfigError, match="demo server"):
        b.connect()
    assert FakeIG.instances[0].create_calls == 0   # never even tried to log in


def test_live_environment_in_response_refused(monkeypatch):
    def factory(*a, **k):
        ig = FakeIG(*a, **k)
        ig.login_response = dict(GOOD_LOGIN, reroutingEnvironment="LIVE")
        return ig
    b = broker.Broker(dict(DEMO_CREDS), ig_service_factory=factory)
    with pytest.raises(broker.BrokerConfigError, match="not DEMO"):
        b.connect()
    assert not b.is_connected


def test_wrong_account_type_refused():
    def factory(*a, **k):
        ig = FakeIG(*a, **k)
        ig.login_response = dict(GOOD_LOGIN, accountType="SPREADBET")
        return ig
    b = broker.Broker(dict(DEMO_CREDS), ig_service_factory=factory)
    with pytest.raises(broker.BrokerConfigError, match="account type"):
        b.connect()


def test_bad_credentials_not_retried():
    def factory(*a, **k):
        ig = FakeIG(*a, **k)
        ig.login_error = IGException('HTTP error: 401 {"errorCode":"error.security.invalid-details"}')
        return ig
    b = broker.Broker(dict(DEMO_CREDS), ig_service_factory=factory)
    with pytest.raises(broker.BrokerAuthError):
        b.connect()
    assert len(FakeIG.instances) == 1 and FakeIG.instances[0].create_calls == 1


# --------------------------------------------------------------------------
# Automatic re-login
# --------------------------------------------------------------------------
@pytest.mark.parametrize("token_error", [
    TokenInvalidException(),
    Exception("error.security.account-token-invalid"),
    Exception("error.security.client-token-missing"),
])
def test_expired_token_triggers_one_relogin(token_error):
    b = make_broker()
    b.connect()
    FakeIG.instances[0].accounts_errors = [token_error]
    acc = b.get_account()
    assert acc["balance"] == 10000.0
    assert b.login_count == 2


def test_relogin_only_once():
    def factory(*a, **k):
        ig = FakeIG(*a, **k)
        ig.accounts_errors = [TokenInvalidException(), TokenInvalidException()]
        return ig
    b = broker.Broker(dict(DEMO_CREDS), ig_service_factory=factory)
    b.connect()
    with pytest.raises(broker.BrokerError):
        b.get_account()
    assert b.login_count == 2   # one original login + exactly one retry


def test_call_before_connect():
    with pytest.raises(broker.BrokerError, match="Not connected"):
        make_broker().get_account()


# --------------------------------------------------------------------------
# Rate limiter
# --------------------------------------------------------------------------
def test_rate_limiter_waits_when_full():
    now = [0.0]
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        now[0] += s

    rl = broker.RateLimiter(3, clock=lambda: now[0], sleep=fake_sleep)
    for _ in range(3):
        rl.wait()
    assert sleeps == []
    now[0] = 10.0
    rl.wait()                      # 4th call within the minute must wait until t=60
    assert sleeps == [pytest.approx(50.0)]


# --------------------------------------------------------------------------
# Streaming login details and the API-error halt rule
# --------------------------------------------------------------------------
def test_stream_login_details_follow_ig_guide():
    b = make_broker()
    b.connect()
    endpoint, user, password = b.stream_login_details()
    assert endpoint == GOOD_LOGIN["lightstreamerEndpoint"]
    assert user == "ABC12"
    assert password == "CST-cst-token|XST-xst-token"


def test_halt_after_consecutive_api_errors():
    b = make_broker()
    b.connect()
    ig = FakeIG.instances[0]
    ig.accounts_errors = [Exception("error.unexpected")] * config.HALT_AFTER_API_ERRORS
    for _ in range(config.HALT_AFTER_API_ERRORS - 1):
        with pytest.raises(broker.BrokerError):
            b.get_account()
    assert b.halt_reason() is None
    with pytest.raises(broker.BrokerError):
        b.get_account()
    assert f"{config.HALT_AFTER_API_ERRORS} IG requests failed in a row" in b.halt_reason()
    b.get_account()                     # a success resets the count
    assert b.halt_reason() is None


def test_token_expiry_is_not_counted_as_an_error():
    b = make_broker()
    b.connect()
    FakeIG.instances[0].accounts_errors = [TokenInvalidException()]
    b.get_account()
    assert b.consecutive_api_errors == 0
