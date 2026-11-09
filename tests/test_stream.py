"""
Tests for broker.PriceStream with a fake Lightstreamer client, a fake broker
and a fake clock: no network, no credentials, no waiting.

They check that prices are parsed with UTC times, that Lightstreamer is left
to recover small drops by itself, that a closed session leads to a fresh login
and a new connection (only ever one open), the waits between failed attempts,
and the automatic-halt rules you chose.
"""

from datetime import datetime, timezone

import pytest

import broker
import config

UTC = timezone.utc
EPICS = ["CS.D.EURUSD.MINI.IP", "IX.D.DAX.IFMS.IP"]


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class FakeDetails:
    def setUser(self, user):
        self.user = user

    def setPassword(self, password):
        self.password = password


class FakeSub:
    def __init__(self, epic):
        self.epic = epic
        self.listeners = []

    def addListener(self, listener):
        self.listeners.append(listener)


class FakeUpdate:
    def __init__(self, values, snapshot=False):
        self._values, self._snapshot = values, snapshot

    def getValue(self, name):
        return self._values.get(name)

    def isSnapshot(self):
        return self._snapshot


class FakeClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.connectionDetails = FakeDetails()
        self.listeners, self.subs = [], []
        self.connect_called = self.disconnected = False

    def addListener(self, listener):
        self.listeners.append(listener)

    def subscribe(self, sub):
        self.subs.append(sub)

    def connect(self):
        self.connect_called = True

    def disconnect(self):
        self.disconnected = True
        self.status("DISCONNECTED")   # like the real client

    # --- helpers used by the tests ---
    def status(self, s):
        for listener in self.listeners:
            listener.onStatusChange(s)

    def server_error(self, code, message):
        self.status("DISCONNECTED")
        for listener in self.listeners:
            listener.onServerError(code, message)

    def push(self, epic, values, snapshot=False):
        for sub in self.subs:
            if sub.epic == epic:
                for listener in sub.listeners:
                    listener.onItemUpdate(FakeUpdate(values, snapshot))


class FakeBroker:
    def __init__(self):
        self.connect_calls = 0
        self.connect_errors = []   # exceptions raised by successive connect() calls

    def stream_login_details(self):
        return ("https://ls.example", "ACC1", f"CST-c{self.connect_calls}|XST-x")

    def connect(self):
        self.connect_calls += 1
        if self.connect_errors:
            raise self.connect_errors.pop(0)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def env():
    clients, ticks, events = [], [], []
    clock = Clock()
    fb = FakeBroker()

    def client_factory(endpoint):
        c = FakeClient(endpoint)
        clients.append(c)
        return c

    stream = broker.PriceStream(
        fb, EPICS, on_tick=ticks.append, on_event=lambda k, d: events.append(k),
        client_factory=client_factory, subscription_factory=FakeSub, clock=clock,
    )
    return stream, fb, clients, ticks, events, clock


def started(env):
    stream, fb, clients, *_ = env
    stream.start(watchdog=False)
    clients[-1].status("CONNECTING")
    clients[-1].status("CONNECTED:WS-STREAMING")
    return stream


def open_clients(clients):
    return [c for c in clients if not c.disconnected]


# --------------------------------------------------------------------------
# Opening the stream and reading prices
# --------------------------------------------------------------------------
def test_start_subscribes_every_epic_with_ig_login(env):
    stream, fb, clients, *_ = env
    stream = started(env)
    c = clients[0]
    assert c.connect_called
    assert [s.epic for s in c.subs] == EPICS
    assert c.connectionDetails.user == "ACC1"
    assert c.connectionDetails.password.startswith("CST-") and "|XST-" in c.connectionDetails.password
    assert stream.is_streaming


def test_tick_is_parsed_with_utc_time(env):
    stream, fb, clients, ticks, *_ = env
    started(env)
    clients[0].push(EPICS[0], {"BID": "1.11895", "OFR": "1.11904", "UTM": "1767225600000"}, snapshot=True)
    t = ticks[0]
    assert t.epic == EPICS[0]
    assert t.time_utc == datetime(2026, 1, 1, tzinfo=UTC)
    assert (t.bid, t.offer) == (1.11895, 1.11904)
    assert t.is_snapshot is True
    assert t.received_utc.tzinfo is not None
    assert stream.tick_count == 1


def test_update_without_time_is_skipped(env):
    stream, fb, clients, ticks, events, _ = env
    started(env)
    clients[0].push(EPICS[0], {"BID": "1.1", "OFR": "1.2", "UTM": None})
    assert ticks == [] and "bad_update" in events


def test_handler_error_does_not_break_stream(env):
    stream, fb, clients, ticks, *_ = env

    def bad_handler(tick):
        raise RuntimeError("boom")

    stream._on_tick = bad_handler
    started(env)
    clients[0].push(EPICS[0], {"BID": "1", "OFR": "2", "UTM": "1767225600000"})
    assert stream.tick_count == 1


# --------------------------------------------------------------------------
# Reconnection
# --------------------------------------------------------------------------
def test_short_drop_left_to_lightstreamer(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    clients[0].status("STALLED")
    clients[0].status("DISCONNECTED:TRYING-RECOVERY")
    clock.advance(20)
    stream._supervise_once()
    clients[0].status("CONNECTED:WS-STREAMING")
    stream._supervise_once()
    assert fb.connect_calls == 0 and len(clients) == 1     # no fresh login needed
    assert stream.health()["disconnects_in_window"] == 1


def test_stall_alone_is_not_counted_as_disconnection(env):
    stream, fb, clients, *_ = env
    started(env)
    clients[0].status("STALLED")
    clients[0].status("CONNECTED:WS-STREAMING")
    assert stream.health()["disconnects_in_window"] == 0


def test_session_closed_by_ig_leads_to_fresh_login_and_new_connection(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    clients[0].server_error(32, "session closed")
    stream._supervise_once()
    assert fb.connect_calls == 1                    # logged in again for fresh tokens
    assert len(clients) == 2
    assert clients[0].disconnected                  # old connection closed first
    assert len(open_clients(clients)) == 1          # never two connections
    assert [s.epic for s in clients[1].subs] == EPICS   # subscribed again
    clients[1].status("CONNECTED:WS-STREAMING")
    assert stream.is_streaming and "reconnected" in events
    # Late events from the old connection are ignored.
    clients[0].push(EPICS[0], {"BID": "1", "OFR": "2", "UTM": "1767225600000"})
    assert ticks == []


def test_not_recovered_within_grace_period(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    clients[0].status("DISCONNECTED:WILL-RETRY")
    clock.advance(config.STREAM_RECOVERY_GRACE_SECONDS - 1)
    stream._supervise_once()
    assert fb.connect_calls == 0
    clock.advance(2)
    stream._supervise_once()
    assert fb.connect_calls == 1 and len(clients) == 2


def test_connection_that_never_streams_times_out(env):
    stream, fb, clients, ticks, events, clock = env
    stream.start(watchdog=False)               # never reaches CONNECTED
    clock.advance(config.STREAM_CONNECT_TIMEOUT_SECONDS + 1)
    stream._supervise_once()                   # marks the attempt failed (waits 5 s)
    assert stream.health()["failed_reconnects_in_row"] == 1
    clock.advance(config.STREAM_RECONNECT_DELAYS_SECONDS[0])
    stream._supervise_once()
    assert fb.connect_calls == 1 and len(clients) == 2


def test_waits_grow_and_halt_after_5_failed_reconnects(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    fb.connect_errors = [broker.BrokerConnectionError("no network")] * 10
    clients[0].server_error(32, "closed")
    stream._supervise_once()                   # attempt 1 fails at once
    assert fb.connect_calls == 1
    for n, delay in enumerate(config.STREAM_RECONNECT_DELAYS_SECONDS[:4], start=2):
        clock.advance(delay - 1)
        stream._supervise_once()
        assert fb.connect_calls == n - 1       # not yet: still waiting
        clock.advance(1)
        stream._supervise_once()
        assert fb.connect_calls == n
    assert stream.health()["failed_reconnects_in_row"] == 5
    assert "5 stream reconnection attempts failed" in stream.halt_reason()


def test_success_resets_failure_count(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    fb.connect_errors = [broker.BrokerConnectionError("no network")]
    clients[0].server_error(32, "closed")
    stream._supervise_once()                   # fails
    clock.advance(config.STREAM_RECONNECT_DELAYS_SECONDS[0])
    stream._supervise_once()                   # succeeds in opening
    clients[-1].status("CONNECTED:WS-STREAMING")
    assert stream.health()["failed_reconnects_in_row"] == 0


def test_login_refused_stops_retrying(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    fb.connect_errors = [broker.BrokerAuthError("invalid-details")]
    clients[0].server_error(1, "user/password check failed")
    stream._supervise_once()
    clock.advance(600)
    stream._supervise_once()
    assert fb.connect_calls == 1               # no second try: avoids locking the login
    assert "cannot log in" in stream.halt_reason()


def test_halt_after_10_disconnections_in_an_hour(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    for _ in range(config.HALT_AFTER_DISCONNECTS - 1):
        clients[0].status("DISCONNECTED:WILL-RETRY")
        clients[0].status("CONNECTED:WS-STREAMING")
        clock.advance(60)
    assert stream.halt_reason() is None
    clients[0].status("DISCONNECTED:WILL-RETRY")
    clients[0].status("CONNECTED:WS-STREAMING")
    assert "10 stream disconnections" in stream.halt_reason()
    clock.advance(config.HALT_DISCONNECT_WINDOW_MINUTES * 60)
    assert stream.halt_reason() is None        # older than the window


def test_stop_closes_and_ignores_later_events(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    stream.stop()
    assert clients[0].disconnected
    clock.advance(1000)
    stream._supervise_once()
    assert fb.connect_calls == 0 and len(clients) == 1


def test_simulate_drop_is_handled_like_a_real_drop(env):
    stream, fb, clients, ticks, events, clock = env
    started(env)
    stream.simulate_drop()
    stream._supervise_once()
    assert fb.connect_calls == 1 and len(open_clients(clients)) == 1
