"""Exercise the group-stop retry path (fleet_sync.notify_group_stop).

The first attempt blocks the button press, so it must stay short and hand off;
the retries must be safe to land late. Both properties are what kept a peer
recording after a lost stop, so they're pinned here.
"""
import asyncio

import pytest

from grabette import fleet_sync


class _FakeResponse:
    def __init__(self, status: int, body: str = "") -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeSession:
    """Replays a scripted sequence of responses/exceptions, counting POSTs."""

    def __init__(self, *outcomes) -> None:
        self._outcomes = list(outcomes)
        self.posts = 0
        self.timeouts: list[float] = []

    def post(self, url, **kw):
        self.posts += 1
        t = kw.get("timeout")
        self.timeouts.append(getattr(t, "total", None))
        out = self._outcomes.pop(0) if self._outcomes else _FakeResponse(200)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture(autouse=True)
def _wire(monkeypatch):
    """Auth + transport stubbed; retry delay collapsed so tests stay instant."""
    monkeypatch.setattr(fleet_sync, "_auth_headers", lambda: {"Authorization": "Bearer t"})
    monkeypatch.setattr(fleet_sync, "_STOP_RETRY_DELAY_S", 0.0)
    fleet_sync._retry_tasks.clear()


def _use(monkeypatch, session):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_http_session():
        yield session

    monkeypatch.setattr(fleet_sync, "_http_session", _fake_http_session)


async def _drain() -> None:
    """Let the background retry task run to completion."""
    while fleet_sync._retry_tasks:
        await asyncio.gather(*list(fleet_sync._retry_tasks), return_exceptions=True)


def test_first_attempt_succeeds_without_spawning_a_retry(monkeypatch):
    session = _FakeSession(_FakeResponse(200))
    _use(monkeypatch, session)

    asyncio.run(fleet_sync.notify_group_stop())

    assert session.posts == 1
    assert not fleet_sync._retry_tasks, "a success must not queue background work"


def test_timeout_hands_off_to_a_background_retry_that_succeeds(monkeypatch):
    # The reported failure: the first POST times out. The peers must still be told.
    session = _FakeSession(asyncio.TimeoutError(), _FakeResponse(200))
    _use(monkeypatch, session)

    async def go():
        await fleet_sync.notify_group_stop()
        assert fleet_sync._retry_tasks, "a transient failure must schedule a retry"
        await _drain()

    asyncio.run(go())
    assert session.posts == 2


def test_retry_task_is_strongly_referenced(monkeypatch):
    # asyncio only holds tasks weakly; a bare create_task can be GC'd mid-flight,
    # which is exactly how the peers end up never being told.
    session = _FakeSession(asyncio.TimeoutError(), _FakeResponse(200))
    _use(monkeypatch, session)

    async def go():
        await fleet_sync.notify_group_stop()
        import gc
        gc.collect()                       # would kill an unreferenced task
        assert fleet_sync._retry_tasks
        await _drain()
        assert not fleet_sync._retry_tasks, "the done callback must release it"

    asyncio.run(go())
    assert session.posts == 2


def test_retries_are_exhausted_then_give_up(monkeypatch):
    session = _FakeSession(*[asyncio.TimeoutError()] * 5)
    _use(monkeypatch, session)

    async def go():
        await fleet_sync.notify_group_stop()
        await _drain()

    asyncio.run(go())
    assert session.posts == 1 + fleet_sync._STOP_RETRIES


def test_4xx_is_not_retried(monkeypatch):
    # An unknown device / bad token fails identically every time — hammering the
    # Space buys nothing.
    session = _FakeSession(_FakeResponse(404, "Device not registered"))
    _use(monkeypatch, session)

    asyncio.run(fleet_sync.notify_group_stop())

    assert session.posts == 1
    assert not fleet_sync._retry_tasks


def test_5xx_is_retried(monkeypatch):
    # A waking/overloaded Space — precisely the case worth retrying.
    session = _FakeSession(_FakeResponse(503, "no healthy upstream"), _FakeResponse(200))
    _use(monkeypatch, session)

    async def go():
        await fleet_sync.notify_group_stop()
        await _drain()

    asyncio.run(go())
    assert session.posts == 2


def test_should_abort_protects_a_new_episode(monkeypatch):
    # The dangerous case: a retry landing after the operator started the NEXT
    # episode would stop that one instead.
    session = _FakeSession(asyncio.TimeoutError(), _FakeResponse(200))
    _use(monkeypatch, session)

    async def go():
        await fleet_sync.notify_group_stop(should_abort=lambda: True)
        await _drain()

    asyncio.run(go())
    assert session.posts == 1, "no retry may fire once a capture is running again"


def test_no_token_is_not_treated_as_a_failure(monkeypatch):
    monkeypatch.setattr(fleet_sync, "_auth_headers", lambda: None)
    session = _FakeSession()
    _use(monkeypatch, session)

    asyncio.run(fleet_sync.notify_group_stop())

    assert session.posts == 0
    assert not fleet_sync._retry_tasks, "a standalone device must not retry forever"


def test_timeout_is_always_passed_per_request(monkeypatch):
    # The borrowed relay session carries its own, longer session-wide timeout.
    # If these stopped being per-request, a stop would silently inherit it and
    # block the press far longer than _STOP_TIMEOUT_S.
    session = _FakeSession(asyncio.TimeoutError(), _FakeResponse(200))
    _use(monkeypatch, session)

    async def go():
        await fleet_sync.notify_group_stop()
        await _drain()

    asyncio.run(go())
    assert session.timeouts == [fleet_sync._STOP_TIMEOUT_S, fleet_sync._STOP_RETRY_TIMEOUT_S]


def test_borrowed_relay_session_is_never_closed(monkeypatch):
    # Closing it would take the relay's poll/heartbeat down with it.
    closed = []

    class _Borrowed(_FakeSession):
        async def close(self):
            closed.append(True)

    session = _Borrowed(_FakeResponse(200))
    monkeypatch.setattr(fleet_sync, "_auth_headers", lambda: {"Authorization": "Bearer t"})
    import grabette.relay_client as rc
    monkeypatch.setattr(rc, "_live_session", session)
    monkeypatch.setattr(session, "closed", False, raising=False)

    asyncio.run(fleet_sync.notify_group_stop())

    assert session.posts == 1
    assert not closed, "the relay's session must survive a group stop"
