"""A group start the fleet holds until the depth cameras are warm ("waiting_camera").

The device starts nothing locally: the fleet's start_capture arrives later like
any peer's. Until then the start counts as pending (LED blink, a press stops it),
and a stop must reach the fleet so it drops the held start.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from grabette import capture_scheduler, fleet_sync
from grabette.capture_scheduler import CaptureScheduler


class _Backend:
    is_capturing = False

    def raise_if_capture_blocked(self):
        pass

    async def prepare_capture(self):
        pass


def test_waiting_for_fleet_counts_as_pending_until_scheduled():
    sched = CaptureScheduler()
    assert not sched.is_pending()

    sched.mark_waiting_for_fleet()
    assert sched.is_pending() and not sched.is_scheduled()

    async def _schedule():
        target = datetime.now(timezone.utc) + timedelta(seconds=30)
        await sched.schedule(_Backend(), None, None, target)
        assert not sched.is_waiting_for_fleet()
        assert sched.is_scheduled()
        sched._task.cancel()

    asyncio.run(_schedule())


def test_waiting_for_fleet_expires(monkeypatch):
    sched = CaptureScheduler()
    sched.mark_waiting_for_fleet()
    now = capture_scheduler.time.monotonic()
    monkeypatch.setattr(capture_scheduler.time, "monotonic",
                        lambda: now + capture_scheduler.FLEET_WAIT_MAX_S + 1)
    assert not sched.is_pending()


def test_cancel_while_waiting_reports_cancelled_wait():
    sched = CaptureScheduler()
    sched.mark_waiting_for_fleet()

    assert asyncio.run(sched.cancel_or_wait(_Backend())) == "cancelled_wait"
    assert not sched.is_pending()
    # Nothing left to cancel the second time.
    assert asyncio.run(sched.cancel_or_wait(_Backend())) == "ran"


class _JsonResponse:
    status = 200

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, body):
        self.body = body
        self.sent = None

    def post(self, url, json=None, **kw):
        self.sent = json
        return _JsonResponse(self.body)


@pytest.fixture
def fleet_answers(monkeypatch):
    def _install(body):
        session = _Session(body)

        @asynccontextmanager
        async def _http():
            yield session
        monkeypatch.setattr(fleet_sync, "_auth_headers", lambda: {"Authorization": "Bearer t"})
        monkeypatch.setattr(fleet_sync, "_http_session", _http)
        return session
    return _install


def test_start_request_announces_it_can_wait(fleet_answers):
    session = fleet_answers({"status": "waiting_camera"})

    res = asyncio.run(fleet_sync.request_group_start(""))
    assert res == {"status": "waiting_camera"}
    assert session.sent["can_wait"] is True
