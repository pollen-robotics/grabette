"""Tests for the processing-Space wake path (_wake_space).

The Hub-runtime shortcuts only fire on states that are painful to reproduce by
hand — a Space that is broken, or paused rather than asleep — so they are pinned
here. Each case is built to return before the first retry sleep, keeping the
suite instant.
"""
import asyncio

import pytest

from grabette.app import main


class _FakeResponse:
    def __init__(self, status: int, ctype: str) -> None:
        self.status = status
        self.headers = {"Content-Type": ctype}

    async def read(self) -> bytes:
        return b""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeSession:
    """Counts probes and replays a scripted response sequence."""

    def __init__(self, *responses: _FakeResponse) -> None:
        self._responses = list(responses)
        self.probes = 0

    def get(self, url, **kw):
        self.probes += 1
        if self._responses:
            return self._responses.pop(0)
        return _FakeResponse(503, "text/html")


_AWAKE = lambda: _FakeResponse(200, "application/json")  # noqa: E731


def _wake(session, **kw):
    return asyncio.run(main._wake_space(session, "https://space.test", {}, **kw))


def test_broken_space_fails_immediately_without_probing(monkeypatch):
    # A BUILD_ERROR Space never wakes; waiting the whole budget is pure delay.
    monkeypatch.setattr(main, "_space_stage", lambda repo, token: "BUILD_ERROR")
    session = _FakeSession()

    reason = _wake(session, space_repo="org/space")

    assert reason is not None
    assert "failed state" in reason and "BUILD_ERROR" in reason
    assert session.probes == 0, "must not probe a Space known to be broken"


def test_paused_space_is_restarted_then_probed(monkeypatch):
    # PAUSED does not wake on traffic — it takes an explicit restart.
    calls = []
    monkeypatch.setattr(main, "_space_stage", lambda repo, token: "PAUSED")
    monkeypatch.setattr(
        main, "_restart_space", lambda repo, token: calls.append(repo) or True,
    )
    session = _FakeSession(_AWAKE())

    assert _wake(session, space_repo="org/space") is None
    assert calls == ["org/space"], "restart must be requested once"
    assert session.probes == 1


def test_paused_space_with_readonly_token_reports_the_manual_step(monkeypatch):
    # A read-only token can't restart: say so instead of timing out silently.
    monkeypatch.setattr(main, "_space_stage", lambda repo, token: "PAUSED")
    monkeypatch.setattr(main, "_restart_space", lambda repo, token: False)
    session = _FakeSession()

    reason = _wake(session, space_repo="org/space")

    assert reason is not None
    assert "PAUSED" in reason and "cannot restart" in reason
    assert session.probes == 0


def test_hub_is_never_consulted_without_a_repo(monkeypatch):
    # The enrichment is opt-in: no space_repo → the old behaviour, untouched.
    def _boom(*a, **kw):
        raise AssertionError("_space_stage must not be called without space_repo")

    monkeypatch.setattr(main, "_space_stage", _boom)
    session = _FakeSession(_AWAKE())

    assert _wake(session) is None
    assert session.probes == 1


def test_unreadable_runtime_falls_back_to_probing(monkeypatch):
    # A token that can't read the runtime yields None — the probe still decides.
    monkeypatch.setattr(main, "_space_stage", lambda repo, token: None)
    session = _FakeSession(_AWAKE())

    assert _wake(session, space_repo="org/space") is None
    assert session.probes == 1


def test_cancellation_aborts_the_wait(monkeypatch):
    monkeypatch.setattr(main, "_space_stage", lambda repo, token: "SLEEPING")
    session = _FakeSession()

    with pytest.raises(main._CommandCancelled):
        _wake(session, is_cancelled=lambda: True, space_repo="org/space")
    assert session.probes == 0
