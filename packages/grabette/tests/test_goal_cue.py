"""play_sound: the fleet's audio signal, played once the take is stopped AND saved.

The fleet sends it on the stop of the take that reached the session's goal; the
device then plays its own stop and saved cues. The goal cue has to wait those
out, and the command must not hold the relay queue while it does.
"""
import asyncio

import pytest

from grabette.app import main


class Backend:
    def __init__(self, capturing=False, stopping=False, finalizing=False):
        self.is_capturing = capturing
        self.is_stopping = stopping
        self.is_finalizing = finalizing


@pytest.fixture
def played(monkeypatch):
    calls = []

    class Speaker:
        def play_goal(self):
            calls.append("goal")

    monkeypatch.setattr("grabette.hardware.sound.get_speaker", lambda: Speaker())
    monkeypatch.setattr(main, "GOAL_CUE_GAP_S", 0)
    return calls


def test_plays_at_once_when_idle(played):
    asyncio.run(main._play_goal_when_saved(Backend()))
    assert played == ["goal"]


def test_waits_for_the_save_to_finish(played):
    backend = Backend(finalizing=True)

    async def run():
        task = asyncio.create_task(main._play_goal_when_saved(backend))
        await asyncio.sleep(0.25)
        assert played == []          # still saving: not yet
        backend.is_finalizing = False
        await task

    asyncio.run(run())
    assert played == ["goal"]


def test_gives_up_rather_than_play_late(played, monkeypatch):
    monkeypatch.setattr(main, "GOAL_CUE_WAIT_S", 0.2)
    asyncio.run(main._play_goal_when_saved(Backend(capturing=True)))
    assert played == []


def test_command_is_acked_without_waiting(played, monkeypatch):
    """The relay runs commands one at a time: the ack must not wait for the save."""
    backend = Backend(stopping=True)

    class Daemon:
        pass

    d = Daemon()
    d.backend = backend
    monkeypatch.setattr(main, "get_daemon_instance", lambda: d)

    async def run():
        result = await asyncio.wait_for(
            main._dispatch_relay_command({"id": "c1", "type": "play_sound", "args": {"cue": "goal"}}),
            timeout=1)
        assert result == {"status": "ok"}
        assert played == []
        backend.is_stopping = False
        await asyncio.gather(*main._sound_tasks)

    asyncio.run(run())
    assert played == ["goal"]


def test_unknown_cue_is_an_error(monkeypatch):
    class Daemon:
        backend = Backend()

    monkeypatch.setattr(main, "get_daemon_instance", lambda: Daemon())
    result = asyncio.run(main._dispatch_relay_command({"id": "c1", "type": "play_sound",
                                                       "args": {"cue": "nope"}}))
    assert result["status"] == "error"
