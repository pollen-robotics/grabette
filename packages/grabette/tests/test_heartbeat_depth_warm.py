"""A depth_warm or progress change cuts the heartbeat wait short."""
import asyncio

from grabette import relay_client
from grabette.relay_client import RelayClient


def _client(state):
    return RelayClient("http://fleet", lambda: "t", "dev-1",
                       depth_warm_provider=lambda: state["warm"])


def _elapsed(coro):
    async def _run():
        loop = asyncio.get_running_loop()
        t = loop.time()
        await coro
        return loop.time() - t
    return asyncio.run(_run())


def test_change_sends_the_heartbeat_early(monkeypatch):
    monkeypatch.setattr(relay_client, "_HEARTBEAT_INTERVAL_S", 1.0)
    monkeypatch.setattr(relay_client, "_DEPTH_WARM_CHECK_S", 0.05)
    state = {"warm": True}
    c = _client(state)
    c._sent_depth_warm = False

    assert _elapsed(c._sleep_until_beat()) < 0.5


def test_no_change_waits_the_full_interval(monkeypatch):
    monkeypatch.setattr(relay_client, "_HEARTBEAT_INTERVAL_S", 0.3)
    monkeypatch.setattr(relay_client, "_DEPTH_WARM_CHECK_S", 0.05)
    state = {"warm": True}
    c = _client(state)
    c._sent_depth_warm = True

    assert _elapsed(c._sleep_until_beat()) >= 0.3


def test_a_progress_change_sends_the_heartbeat_early(monkeypatch):
    # The fleet's progress bar would otherwise miss the short steps.
    monkeypatch.setattr(relay_client, "_HEARTBEAT_INTERVAL_S", 1.0)
    monkeypatch.setattr(relay_client, "_DEPTH_WARM_CHECK_S", 0.05)
    c = RelayClient("http://fleet", lambda: "t", "dev-1",
                    progress_provider=lambda: "cmd1:slam:1/3")
    c._sent_progress = "cmd1:check:3/3"

    assert _elapsed(c._sleep_until_beat()) < 0.5
