"""Shared fixtures for the casquette sync tests.

All tests run against the hardware-free MockBackend and a SessionManager
rooted at a per-test tmp dir, so nothing touches real hardware or the
user's ~/casquette-data.
"""

from __future__ import annotations

import pytest

from casquette.backend.mock import MockBackend
from casquette.scheduler import EpisodeScheduler
from casquette.session import SessionManager


@pytest.fixture
def sm(tmp_path):
    return SessionManager(data_dir=tmp_path)


@pytest.fixture
async def scheduler(sm):
    """A started EpisodeScheduler over a MockBackend.

    Cleans up any in-flight schedule/recording and stops the backend on
    teardown so a failing test can't leak an asyncio task into the next.
    """
    backend = MockBackend()
    await backend.start()
    s = EpisodeScheduler(backend, sm)
    yield s
    try:
        await s.stop()
    except RuntimeError:
        pass  # already IDLE
    await backend.stop()
