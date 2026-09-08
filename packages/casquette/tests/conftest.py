"""Fixtures for the casquette fleet tests.

Everything runs against the hardware-free MockBackend and a TaskManager rooted
at a per-test tmp dir, with the module-level singletons (daemon, task manager,
capture scheduler) monkeypatched so nothing leaks between tests or touches the
real ~/casquette-data.
"""

from __future__ import annotations

import pytest

import casquette.app.main as main_mod
import casquette.fleet.capture_scheduler as cs_mod
import casquette.task as task_mod
from casquette.backend.mock import MockBackend
from casquette.daemon import Daemon
from casquette.task import TaskManager


@pytest.fixture(autouse=True)
def _fresh_scheduler(monkeypatch):
    """A fresh CaptureScheduler per test (it's a process-wide singleton)."""
    monkeypatch.setattr(cs_mod, "_scheduler", cs_mod.CaptureScheduler())


@pytest.fixture
def tm(tmp_path, monkeypatch):
    m = TaskManager(data_dir=tmp_path)
    monkeypatch.setattr(task_mod, "_task_manager", m)
    return m


@pytest.fixture
async def daemon(monkeypatch):
    backend = MockBackend()
    d = Daemon(backend)
    await d.start()
    monkeypatch.setattr(main_mod, "_daemon", d)
    yield d
    await d.stop()
