"""A busy grabette must refuse to record, and must look busy.

Starting a recording on top of dataset work does not fail loudly — it produces
an episode. A worse one: hf_xet chunks and hashes in native threads, so an
upload competes for CPU with the H.264 encoders and the OAK-D drainers, and the
take comes back with dropped frames and jittered timestamps. An episode that
looks recorded and is quietly degraded is the most expensive kind.

The fleet already refused to dispatch a start to a device it knew was uploading.
Two holes made that insufficient, and both are closed here:

  * the PHYSICAL BUTTON and the local dashboard never consulted the fleet, so
    they walked straight past that gate;
  * the device could not see fleet-dispatched work at all — a relay command
    creates no local job, so the heartbeat reported "idle" through a 20-minute
    upload, and the fleet's own view was only as good as its command queue.
"""
import asyncio
import sys
import threading
import types

import pytest

from grabette.app import main
from grabette.backend.base import Backend


@pytest.fixture(autouse=True)
def _clean_activity():
    main._active_dataset_work.clear()
    main._stalled_uploads = 0
    yield
    main._active_dataset_work.clear()
    main._stalled_uploads = 0


# --- one answer to "what is this device doing" -------------------------------

def test_idle_device_is_free():
    assert main._device_activity() == "idle"
    assert main._busy_reason() == ""


def test_relay_dataset_work_is_visible_to_the_device_itself():
    # The blind spot: relay commands create no local job, so this used to read
    # "idle" for the whole upload — to the fleet AND to the capture gate.
    main._active_dataset_work["cmd1"] = "uploading"
    assert main._device_activity() == "uploading"

    main._active_dataset_work["cmd1"] = "processing"
    assert main._device_activity() == "processing"


def test_processing_wins_over_uploading():
    main._active_dataset_work.update({"a": "uploading", "b": "processing"})

    assert main._device_activity() == "processing"


def test_an_abandoned_upload_thread_still_counts_as_busy():
    # We stopped waiting on it; it did not stop running. On a Pi it is hashing,
    # not sleeping on a socket — the device is not free.
    main._stalled_uploads = 1

    assert main._device_activity() == "uploading"
    assert "upload" in main._busy_reason()


def test_the_upload_command_marks_the_device_busy_while_it_runs():
    # End-to-end through the relay entry point: what the handler does is
    # irrelevant, what matters is that the device reads as busy for its duration
    # and free again straight after — including on an early error return.
    seen = []

    async def _fake_dispatch(cmd):
        seen.append(main._device_activity())
        return {"status": "error", "message": "whatever"}

    orig = main._dispatch_relay_command
    main._dispatch_relay_command = _fake_dispatch
    try:
        asyncio.run(main._handle_relay_command(
            {"id": "c1", "type": "upload_episodes", "args": {}}))
    finally:
        main._dispatch_relay_command = orig

    assert seen == ["uploading"]
    assert main._device_activity() == "idle", "the finally must always release it"


def test_an_unrelated_command_does_not_mark_the_device_busy():
    seen = []

    async def _fake_dispatch(cmd):
        seen.append(main._device_activity())
        return {"status": "ok"}

    orig = main._dispatch_relay_command
    main._dispatch_relay_command = _fake_dispatch
    try:
        asyncio.run(main._handle_relay_command({"id": "c1", "type": "get_state"}))
    finally:
        main._dispatch_relay_command = orig

    assert seen == ["idle"]


def test_capturing_is_not_a_blocker():
    # start_capture has its own, clearer "Already capturing" error; reporting a
    # recording as "busy" here would mask it.
    assert main._RECORDING_BLOCKERS.get("capturing") is None


# --- the gate itself ----------------------------------------------------------

class _Backend(Backend):
    """Minimal concrete Backend — only the gate is under test."""

    async def start(self): ...
    async def stop(self): ...
    def get_state(self): ...
    async def start_capture(self, episode_dir): self.raise_if_capture_blocked()
    async def stop_capture(self): ...
    def get_capture_status(self): ...
    @property
    def is_capturing(self): return False
    def get_frame_jpeg(self): return None


def test_no_probe_means_free():
    # A backend nobody installed a probe on must record normally.
    assert _Backend().busy_reason == ""
    _Backend().raise_if_capture_blocked()


def test_the_gate_refuses_while_busy():
    b = _Backend()
    b.set_busy_probe(lambda: "uploading episodes")

    with pytest.raises(RuntimeError, match="uploading episodes"):
        b.raise_if_capture_blocked()


def test_start_capture_refuses_while_busy():
    b = _Backend()
    b.set_busy_probe(lambda: "uploading episodes")

    with pytest.raises(RuntimeError, match="uploading episodes"):
        asyncio.run(b.start_capture("/tmp/ep"))


def test_a_broken_probe_never_blocks_recording():
    # The probe is a diagnostic. If it throws, the failure mode must be "record
    # anyway", not "this grabette can no longer record".
    b = _Backend()
    b.set_busy_probe(lambda: 1 / 0)

    assert b.busy_reason == ""
    b.raise_if_capture_blocked()


def test_a_hardware_fault_outranks_a_busy_state():
    # Both refuse, but the operator needs the actionable one: busy clears by
    # itself, a fault does not.
    class _Faulty(_Backend):
        hardware_error = "no OAK-D calibration"

    b = _Faulty()
    b.set_busy_probe(lambda: "uploading episodes")

    with pytest.raises(RuntimeError, match="calibration"):
        b.raise_if_capture_blocked()


def test_the_scheduler_refuses_up_front():
    # A group start must fail at the press, not silently in a background task
    # seconds later at T0.
    from grabette.capture_scheduler import CaptureScheduler

    b = _Backend()
    b.set_busy_probe(lambda: "uploading episodes")

    async def _go():
        await CaptureScheduler().schedule(b, None, "/tmp/ep", None)

    with pytest.raises(RuntimeError, match="uploading episodes"):
        asyncio.run(_go())


def test_a_free_device_schedules_normally():
    from grabette.capture_scheduler import CaptureScheduler

    async def _go():
        s = CaptureScheduler()
        await s.schedule(_Backend(), None, "/tmp/ep", None)
        assert s.is_scheduled()
        s._task.cancel()

    asyncio.run(_go())


# --- and it must LOOK busy ----------------------------------------------------

if "gpiod" not in sys.modules:  # pragma: no cover - import shim, see test_hardware_error
    _line = types.SimpleNamespace(
        Bias=types.SimpleNamespace(PULL_UP="pull_up"),
        Direction=types.SimpleNamespace(OUTPUT="out", INPUT="in"),
        Value=types.SimpleNamespace(ACTIVE="active", INACTIVE="inactive"),
    )
    gpiod = types.ModuleType("gpiod")
    gpiod.line = _line
    gpiod.LineSettings = lambda **kw: kw
    gpiod.request_lines = lambda *a, **kw: None
    sys.modules["gpiod"] = gpiod
    sys.modules["gpiod.line"] = _line


class _LedBackend:
    is_teleop_active = False
    is_teleop_sending = False
    is_stopping = False
    is_starting = False
    is_capturing = False
    hardware_error = ""
    busy_reason = ""


def _desired(**state):
    from grabette.button_listener import ButtonListener

    b = _LedBackend()
    for k, v in state.items():
        setattr(b, k, v)
    return ButtonListener(b, None)._desired_led()


def test_a_busy_device_does_not_look_idle():
    # Dark is the signal for "available" — an operator picks it up and presses.
    assert _desired() == "off"
    assert _desired(busy_reason="uploading episodes") == "busy"


def test_a_running_capture_outranks_busy():
    # If a recording IS somehow live, showing it wins.
    assert _desired(busy_reason="uploading", is_capturing=True) == "on"
    assert _desired(busy_reason="uploading", is_stopping=True) == "blink_fast"


def test_a_fault_outranks_busy_on_the_led_too():
    assert _desired(busy_reason="uploading", hardware_error="boom") == "error"


def test_busy_and_error_patterns_are_distinguishable():
    from grabette import button_listener as bl

    # Same burst family, different count — "count the blips" only works if the
    # counts differ and the gap is long enough to separate the bursts.
    assert bl._BUSY_PULSES != bl._ERROR_PULSES
    assert bl._BUSY_GAP_S >= bl._ERROR_GAP_S
    assert bl._ERROR_GAP_S > bl._ERROR_PULSE_OFF_S * 4
