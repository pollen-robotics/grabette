"""A grabette that cannot record convertible episodes must say so, loudly.

The failure being closed here is silent by construction: the OAK-D answers, the
pipeline runs, the dashboard looks healthy — but oakd_calib_offline.json is
never written, so every episode of the session is rejected days later by the
SLAM Space. Nothing on the device said a word.

Three things now have to hold:
  * an unreadable/degenerate calibration is an error, not a warning;
  * an episode is never written without it;
  * the fault is visible on the device itself — the LED pattern, since a
    grabette in the field has no screen.
"""
import sys
import threading
import time
import types

import pytest

# hardware.button talks to gpiod at import time, and the test host has no GPIO.
# Stub it so the LED PATTERN (pure timing logic over a line handle) stays
# testable off-hardware — it is the operator's only feedback channel, so it is
# exactly the part that must not be left unpinned.
if "gpiod" not in sys.modules:  # pragma: no cover - import shim
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

from grabette.hardware.button import LedButton  # noqa: E402
from grabette.hardware.oakd import OakdCalibrationError, OakdCapture  # noqa: E402


# --- the calibration itself ---------------------------------------------------

class _Calib:
    def __init__(self, fx=600.0, fy=600.0, baseline_cm=7.5, raises=False):
        self._fx, self._fy, self._baseline_cm, self._raises = fx, fy, baseline_cm, raises

    def getCameraIntrinsics(self, socket, w, h):  # noqa: N802 - depthai API
        if self._raises:
            raise RuntimeError("EEPROM read failed")
        return [[self._fx, 0.0, w / 2], [0.0, self._fy, h / 2], [0.0, 0.0, 1.0]]

    def getImuToCameraExtrinsics(self, socket, useSpecTranslation):  # noqa: N802,N803
        return [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]

    def getBaselineDistance(self, a, b):  # noqa: N802 - depthai API
        return self._baseline_cm


class _Device:
    def __init__(self, calib):
        self._calib = calib

    def readCalibration(self):  # noqa: N802 - depthai API
        return self._calib


@pytest.fixture(autouse=True)
def _stub_depthai(monkeypatch):
    dai = types.ModuleType("depthai")
    dai.CameraBoardSocket = types.SimpleNamespace(CAM_B="CAM_B", CAM_C="CAM_C")
    monkeypatch.setitem(sys.modules, "depthai", dai)


def test_a_good_calibration_comes_back_flat():
    calib = OakdCapture._dump_calib_offline(_Device(_Calib()), (640, 400))

    assert calib["fx"] == 600.0
    assert calib["baseline"] == pytest.approx(0.075)
    assert {"width", "height", "fx", "fy", "cx", "cy", "baseline", "imu_to_cam"} <= calib.keys()


def test_an_unreadable_calibration_raises_instead_of_returning_empty():
    # It used to log a warning and return {} — start_recording then simply
    # skipped the file, which is the whole incident.
    with pytest.raises(OakdCalibrationError) as e:
        OakdCapture._dump_calib_offline(_Device(_Calib(raises=True)), (640, 400))

    assert "EEPROM read failed" in str(e.value)


@pytest.mark.parametrize("kwargs", [
    {"fx": 0.0},           # degenerate focal length
    {"baseline_cm": 0.0},  # no stereo baseline
])
def test_a_degenerate_calibration_is_refused(kwargs):
    # Present-but-unusable fails the same upstream check as absent, so it has to
    # fail here too — otherwise the file is written and the rejection just moves
    # back to the Space.
    with pytest.raises(OakdCalibrationError):
        OakdCapture._dump_calib_offline(_Device(_Calib(**kwargs)), (640, 400))


def test_start_recording_refuses_without_a_calibration(tmp_path):
    # Belt and braces behind the init-time gate: even if something bypassed it,
    # no episode dir is created and nothing is recorded.
    cap = object.__new__(OakdCapture)
    cap._initialized = True
    cap._recording = False
    cap.sync = types.SimpleNamespace(is_started=True)
    cap._calib_offline = None
    ep = tmp_path / "20250101_120000"

    with pytest.raises(OakdCalibrationError):
        OakdCapture.start_recording(cap, ep)

    assert not ep.exists()


# --- the error LED ------------------------------------------------------------

class _FakeLine:
    """Records every (t, value) written to the LED line."""

    def __init__(self):
        self.writes = []
        self._t0 = time.monotonic()

    def set_value(self, pin, value):
        self.writes.append((time.monotonic() - self._t0, value))


def _led():
    btn = object.__new__(LedButton)
    btn._led_pin = 11
    btn._led_request = _FakeLine()
    btn._blink_thread = None
    btn._blink_stop = threading.Event()
    return btn


def test_error_pattern_is_a_burst_of_three_then_a_gap():
    btn = _led()
    btn.led_pulses(count=3, on=0.01, off=0.01, gap=1.0)
    time.sleep(0.3)  # well inside the gap: one burst has fired, the next has not
    btn.led_off()

    writes = btn._led_request.writes
    ons = [t for t, v in writes if v == "active"]
    assert len(ons) == 3, f"expected one burst of 3 pulses, got {len(ons)}"
    # The dark gap is what makes the pattern read as a fault rather than as yet
    # another blink: the whole burst must be brief next to the pause following it.
    burst_span = ons[-1] - ons[0]
    assert burst_span < 0.25, f"burst took {burst_span:.3f}s — it should be a blip"
    assert writes[-1][1] == "inactive", "the LED must end the burst dark"


def test_error_pattern_repeats():
    btn = _led()
    btn.led_pulses(count=3, on=0.01, off=0.01, gap=0.05)
    time.sleep(0.35)
    btn.led_off()

    ons = [t for t, v in btn._led_request.writes if v == "active"]
    assert len(ons) >= 6, "the burst must repeat, not fire once"


def test_switching_pattern_stops_the_previous_one():
    # Two pattern threads on one line produce a pattern that is neither. A
    # switch must leave exactly one writer.
    btn = _led()
    btn.led_pulses(count=3, on=0.01, off=0.01, gap=0.05)
    time.sleep(0.05)
    btn.led_blink(0.01)
    time.sleep(0.05)
    live = [t for t in threading.enumerate() if t is btn._blink_thread]
    btn.led_off()

    assert len(live) == 1
    assert threading.active_count() < 10  # no thread pile-up


# --- the LED state machine ----------------------------------------------------

class _Backend:
    is_teleop_active = False
    is_teleop_sending = False
    is_stopping = False
    is_starting = False
    is_capturing = False
    hardware_error = ""


def _desired(**state):
    from grabette.button_listener import ButtonListener

    b = _Backend()
    for k, v in state.items():
        setattr(b, k, v)
    listener = ButtonListener(b, None)
    return listener._desired_led()


def test_hardware_error_drives_the_error_state():
    assert _desired(hardware_error="no OAK-D calibration") == "error"


def test_hardware_error_outranks_every_busy_state():
    # A fault the busy patterns can hide is a fault nobody sees.
    assert _desired(hardware_error="boom", is_capturing=True) == "error"
    assert _desired(hardware_error="boom", is_starting=True) == "error"
    assert _desired(hardware_error="boom", is_teleop_active=True) == "error"


def test_no_error_leaves_the_normal_states_alone():
    assert _desired() == "off"
    assert _desired(is_capturing=True) == "on"
    assert _desired(is_stopping=True) == "blink_fast"


# --- the backend gate: refuse to record, but stay recoverable -----------------

class _FakeOakd:
    """Stands in for OakdCapture: init_device either works or fails the way a
    calibration read does."""

    instances = []

    def __init__(self, sync, fail=False):
        self.fail = fail
        self.is_initialized = False

    def init_device(self):
        if self.fail:
            raise OakdCalibrationError("could not read the OAK-D calibration: EEPROM read failed")
        self.is_initialized = True


def _backend(monkeypatch, *, fail):
    from grabette.backend import rpi
    from grabette.hardware import oakd as oakd_mod

    state = {"fail": fail}

    def _factory(sync):
        return _FakeOakd(sync, fail=state["fail"])

    monkeypatch.setattr(oakd_mod, "OakdCapture", _factory)
    b = rpi.RpiBackend()
    return b, state


def test_a_calibration_failure_latches_a_hardware_error(monkeypatch):
    b, _ = _backend(monkeypatch, fail=True)

    b._init_oakd()

    assert "calibration" in b.hardware_error
    # And it names what to do about it — a fault the operator can't act on is
    # only marginally better than a silent one.
    assert "Power-cycle" in b.hardware_error


def test_capture_is_refused_while_the_fault_stands(monkeypatch, tmp_path):
    b, _ = _backend(monkeypatch, fail=True)

    with pytest.raises(RuntimeError) as e:
        asyncio_run(b.start_capture(tmp_path / "ep"))

    assert "calibration" in str(e.value)
    assert not b.is_capturing
    assert not (tmp_path / "ep").exists()  # nothing was recorded


def test_the_fault_clears_once_the_calibration_reads(monkeypatch, tmp_path):
    # The gate runs AFTER the OAK-D bring-up, so pressing again retries the
    # calibration read. Without that, a fault latched at boot would make a reboot
    # the only way out — the same dead end this whole change is about.
    b, state = _backend(monkeypatch, fail=True)
    with pytest.raises(RuntimeError):
        asyncio_run(b.start_capture(tmp_path / "ep1"))

    state["fail"] = False  # cable reseated
    b._init_oakd()

    assert b.hardware_error == ""


def test_a_plain_missing_oakd_is_not_a_fault(monkeypatch):
    # A bench setup with no OAK-D attached keeps working as before: only an
    # unusable CALIBRATION is treated as a fault.
    from grabette.hardware import oakd as oakd_mod

    def _boom(sync):
        raise RuntimeError("no device found")

    monkeypatch.setattr(oakd_mod, "OakdCapture", _boom)
    from grabette.backend import rpi
    b = rpi.RpiBackend()

    b._init_oakd()

    assert b.hardware_error == ""


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)
