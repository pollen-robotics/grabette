"""The depth-camera Protocol is only useful if implementations really satisfy it.

`RpiBackend` calls these members on whatever `depth_camera` selected, so a member
going missing or changing arity is a runtime AttributeError/TypeError on the Pi,
during a capture, with hardware attached — the worst place to find out. These
tests catch it on a workstation with no camera plugged in.

`isinstance` against a runtime_checkable Protocol only checks that names exist,
which is too weak on its own: it would pass a `start_recording` that took the
wrong arguments. So signatures are compared explicitly as well.
"""
import inspect

import pytest

from grabette.hardware.depth_camera import DepthCameraCapture

# depthai is an optional [rpi] dependency and is absent on a plain dev install.
# The module only imports it inside methods, so importing the class is safe, but
# skip rather than fail if the environment is missing cv2/numpy entirely.
oakd = pytest.importorskip("grabette.hardware.oakd")
OakdCapture = oakd.OakdCapture


# Members RpiBackend actually calls, with the arity it calls them at.
# Keep this list in sync with backend/rpi.py, not with the Protocol — the point
# is to pin what the *caller* depends on.
REQUIRED_METHODS = [
    "init_device",
    "shutdown",
    "wait_until_ready",
    "start_recording",
    "stop_recording",
    "get_latest_imu",
    "get_depth_jpeg",
]
REQUIRED_PROPERTIES = ["is_initialized", "is_recording", "imu_sample_count"]


def test_oakd_capture_satisfies_protocol():
    # issubclass() is rejected for Protocols carrying non-method members (the
    # three properties), so check an instance. __init__ only assigns fields —
    # it touches no hardware — so a None SyncManager is fine here.
    assert isinstance(OakdCapture(None), DepthCameraCapture)


@pytest.mark.parametrize("name", REQUIRED_METHODS)
def test_method_present_and_callable(name):
    attr = getattr(OakdCapture, name, None)
    assert attr is not None, f"OakdCapture is missing {name}()"
    assert callable(attr), f"OakdCapture.{name} is not callable"


@pytest.mark.parametrize("name", REQUIRED_PROPERTIES)
def test_property_present(name):
    attr = getattr(OakdCapture, name, None)
    assert isinstance(attr, property), f"OakdCapture.{name} must be a property"


def test_start_recording_takes_an_output_dir():
    # RpiBackend calls start_recording(session_dir) positionally.
    sig = inspect.signature(OakdCapture.start_recording)
    params = [p for p in sig.parameters if p != "self"]
    assert len(params) == 1, f"expected one arg, got {params}"


def test_wait_until_ready_accepts_a_bare_timeout():
    # RpiBackend calls run_in_executor(None, wait_until_ready, TIMEOUT) — a
    # single positional. Extra params are fine only if they have defaults.
    sig = inspect.signature(OakdCapture.wait_until_ready)
    params = [p for n, p in sig.parameters.items() if n != "self"]
    assert params, "wait_until_ready takes no timeout"
    assert all(p.default is not inspect.Parameter.empty for p in params[1:]), (
        "every parameter after `timeout` must have a default, or RpiBackend's "
        "single-positional call breaks"
    )


def test_stop_recording_reports_imu_samples_key():
    # rpi.py does oakd_stats.get("imu_samples", 0) for metadata.json. A camera
    # with no IMU must still return the key (as 0) rather than omit it.
    src = inspect.getsource(OakdCapture.stop_recording)
    assert "imu_samples" in src, "stop_recording must report an imu_samples count"
