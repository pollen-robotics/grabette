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
orbbec = pytest.importorskip("grabette.hardware.orbbec")
OrbbecCapture = orbbec.OrbbecCapture

# Both implementations must satisfy the same contract; RpiBackend picks between
# them at runtime from the `depth_camera` setting and cannot tell them apart.
IMPLS = [OakdCapture, OrbbecCapture]


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
    "camera_info",
]
REQUIRED_PROPERTIES = ["is_initialized", "is_recording", "imu_sample_count"]


@pytest.mark.parametrize("cls", IMPLS, ids=lambda c: c.__name__)
def test_capture_satisfies_protocol(cls):
    # issubclass() is rejected for Protocols carrying non-method members (the
    # three properties), so check an instance. __init__ only assigns fields —
    # it touches no hardware — so a None SyncManager is fine here.
    assert isinstance(cls(None), DepthCameraCapture)


@pytest.mark.parametrize("cls", IMPLS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", REQUIRED_METHODS)
def test_method_present_and_callable(cls, name):
    attr = getattr(cls, name, None)
    assert attr is not None, f"{cls.__name__} is missing {name}()"
    assert callable(attr), f"{cls.__name__}.{name} is not callable"


@pytest.mark.parametrize("cls", IMPLS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", REQUIRED_PROPERTIES)
def test_property_present(cls, name):
    attr = getattr(cls, name, None)
    assert isinstance(attr, property), f"{cls.__name__}.{name} must be a property"


@pytest.mark.parametrize("cls", IMPLS, ids=lambda c: c.__name__)
def test_start_recording_takes_an_output_dir(cls):
    # RpiBackend calls start_recording(session_dir) positionally.
    sig = inspect.signature(cls.start_recording)
    params = [p for p in sig.parameters if p != "self"]
    assert len(params) == 1, f"expected one arg, got {params}"


@pytest.mark.parametrize("cls", IMPLS, ids=lambda c: c.__name__)
def test_wait_until_ready_accepts_a_bare_timeout(cls):
    # RpiBackend calls run_in_executor(None, wait_until_ready, TIMEOUT) — a
    # single positional. Extra params are fine only if they have defaults.
    sig = inspect.signature(cls.wait_until_ready)
    params = [p for n, p in sig.parameters.items() if n != "self"]
    assert params, "wait_until_ready takes no timeout"
    assert all(p.default is not inspect.Parameter.empty for p in params[1:]), (
        "every parameter after `timeout` must have a default, or RpiBackend's "
        "single-positional call breaks"
    )


@pytest.mark.parametrize("cls", IMPLS, ids=lambda c: c.__name__)
def test_stop_recording_reports_imu_samples_key(cls):
    # rpi.py does oakd_stats.get("imu_samples", 0) for metadata.json. A camera
    # with no IMU must still return the key (as 0) rather than omit it.
    src = inspect.getsource(cls.stop_recording)
    assert "imu_samples" in src, "stop_recording must report an imu_samples count"


def test_orbbec_reports_no_imu():
    # The 305 genuinely has no IMU. Consumers must see a clean "no IMU" rather
    # than a transient None that might later become a sample.
    cap = OrbbecCapture(None)
    assert cap.get_latest_imu() is None
    assert cap.imu_sample_count == 0


# ── camera identity for episode provenance ───────────────────────────────────

@pytest.mark.parametrize("cls", IMPLS, ids=lambda c: c.__name__)
def test_camera_info_present_and_returns_a_dict(cls):
    # rpi.py writes this into metadata.json. Both cameras must answer it, and
    # answer it BEFORE init_device() too — stop_capture calls it on whatever
    # state the device is in.
    cap = cls(None)
    info = cap.camera_info()
    assert isinstance(info, dict)


def test_orbbec_camera_info_reports_no_imu():
    # The absence has to be recorded, not merely implied by a missing file.
    cap = OrbbecCapture(None)
    assert cap.camera_info()["imu"] is None


def test_camera_info_is_in_the_protocol():
    from grabette.hardware.depth_camera import DepthCameraCapture

    assert hasattr(DepthCameraCapture, "camera_info")


def test_camera_metadata_preserves_a_meaningful_null():
    # Regression: an earlier version filtered None out of the metadata block,
    # which deleted the Gemini's "imu": None — the one field that records the
    # camera has no IMU rather than leaving it inferred from a missing file.
    from grabette.backend.rpi import _camera_metadata

    meta = _camera_metadata("gemini305", OrbbecCapture(None))
    assert meta["model"] == "gemini305"
    assert "imu" in meta, "the explicit no-IMU marker was dropped"
    assert meta["imu"] is None


def test_camera_metadata_survives_a_camera_that_raises():
    # stop_capture must never fail because identity could not be read.
    from grabette.backend.rpi import _camera_metadata

    class Broken:
        def camera_info(self):
            raise RuntimeError("device gone")

    assert _camera_metadata("oakd", Broken()) == {"model": "oakd"}


def test_camera_metadata_without_a_camera():
    from grabette.backend.rpi import _camera_metadata

    assert _camera_metadata("oakd", None) == {"model": "oakd"}
