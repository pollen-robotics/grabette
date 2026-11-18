"""StereoUvcCapture against a fake V4L2 camera: what lands on disk, and when.

The fake hands back raw MJPG buffers (as CAP_PROP_CONVERT_RGB=0 does) with a
kernel-style CAP_PROP_POS_MSEC on the monotonic clock, so the recording path
runs exactly as on the Pi, minus the hardware.
"""
import json
import time

import cv2
import numpy as np
import pytest

from grabette.hardware import stereo_uvc
from grabette.hardware.stereo_uvc import StereoUvcCapture
from grabette.hardware.sync import SyncManager

W, H = 3040, 1080
# The SKL sends [right|left]: the second half is the LEFT eye. Make the halves
# distinguishable so the preview can be checked for the swap.
_FRAME = np.zeros((H, W, 3), np.uint8)
_FRAME[:, W // 2:] = 200  # left eye (bright)
_JPEG = cv2.imencode(".jpg", _FRAME)[1]  # 1-D uint8, like a raw MJPG buffer


class FakeCapture:
    """Stands in for cv2.VideoCapture(index, CAP_V4L2) on the SKL."""

    def __init__(self, index, api, negotiated=(W, H)):
        self.props = {}
        self.negotiated = negotiated
        self.pos_ms = 0.0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.negotiated[0]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.negotiated[1]
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pos_ms
        return self.props.get(prop, 0)

    def read(self):
        time.sleep(1 / 200)  # pace the reader thread
        # Kernel timestamp: on CLOCK_MONOTONIC, a little before read() returns.
        self.pos_ms = time.monotonic() * 1000 - 30
        return True, _JPEG.copy()

    def release(self):
        self.released = True


@pytest.fixture
def fake_camera(monkeypatch):
    fakes = []

    def make(index, api):
        fakes.append(FakeCapture(index, api))
        return fakes[-1]

    monkeypatch.setattr(stereo_uvc.cv2, "VideoCapture", make)
    monkeypatch.setattr(stereo_uvc, "find_video_node",
                        lambda card: (1, "SPCA2100 PC Camera: PC Camera"))
    return fakes


def count_jpegs(data: bytes) -> int:
    """Number of JPEG images in a concatenated MJPG stream (SOI markers)."""
    return data.count(b"\xff\xd8\xff")


def test_records_raw_mjpg_with_matching_timestamps(fake_camera, tmp_path):
    sync = SyncManager()
    cap = StereoUvcCapture(sync)
    cap.init_device()
    assert cap.wait_until_ready(timeout=2.0)

    sync.start()
    cap.start_recording(tmp_path)
    time.sleep(0.2)
    stats = cap.stop_recording()
    cap.shutdown()

    samples = json.loads((tmp_path / "dcam_stereo_timestamps.json").read_text())["samples"]
    data = (tmp_path / "dcam_stereo.mjpeg").read_bytes()
    # One timestamp per frame on disk, frames stored byte-for-byte.
    assert stats["stereo_frames"] == len(samples) > 10
    assert count_jpegs(data) == len(samples)
    assert data[:len(_JPEG)] == _JPEG.tobytes()
    assert stats["imu_samples"] == 0
    # Kernel stamps land on the SyncManager timeline, in order.
    assert [s["seq"] for s in samples] == list(range(len(samples)))
    host = [s["host_ms"] for s in samples]
    assert all(b > a for a, b in zip(host, host[1:]))
    assert -50 < host[0] < 100
    # Calibration travels with the episode.
    calib = json.loads((tmp_path / "dcam_calib_stereo.json").read_text())
    assert "baseline_m" in calib or "T" in calib
    info = json.loads((tmp_path / "dcam_calib.json").read_text())
    assert info["profile"] == "skl2mp220" and info["swap_eyes"] is True


def test_sets_mjpg_passthrough_and_manual_exposure(fake_camera):
    cap = StereoUvcCapture(SyncManager(), exposure=250)
    cap.init_device()
    props = fake_camera[0].props
    cap.shutdown()
    assert props[cv2.CAP_PROP_CONVERT_RGB] == 0
    assert props[cv2.CAP_PROP_EXPOSURE] == 250
    # Shutdown hands auto-exposure back (UVC controls persist on the device).
    assert props[cv2.CAP_PROP_AUTO_EXPOSURE] == 3
    assert fake_camera[0].released


def test_gamma_set_and_restored(fake_camera):
    cap = StereoUvcCapture(SyncManager(), exposure=150, gamma=180)
    cap.init_device()
    gamma_while_running = fake_camera[0].props[cv2.CAP_PROP_GAMMA]
    cap.shutdown()
    assert gamma_while_running == 180
    # Back to the camera default: UVC controls persist on the device.
    assert fake_camera[0].props[cv2.CAP_PROP_GAMMA] == 100


def test_gamma_untouched_by_default(fake_camera):
    cap = StereoUvcCapture(SyncManager())
    cap.init_device()
    cap.shutdown()
    assert cv2.CAP_PROP_GAMMA not in fake_camera[0].props


class FlakyCapture(FakeCapture):
    """Fails `fail` reads in a row after `after` good frames, then recovers."""

    def __init__(self, index, api, after=20, fail=5):
        super().__init__(index, api)
        self.n, self.after, self.fail = 0, after, fail

    def read(self):
        self.n += 1
        if self.after < self.n <= self.after + self.fail:
            time.sleep(1 / 200)
            return False, None
        return super().read()


class StallingCapture(FakeCapture):
    """Stops delivering frames (read blocks) after `after` frames."""

    def __init__(self, index, api, after=20):
        super().__init__(index, api)
        self.n, self.after = 0, after

    def read(self):
        self.n += 1
        if self.n > self.after:
            time.sleep(0.05)  # blocked in the driver, no frame
            return False, None
        return super().read()


def _record(monkeypatch, tmp_path, cls, seconds):
    monkeypatch.setattr(stereo_uvc.cv2, "VideoCapture", cls)
    monkeypatch.setattr(stereo_uvc, "find_video_node", lambda card: (1, "SPCA2100"))
    sync = SyncManager()
    sync.start()
    cap = StereoUvcCapture(sync)
    cap.init_device()
    cap.start_recording(tmp_path)
    time.sleep(seconds)
    stats = cap.stop_recording()
    cap.shutdown()
    return stats


def test_failed_reads_are_counted(monkeypatch, tmp_path, caplog):
    stats = _record(monkeypatch, tmp_path, FlakyCapture, 0.3)
    assert stats["read_failures"] == 5
    assert "failed reads" in caplog.text


def test_frames_stopping_before_stop_is_a_gap(monkeypatch, tmp_path, caplog):
    # The case that went unnoticed: frames stop arriving well before the stop,
    # with no gap BETWEEN recorded frames.
    stats = _record(monkeypatch, tmp_path, StallingCapture, 0.5)
    assert stats["max_gap_s"] > 0.3
    assert "longest gap" in caplog.text


def test_clean_recording_has_no_warning(fake_camera, tmp_path, caplog):
    sync = SyncManager()
    sync.start()
    cap = StereoUvcCapture(sync)
    cap.init_device()
    cap.start_recording(tmp_path)
    time.sleep(0.2)
    stats = cap.stop_recording()
    cap.shutdown()
    assert stats["read_failures"] == 0
    assert stats["max_gap_s"] < 0.1
    assert "longest gap" not in caplog.text


def test_nothing_written_while_idle(fake_camera, tmp_path):
    cap = StereoUvcCapture(SyncManager())
    cap.init_device()
    time.sleep(0.1)
    cap.shutdown()
    assert list(tmp_path.iterdir()) == []


def test_preview_is_the_left_eye_after_swap(fake_camera):
    cap = StereoUvcCapture(SyncManager())
    cap.init_device()
    deadline = time.monotonic() + 2.0
    while cap.get_depth_jpeg() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    jpg = cap.get_depth_jpeg()
    cap.shutdown()
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    # Quarter-scale single eye, and the bright (left) half.
    assert img.shape == (H // 4, W // 8)
    assert img.mean() > 150


def test_init_fails_on_wrong_negotiated_size(monkeypatch):
    monkeypatch.setattr(stereo_uvc.cv2, "VideoCapture",
                        lambda i, a: FakeCapture(i, a, negotiated=(1280, 480)))
    monkeypatch.setattr(stereo_uvc, "find_video_node", lambda card: (1, "SPCA2100"))
    with pytest.raises(RuntimeError, match="negotiated 1280x480"):
        StereoUvcCapture(SyncManager()).init_device()


def test_init_fails_without_camera(monkeypatch):
    monkeypatch.setattr(stereo_uvc, "find_video_node", lambda card: None)
    with pytest.raises(RuntimeError, match="No skl2mp220 camera"):
        StereoUvcCapture(SyncManager()).init_device()


def test_every_profile_mode_has_a_calibration():
    # init_device() refuses to start without one, so a missing file would only
    # show up on the Pi.
    for name, prof in stereo_uvc.PROFILES.items():
        for mode in prof["modes"]:
            assert (stereo_uvc._CALIB_DIR / f"{name}_{mode}.json").exists(), (name, mode)
