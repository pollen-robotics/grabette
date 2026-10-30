"""Saved pictures, rather than preview callbacks, own their capture times."""
import gc
import json
import sys
import types

import pytest

from grabette.hardware import camera as camera_module
from grabette.hardware.camera import FrameAlignmentError, VideoCapture


class FakeSync:
    is_started = True

    def boottime_ns_to_ms(self, sensor_ns):
        return (sensor_ns - 1_000_000_000) / 1_000_000

    def get_timestamp_ms(self):
        raise AssertionError("Do not replace missing capture times with delivery time")


class FakeFileOutput:
    """Model FileOutput's keyframe gate and write-before-outputtimestamp order."""
    def __init__(self, path):
        self.dead = False
        self.fail_next = False
        self.dead_next = False
        self.have_keyframe = False
        self.written = []
        self.closed = False

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
        if audio or (not self.have_keyframe and not keyframe):
            return
        self.have_keyframe = True
        if self.fail_next:
            self.fail_next = False
            raise OSError("injected write/flush failure")
        if self.dead_next:
            self.dead_next = False
            self.dead = True
            return
        self.written.append(frame)
        self.outputtimestamp(timestamp)

    def close(self):
        self.closed = True


class FakeEncoder:
    def __init__(self, bitrate):
        pass

    def _timestamp(self, request):
        return request.pts_us


class FakeCamera:
    def create_video_configuration(self, **kwargs):
        return kwargs

    def configure(self, configuration):
        self.configuration = configuration

    def start(self):
        pass

    def start_encoder(self, encoder, output):
        self.output = output

    def stop_encoder(self):
        if hasattr(self, "drain"):
            self.drain()
        self.output.close()

    def stop(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def capture(monkeypatch, tmp_path):
    modules = {
        "picamera2": types.SimpleNamespace(Picamera2=FakeCamera, Preview=object()),
        "picamera2.encoders": types.SimpleNamespace(H264Encoder=FakeEncoder),
        "picamera2.outputs": types.SimpleNamespace(FileOutput=FakeFileOutput),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    enabled = gc.isenabled()
    camera = VideoCapture(FakeSync())
    camera.init_camera()
    hardware = camera._picam2
    path = tmp_path / "raw_video.mp4"
    camera.start_recording(path)
    path.with_suffix(".h264").write_bytes(b"recoverable original bitstream")

    def mux():
        camera._frame_count = len(hardware.output.written)
        camera._has_b_frames = 0

    monkeypatch.setattr(camera, "_mux_to_mp4", mux)
    try:
        yield camera, hardware, path
    finally:
        gc.enable() if enabled else gc.disable()


def frame(camera, hardware, pts, sensor_ns, *, keyframe=True):
    request = types.SimpleNamespace(
        pts_us=pts, get_metadata=lambda: {"SensorTimestamp": sensor_ns})
    # Exercise the actual encoder override installed by init_camera().
    assert camera._encoder._timestamp(request) == pts
    hardware.output.outputframe(b"picture", keyframe=keyframe, timestamp=pts)


def test_written_frames_retain_sensor_times_and_existing_fps_default(capture):
    camera, hardware, path = capture
    assert camera.fps == 46
    frame(camera, hardware, 0, 1_010_000_123)
    frame(camera, hardware, 25_000, 1_035_000_456, keyframe=False)
    assert camera.stop() == [10.000123, 35.000456]
    assert camera.timestamp_alignment["valid"]
    assert not path.with_suffix(".h264").exists()
    provenance = json.loads(path.with_suffix(".frames.json").read_text())
    assert [f["sensor_timestamp_ns"] for f in provenance["frames"]] == [
        1_010_000_123, 1_035_000_456]


def test_initial_nonkeyframe_does_not_shift_the_next_timestamp(capture):
    camera, hardware, _ = capture
    frame(camera, hardware, 0, 1_010_000_123, keyframe=False)
    frame(camera, hardware, 25_000, 1_035_000_456)
    assert camera.stop() == [35.000456]
    assert camera.timestamp_alignment["unwritten_input_count"] == 1


def test_encoder_drain_records_a_frame_after_recording_flag_is_cleared(capture):
    camera, hardware, _ = capture
    frame(camera, hardware, 0, 1_010_000_123)
    request = types.SimpleNamespace(
        pts_us=25_000, get_metadata=lambda: {"SensorTimestamp": 1_035_000_456})
    camera._encoder._timestamp(request)

    def drain():
        assert not camera._recording
        hardware.output.outputframe(b"last picture", keyframe=False, timestamp=25_000)

    hardware.drain = drain
    assert camera.stop() == [10.000123, 35.000456]


@pytest.mark.parametrize("failure", ["exception", "dead"])
def test_write_failure_is_reported_even_when_partial_prefix_counts_match(capture, failure):
    camera, hardware, path = capture
    frame(camera, hardware, 0, 1_010_000_123)
    if failure == "exception":
        hardware.output.fail_next = True
    else:
        hardware.output.dead_next = True
    # Returning from the callback lets the encoder release its buffers.
    frame(camera, hardware, 25_000, 1_035_000_456, keyframe=False)
    with pytest.raises(FrameAlignmentError):
        camera.stop()
    assert not camera.is_open
    assert camera.frame_timestamps == [10.000123]
    assert camera.timestamp_alignment["video_frame_count"] == 1
    assert camera.timestamp_alignment["timestamp_count"] == 1
    assert not camera.timestamp_alignment["valid"]
    assert path.with_suffix(".h264").exists()


def test_failed_provenance_write_preserves_source_and_reports_alignment_failure(capture, monkeypatch):
    camera, hardware, path = capture
    frame(camera, hardware, 0, 1_010_000_123)

    def fail(*args, **kwargs):
        raise OSError("injected provenance write failure")

    monkeypatch.setattr(camera_module, "write_json_atomic", fail)
    with pytest.raises(FrameAlignmentError, match="provenance"):
        camera.stop()
    assert path.with_suffix(".h264").exists()


def test_failed_encoder_start_restores_gc_and_closes_output(capture):
    camera, hardware, path = capture
    frame(camera, hardware, 0, 1_010_000_123)
    camera.stop()
    camera.init_camera()
    hardware = camera._picam2

    def fail(encoder, output):
        hardware.failed_output = output
        raise RuntimeError("encoder start failed")

    hardware.start_encoder = fail
    with pytest.raises(RuntimeError, match="encoder start"):
        camera.start_recording(path)
    assert not camera._recording
    assert hardware.failed_output.closed
    assert gc.isenabled()


def test_matching_counts_do_not_validate_unknown_presentation_order(capture, monkeypatch):
    camera, hardware, path = capture
    frame(camera, hardware, 0, 1_010_000_123)

    def mux():
        camera._frame_count = 1
        camera._has_b_frames = 2

    monkeypatch.setattr(camera, "_mux_to_mp4", mux)
    with pytest.raises(FrameAlignmentError, match="presentation order"):
        camera.stop()
    assert path.with_suffix(".h264").exists()


@pytest.mark.parametrize("failure", ["missing_sensor_time", "encoder_timestamp_error"])
def test_encoder_input_timestamp_failure_invalidates_a_matching_saved_prefix(capture, monkeypatch, failure):
    camera, hardware, path = capture
    frame(camera, hardware, 0, 1_010_000_123)
    request = types.SimpleNamespace(pts_us=25_000, get_metadata=lambda: {})
    if failure == "encoder_timestamp_error":
        def fail(self, request):
            raise RuntimeError("injected encoder timestamp failure")

        monkeypatch.setattr(FakeEncoder, "_timestamp", fail)
    # The worker may stop delivering frames, but stop() must still reject the
    # one-frame prefix even though its video/timestamp counts match exactly.
    with pytest.raises((KeyError, RuntimeError)):
        camera._encoder._timestamp(request)
    with pytest.raises(FrameAlignmentError, match="timestamp encoder input"):
        camera.stop()
    assert camera.timestamp_alignment["video_frame_count"] == 1
    assert camera.timestamp_alignment["timestamp_count"] == 1
    assert path.with_suffix(".h264").exists()
