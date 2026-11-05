"""Saved pictures, rather than preview callbacks, own their capture times."""
import gc
import json
import sys
import threading
import time
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


class FakeOutput:
    """picamera2's public Output base: the only part of picamera2 it relies on."""
    def __init__(self, pts=None):
        self.recording = False

    def start(self):
        self.recording = True

    def stop(self):
        self.recording = False


class FaultyFile:
    """Wraps the real .h264 file to stall or fail writes, like an SD card."""
    def __init__(self, file):
        self._file = file
        self.writes = 0
        self.gate = None  # threading.Event: writes wait on it, like a stalled SD card
        self.fail_write = False
        self.fail_flush = False
        self.fail_close = False

    @property
    def closed(self):
        return self._file.closed

    def write(self, frame):
        if self.gate is not None:
            self.gate.wait()
        if self.fail_write:
            self.fail_write = False
            raise OSError("injected write failure")
        self._file.write(frame)
        self.writes += 1

    def flush(self):
        if self.fail_flush:
            self.fail_flush = False
            raise OSError("injected flush failure")
        self._file.flush()

    def close(self):
        self._file.close()
        if self.fail_close:
            raise OSError("injected close failure")


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
        # As picamera2's Encoder.start: outputs are started before any frame.
        self.output = output
        output.start()

    def stop_encoder(self):
        if hasattr(self, "drain"):
            self.drain()
        self.output.stop()

    def stop(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def capture(monkeypatch, tmp_path):
    modules = {
        "picamera2": types.SimpleNamespace(Picamera2=FakeCamera, Preview=object()),
        "picamera2.encoders": types.SimpleNamespace(H264Encoder=FakeEncoder),
        "picamera2.outputs": types.SimpleNamespace(Output=FakeOutput),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    enabled = gc.isenabled()
    camera = VideoCapture(FakeSync())
    camera.init_camera()
    hardware = camera._picam2
    path = tmp_path / "raw_video.mp4"
    camera.start_recording(path)
    hardware.output._file = FaultyFile(hardware.output._file)

    def mux():
        camera._frame_count = hardware.output._file.writes

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
    # Writes run on the output's own thread: wait for this one, so a test can
    # inject a failure into the NEXT write.
    hardware.output._queue.join()


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


def test_camera_keeps_every_buffer_for_the_pipeline(capture):
    camera, hardware, _ = capture
    assert hardware.configuration["buffer_count"] == VideoCapture.DEFAULT_BUFFER_COUNT
    assert hardware.configuration["queue"] is False
    frame(camera, hardware, 0, 1_010_000_123)
    camera.stop()
    assert camera.timestamp_alignment["camera_buffer_count"] == VideoCapture.DEFAULT_BUFFER_COUNT


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


def test_a_stalled_write_neither_blocks_the_encoder_nor_loses_pictures(capture):
    camera, hardware, _ = capture
    hardware.output._file.gate = threading.Event()
    start = time.monotonic()
    for i in range(20):
        request = types.SimpleNamespace(
            pts_us=i * 20_000, get_metadata=lambda i=i: {"SensorTimestamp": 1_000_000_000 + i * 20_000_000})
        camera._encoder._timestamp(request)
        hardware.output.outputframe(b"picture", keyframe=i == 0, timestamp=i * 20_000)
    # The encoder thread handed every picture over without waiting for the disk.
    assert time.monotonic() - start < 0.5
    assert hardware.output._file.writes == 0
    # The disk recovers only once stop() is under way: what was queued must
    # still be written, although stopping clears the output's `recording`.
    threading.Timer(0.2, hardware.output._file.gate.set).start()
    assert camera.stop() == [i * 20.0 for i in range(20)]
    assert camera.timestamp_alignment["valid"]
    assert camera.timestamp_alignment["writer_queue_peak"] >= 19


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_write_failure_is_reported_even_when_partial_prefix_counts_match(capture, failure):
    camera, hardware, path = capture
    frame(camera, hardware, 0, 1_010_000_123)
    setattr(hardware.output._file, f"fail_{failure}", True)
    # Returning from the callback lets the encoder release its buffers.
    frame(camera, hardware, 25_000, 1_035_000_456, keyframe=False)
    with pytest.raises(FrameAlignmentError):
        camera.stop()
    assert not camera.is_open
    assert camera.frame_timestamps == [10.000123]
    if failure == "write":
        # Nothing reached the file: counts match, only the error can tell.
        assert camera.timestamp_alignment["video_frame_count"] == 1
    assert camera.timestamp_alignment["timestamp_count"] == 1
    assert any("write failed" in e for e in camera.timestamp_alignment["errors"])
    assert not camera.timestamp_alignment["valid"]
    assert path.with_suffix(".h264").exists()


def test_close_failure_invalidates_the_recording(capture):
    # A full SD card may only show up when the last buffered bytes are flushed.
    camera, hardware, path = capture
    frame(camera, hardware, 0, 1_010_000_123)
    hardware.output._file.fail_close = True
    with pytest.raises(FrameAlignmentError, match="close failed"):
        camera.stop()
    assert path.with_suffix(".h264").exists()


def test_the_h264_file_holds_exactly_the_written_pictures(capture):
    camera, hardware, path = capture
    frame(camera, hardware, 0, 1_010_000_123, keyframe=False)  # dropped: no keyframe yet
    frame(camera, hardware, 25_000, 1_035_000_456)
    frame(camera, hardware, 50_000, 1_060_000_789, keyframe=False)
    hardware.output.close()  # flush to disk before reading it back
    assert path.with_suffix(".h264").read_bytes() == b"picture" * 2


@pytest.mark.parametrize("call", ["old_three_args", "new_five_args"])
def test_output_accepts_both_picamera2_outputframe_signatures(capture, call):
    camera, hardware, _ = capture
    request = types.SimpleNamespace(
        pts_us=0, get_metadata=lambda: {"SensorTimestamp": 1_010_000_123})
    camera._encoder._timestamp(request)
    if call == "old_three_args":
        hardware.output.outputframe(b"picture", True, 0)
    else:
        hardware.output.outputframe(b"picture", True, 0, None, False)
    # An audio packet (never configured here) must not reach the H.264 file.
    hardware.output.outputframe(None, True, 0, object(), True)
    assert camera.stop() == [10.000123]


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
    assert hardware.failed_output._file.closed
    assert gc.isenabled()


def test_reordered_output_is_invalid_even_when_counts_match(capture):
    # B-frames (or any reordering) make the encoder emit pictures out of
    # presentation order. The output-PTS order check catches this without
    # relying on ffprobe's has_b_frames heuristic.
    camera, hardware, path = capture
    for pts, sensor_ns in ((0, 1_010_000_000), (25_000, 1_035_000_000)):
        request = types.SimpleNamespace(
            pts_us=pts, get_metadata=lambda ns=sensor_ns: {"SensorTimestamp": ns})
        camera._encoder._timestamp(request)
    hardware.output.outputframe(b"picture", keyframe=True, timestamp=25_000)
    hardware.output.outputframe(b"picture", keyframe=False, timestamp=0)
    hardware.output._queue.join()
    with pytest.raises(FrameAlignmentError, match="Non-increasing encoder output PTS"):
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
