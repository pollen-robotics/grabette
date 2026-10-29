"""An encoder slow to start must not punch a hole in the recording.

On the Pi, ffmpeg takes 0.3-3.5 s before it reads its stdin. When the capture
loop wrote to the pipe itself, it blocked for that long and the SDK dropped every
frame in between: each Orbbec episode had one frame, then a gap. Here a fake
camera produces frames at 30 fps and fake encoders wait 1 s before reading.
"""
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest

orbbec = pytest.importorskip("grabette.hardware.orbbec")

W, H = 64, 40
PERIOD_S = 1 / 30
ENCODER_START_S = 1.0


class _FakeFrame:
    def __init__(self, data, index, t_us):
        self._data, self._index, self._t_us = data, index, t_us

    def get_data(self):
        return self._data

    def get_height(self):
        return H

    def get_width(self):
        return W

    def get_depth_scale(self):
        return 1.0

    def get_index(self):
        return self._index

    def get_timestamp_us(self):
        return self._t_us

    def get_global_timestamp_us(self):
        return self._t_us


class _FakePipeline:
    """Delivers a frame every PERIOD_S, dropping those nobody waits for, like the SDK."""

    def __init__(self):
        self._t0 = time.monotonic()

    def wait_for_frames(self, timeout_ms):
        now = time.monotonic() - self._t0
        index = int(now / PERIOD_S) + 1
        time.sleep(index * PERIOD_S - now)
        t_us = int(index * PERIOD_S * 1e6)
        depth = _FakeFrame(np.full((H, W), 500, np.uint16).tobytes(), index, t_us)
        ir = _FakeFrame(np.zeros((H, W), np.uint8).tobytes(), index, t_us)
        return types.SimpleNamespace(get_depth_frame=lambda: depth,
                                     get_left_ir_frame=lambda: ir)


def _slow_encoder(path):
    code = ("import shutil, sys, time; time.sleep(%r); "
            "shutil.copyfileobj(sys.stdin.buffer, open(sys.argv[1], 'wb'))" % ENCODER_START_S)
    return subprocess.Popen([sys.executable, "-c", code, str(path)],
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def test_slow_encoder_start_leaves_no_gap(tmp_path, monkeypatch):
    cap = orbbec.OrbbecCapture(
        types.SimpleNamespace(is_started=True, monotonic_s_to_ms=lambda s: s * 1000.0),
        depth_resolution=(W, H))
    cap._pipeline = _FakePipeline()
    cap._initialized = True
    monkeypatch.setattr(cap, "_spawn_encoder", _slow_encoder)
    monkeypatch.setattr(cap, "_spawn_depth_encoder", _slow_encoder)
    loop = threading.Thread(target=cap._writer_loop, daemon=True)
    loop.start()

    cap.start_recording(tmp_path)
    time.sleep(ENCODER_START_S + 0.5)
    stats = cap.stop_recording()
    cap._stop_event.set()
    loop.join(timeout=2)

    seqs = [s["seq"] for s in cap._left_ts]
    assert np.diff(seqs).max() == 1, (
        f"{np.diff(seqs).max() - 1} frames lost after seq {seqs[np.diff(seqs).argmax()]}")
    assert len(seqs) > 30
    assert [s["seq"] for s in cap._depth_ts] == seqs
    assert stats["dropped_frames"] == 0
    assert (tmp_path / "dcam_left.mp4").stat().st_size == len(seqs) * W * H
    assert (tmp_path / "dcam_depth.mkv").stat().st_size == len(seqs) * W * H * 2
