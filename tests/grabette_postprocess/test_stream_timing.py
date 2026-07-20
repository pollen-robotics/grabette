"""Cross-stream timing invariants.

The OAK and the Arducam are rigidly co-mounted and recorded together, so their
timestamp streams must span (roughly) the same wall-clock window. A large
duration mismatch means a truncated/dropped stream — data that silently corrupts
the synchronized dataset. These tests pin the timestamp loaders and that
duration-similarity invariant.
"""

import json

import numpy as np

from grabette_postprocess.checks.sync import (
    arducam_frame_ts,
    load_oak_gyro_norm,
    oak_left_frame_ts,
)


def _duration(ts):
    return float(ts[-1] - ts[0])


def test_oak_left_frame_ts_prefers_device_clock(tmp_path):
    """When device_us is present it is used (in seconds); host_ms is ignored."""
    # device_us present -> used (in seconds); host_ms ignored.
    samples = [{"seq": i, "device_us": i * 1_000_000, "host_ms": i * 999}
               for i in range(4)]
    (tmp_path / "oakd_left_timestamps.json").write_text(json.dumps({"samples": samples}))
    ts = oak_left_frame_ts(tmp_path)
    np.testing.assert_allclose(ts, [0.0, 1.0, 2.0, 3.0])


def test_oak_left_frame_ts_falls_back_to_host_ms(tmp_path):
    """Legacy recordings without device_us fall back to host_ms (ms→s)."""
    # Legacy recording without device_us -> host_ms (ms -> s).
    samples = [{"seq": i, "host_ms": i * 500} for i in range(3)]
    (tmp_path / "oakd_left_timestamps.json").write_text(json.dumps({"samples": samples}))
    np.testing.assert_allclose(oak_left_frame_ts(tmp_path), [0.0, 0.5, 1.0])


def test_oak_left_frame_ts_missing_is_none(tmp_path):
    """A missing timestamps file returns None."""
    assert oak_left_frame_ts(tmp_path) is None


def test_arducam_frame_ts_reads_ms_list(tmp_path):
    """Arducam frame timestamps load from a ms list (ms→s); absent/empty → None."""
    (tmp_path / "frame_timestamps.json").write_text(json.dumps([0.0, 100.0, 200.0]))
    np.testing.assert_allclose(arducam_frame_ts(tmp_path), [0.0, 0.1, 0.2])
    # Absent/empty -> None (caller falls back to uniform fps).
    assert arducam_frame_ts(tmp_path / "nope") is None


def test_load_oak_gyro_norm(tmp_path):
    """Only gyro samples are loaded, with device_us times and per-sample vector norms."""
    samples = [
        {"kind": "gyro", "device_us": 0, "value": [3.0, 4.0, 0.0]},        # norm 5
        {"kind": "gyro", "device_us": 1_000_000, "value": [0.0, 0.0, 2.0]},  # norm 2
        {"kind": "accel", "device_us": 0, "value": [0.0, 0.0, 9.81]},       # ignored
    ]
    (tmp_path / "oakd_imu.json").write_text(json.dumps({"samples": samples}))
    ts, norms = load_oak_gyro_norm(tmp_path / "oakd_imu.json")
    np.testing.assert_allclose(ts, [0.0, 1.0])
    np.testing.assert_allclose(norms, [5.0, 2.0])


def test_oak_and_arducam_span_similar_durations(valid_episode):
    """On a well-formed episode the OAK and Arducam streams cover the same window."""
    # Well-formed episode: the two camera streams cover the same window.
    oak = oak_left_frame_ts(valid_episode)
    arducam = arducam_frame_ts(valid_episode)
    assert oak is not None and arducam is not None
    assert abs(_duration(oak) - _duration(arducam)) < 0.05


def test_truncated_arducam_breaks_duration_match(tmp_path, episode_builder):
    """A stream recorded far longer than the other breaks the duration-match invariant."""
    # A stream recorded far longer than the other is exactly the mismatch the
    # invariant is meant to catch.
    ep = episode_builder(tmp_path / "ep", n_frames=4)
    # OAK left runs ~10x longer than the Arducam's 4 * 33 ms window.
    long_oak = [{"seq": i, "device_us": i * 330_000, "host_ms": i * 330}
                for i in range(4)]
    (ep / "oakd_left_timestamps.json").write_text(json.dumps({"samples": long_oak}))
    oak = oak_left_frame_ts(ep)
    arducam = arducam_frame_ts(ep)
    assert abs(_duration(oak) - _duration(arducam)) > 0.5
