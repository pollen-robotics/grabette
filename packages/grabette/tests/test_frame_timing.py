"""Focused contract tests for recorded-frame timestamp association."""

import random
import unittest

from grabette.hardware.frame_timing import LOCK_FRAMES, FrameTiming


class FakeSync:
    """Only original sensor timestamps may determine saved capture times."""

    def __init__(self, start_ns=1_000_000_000):
        self.start_ns = start_ns
        self.now_ns = start_ns

    def boottime_ns_to_ms(self, sensor_ns):
        return (sensor_ns - self.start_ns) / 1_000_000

    def get_timestamp_ms(self):
        raise AssertionError("Output-arrival time must not timestamp video frames")


def sensor_times(count, start_ns=1_010_000_123):
    """~20 ms frames whose intervals jitter by a few us, like the real sensor
    (median 6 us between consecutive intervals on the takes of 2026-11-04/05)."""
    rng = random.Random(count)
    return [start_ns + i * 20_000_000 + round(rng.gauss(0, 5_000)) for i in range(count)]


def picamera2_pts(sensor_ns, first_ns):
    # What picamera2 0.3.x does; the code under test must not rely on it.
    return int(sensor_ns / 1000) - int(first_ns / 1000)


class FrameTimingTests(unittest.TestCase):
    def setUp(self):
        self.sync = FakeSync()
        self.timing = FrameTiming(self.sync)

    def record(self, camera_ns, written_ns, pts=None):
        """Camera sees `camera_ns`; the output writes `written_ns` (a subset)."""
        pts = pts or (lambda ns: picamera2_pts(ns, written_ns[0]))
        for ns in camera_ns:
            self.timing.camera_frame(ns)
        for ns in written_ns:
            self.timing.written(pts(ns))
        self.timing.finish()
        return self.timing.result(len(written_ns))

    def expected_ms(self, written_ns):
        return [(ns - self.sync.start_ns) / 1_000_000 for ns in written_ns]

    def test_written_frames_get_their_exact_sensor_times(self):
        camera = sensor_times(40)
        # Camera frames before the encoder started and after it stopped are
        # not written: neither is a loss.
        written = camera[3:35]
        result = self.record(camera, written)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(self.timing.timestamps, self.expected_ms(written))
        self.assertEqual([f["sensor_timestamp_ns"] for f in self.timing.frames], written)
        self.assertEqual([f["frame_index"] for f in self.timing.frames], list(range(32)))
        self.assertEqual(result["pts_offset_us"], written[0] // 1000)
        self.assertEqual(result["camera_frame_count"], 40)
        self.assertEqual(result["unwritten_camera_frame_count"], 0)

    def test_any_pts_origin_is_found(self):
        camera = sensor_times(30)
        # E.g. a picamera2 that passes the absolute sensor time through.
        result = self.record(camera, camera, pts=lambda ns: ns // 1000)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["pts_offset_us"], 0)
        self.assertEqual(self.timing.timestamps, self.expected_ms(camera))

    def test_pts_rounding_differences_are_tolerated(self):
        camera = sensor_times(30)
        result = self.record(camera, camera,
                             pts=lambda ns: round(ns / 1000) - camera[0] // 1000 + 1)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(self.timing.timestamps, self.expected_ms(camera))

    def test_frame_skipped_by_encoder_keeps_exact_mapping(self):
        camera = sensor_times(30)
        written = camera[:12] + camera[13:]
        result = self.record(camera, written)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["unwritten_camera_frame_count"], 1)
        self.assertEqual(self.timing.timestamps, self.expected_ms(written))

    def test_ambiguous_offset_is_invalid_rather_than_guessed(self):
        # Perfectly regular frames: shifting by whole frames fits too.
        camera = [1_010_000_000 + i * 20_000_000 for i in range(30)]
        result = self.record(camera, camera[:20])
        self.assertFalse(result["valid"])
        self.assertIn("offsets fit", " ".join(result["errors"]))
        self.assertEqual(self.timing.timestamps, [])

    def test_pts_not_derived_from_sensor_time_is_invalid(self):
        camera = sensor_times(30)
        # A nominal frame-rate clock instead of the sensor: no offset fits.
        result = self.record(camera, camera, pts=lambda ns: camera.index(ns) * 20_000)
        self.assertFalse(result["valid"])
        self.assertIn("0 offsets", " ".join(result["errors"]))
        self.assertEqual(self.timing.timestamps, [])

    def test_short_take_is_matched_when_the_encoder_drains(self):
        camera = sensor_times(LOCK_FRAMES - 4)
        for ns in camera:
            self.timing.camera_frame(ns)
        for ns in camera:
            self.timing.written(picamera2_pts(ns, camera[0]))
        self.assertEqual(self.timing.timestamps, [])  # still waiting for LOCK_FRAMES
        self.timing.finish()
        result = self.timing.result(len(camera))
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(self.timing.timestamps, self.expected_ms(camera))

    def test_output_drained_after_capture_uses_capture_time(self):
        camera = sensor_times(20)
        for ns in camera:
            self.timing.camera_frame(ns)
        for ns in camera[:15]:
            self.timing.written(picamera2_pts(ns, camera[0]))
        # The camera has stopped; encoder draining happens later and must not
        # require any recording flag to remain set.
        self.sync.now_ns += 2_000_000_000
        for ns in camera[15:]:
            self.timing.written(picamera2_pts(ns, camera[0]))
        self.timing.finish()
        result = self.timing.result(20)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(self.timing.timestamps, self.expected_ms(camera))

    def test_unknown_output_is_invalid_without_fabricated_timestamp(self):
        camera = sensor_times(20)
        for ns in camera:
            self.timing.camera_frame(ns)
        for ns in camera:
            self.timing.written(picamera2_pts(ns, camera[0]))
        self.timing.written(picamera2_pts(camera[-1], camera[0]) + 20_000)
        self.timing.finish()
        result = self.timing.result(21)
        self.assertFalse(result["valid"])
        self.assertIn("No camera frame for output PTS", " ".join(result["errors"]))
        self.assertEqual(result["written_count"], 21)
        self.assertEqual(result["timestamp_count"], 20)

    def test_nonmonotonic_output_is_invalid(self):
        camera = sensor_times(20)
        for ns in camera:
            self.timing.camera_frame(ns)
        order = list(range(20))
        order[5], order[6] = order[6], order[5]
        for i in order:
            self.timing.written(picamera2_pts(camera[i], camera[0]))
        self.timing.finish()
        result = self.timing.result(20)
        self.assertFalse(result["valid"])
        self.assertIn("Non-increasing encoder output PTS", " ".join(result["errors"]))

    def test_nonincreasing_camera_times_are_invalid(self):
        camera = sensor_times(20)
        camera[8] = camera[7]
        for ns in camera:
            self.timing.camera_frame(ns)
        self.assertFalse(self.timing.result(0)["valid"])
        self.assertIn("Non-increasing camera sensor timestamp", " ".join(self.timing.errors))

    def test_missing_sensor_time_is_invalid(self):
        camera = sensor_times(20)
        result = self.record(camera[:5] + [None] + camera[5:], camera)
        self.assertFalse(result["valid"])
        self.assertIn("Missing sensor timestamp", " ".join(result["errors"]))

    def test_video_frame_count_mismatch_is_invalid(self):
        camera = sensor_times(20)
        self.record(camera, camera)
        for decoded_count in [19, 21]:
            with self.subTest(decoded_count=decoded_count):
                self.assertFalse(self.timing.result(decoded_count)["valid"])
        self.assertTrue(self.timing.result(20)["valid"])

    def test_empty_recording_is_invalid(self):
        self.timing.finish()
        self.assertFalse(self.timing.result(0)["valid"])

    def test_recorded_output_error_invalidates_matching_frame_counts(self):
        camera = sensor_times(20)
        self.record(camera, camera)
        self.timing.record_error("injected output flush failure")
        result = self.timing.result(20)
        self.assertEqual(result["video_frame_count"], result["timestamp_count"])
        self.assertFalse(result["valid"])
        self.assertIn("injected output flush failure", result["errors"])

    def test_separate_sessions_reset_mapping_and_error_state(self):
        self.record(sensor_times(20), sensor_times(20), pts=lambda ns: 7)
        self.assertFalse(self.timing.result(20)["valid"])

        new_sync = FakeSync(start_ns=10_000_000_000)
        new_timing = FrameTiming(new_sync)
        camera = sensor_times(12, start_ns=10_005_000_123)
        for ns in camera:
            new_timing.camera_frame(ns)
        for ns in camera:
            new_timing.written(picamera2_pts(ns, camera[0]))
        new_timing.finish()
        result = new_timing.result(12)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(new_timing.timestamps[0], (camera[0] - 10_000_000_000) / 1_000_000)
        self.assertEqual(result["camera_frame_count"], 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
