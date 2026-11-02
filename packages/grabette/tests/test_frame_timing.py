"""Focused contract tests for recorded-frame timestamp association."""

import unittest

from grabette.hardware.frame_timing import FrameTiming


class FakeSync:
    """Only original sensor timestamps may determine saved capture times."""

    def __init__(self, start_ns=1_000_000_000):
        self.start_ns = start_ns
        self.now_ns = start_ns

    def boottime_ns_to_ms(self, sensor_ns):
        return (sensor_ns - self.start_ns) / 1_000_000

    def get_timestamp_ms(self):
        raise AssertionError("Output-arrival time must not timestamp video frames")


class FrameTimingTests(unittest.TestCase):
    def setUp(self):
        self.sync = FakeSync()
        self.timing = FrameTiming(self.sync)

    def submit(self, pts_us, offset_ns):
        self.timing.submitted(pts_us, self.sync.start_ns + offset_ns)

    def test_missing_middle_output_keeps_exact_sensor_mapping(self):
        offsets = [10_000_123, 35_000_456, 60_000_789]
        for pts, offset in zip([0, 25_000, 50_000], offsets):
            self.submit(pts, offset)
        self.timing.written(0)
        self.timing.written(50_000)

        result = self.timing.result(2)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["unwritten_input_count"], 1)
        self.assertEqual(self.timing.timestamps, [10.000123, 60.000789])
        self.assertEqual(
            [frame["sensor_timestamp_ns"] for frame in self.timing.frames],
            [self.sync.start_ns + offsets[0], self.sync.start_ns + offsets[2]],
        )
        self.assertEqual(
            [frame["frame_index"] for frame in self.timing.frames], [0, 1]
        )

    def test_delayed_output_uses_capture_time_after_input_has_finished(self):
        self.submit(0, 10_000_123)
        self.submit(25_000, 35_000_456)
        self.timing.written(0)
        # The capture owner has stopped submitting requests; encoder draining
        # happens later and must not require any recording flag to remain set.
        self.sync.now_ns += 2_000_000_000
        self.timing.written(25_000)

        result = self.timing.result(2)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(self.timing.timestamps, [10.000123, 35.000456])

    def test_duplicate_pending_input_is_invalid_without_overwriting_timestamp(self):
        self.submit(0, 10_000_000)
        self.submit(0, 35_000_000)
        self.timing.written(0)

        self.assertFalse(self.timing.result(1)["valid"])
        self.assertEqual(self.timing.timestamps, [10.0])

    def test_duplicate_input_after_prior_output_is_still_invalid(self):
        self.submit(0, 10_000_000)
        self.timing.written(0)
        self.submit(0, 35_000_000)
        self.timing.written(0)
        self.assertFalse(self.timing.result(2)["valid"])

    def test_unknown_output_is_invalid_without_fabricated_timestamp(self):
        self.submit(0, 10_000_000)
        self.timing.written(0)
        self.timing.written(25_000)

        result = self.timing.result(2)
        self.assertFalse(result["valid"])
        self.assertEqual(result["written_count"], 2)
        self.assertEqual(result["timestamp_count"], 1)
        self.assertEqual(self.timing.timestamps, [10.0])

    def test_nonmonotonic_input_pts_is_invalid(self):
        self.submit(25_000, 10_000_000)
        self.submit(0, 35_000_000)
        self.timing.written(25_000)
        self.timing.written(0)
        self.assertFalse(self.timing.result(2)["valid"])

    def test_nonmonotonic_output_is_invalid(self):
        self.submit(0, 10_000_000)
        self.submit(25_000, 35_000_000)
        self.timing.written(25_000)
        self.timing.written(0)
        self.assertFalse(self.timing.result(2)["valid"])

    def test_nonincreasing_sensor_times_are_invalid(self):
        for second_offset in [10_000_000, 9_000_000]:
            with self.subTest(second_offset=second_offset):
                timing = FrameTiming(FakeSync())
                timing.submitted(0, 1_010_000_000)
                timing.submitted(25_000, 1_000_000_000 + second_offset)
                timing.written(0)
                timing.written(25_000)
                self.assertFalse(timing.result(2)["valid"])

    def test_video_frame_count_mismatch_is_invalid(self):
        self.submit(0, 10_000_000)
        self.timing.written(0)
        for decoded_count in [0, 2]:
            with self.subTest(decoded_count=decoded_count):
                self.assertFalse(self.timing.result(decoded_count)["valid"])
        self.assertTrue(self.timing.result(1)["valid"])

    def test_empty_recording_is_invalid(self):
        self.assertFalse(self.timing.result(0)["valid"])

    def test_recorded_output_error_invalidates_matching_frame_counts(self):
        self.submit(0, 10_000_000)
        self.timing.written(0)
        self.timing.record_error("injected output flush failure")
        result = self.timing.result(1)
        self.assertEqual(result["video_frame_count"], result["timestamp_count"])
        self.assertFalse(result["valid"])
        self.assertIn("injected output flush failure", result["errors"])

    def test_separate_sessions_reset_mapping_and_error_state(self):
        self.submit(0, 10_000_000)
        self.timing.written(25_000)
        self.assertFalse(self.timing.result(1)["valid"])

        new_sync = FakeSync(start_ns=10_000_000_000)
        new_timing = FrameTiming(new_sync)
        new_timing.submitted(0, 10_005_000_123)
        new_timing.written(0)
        result = new_timing.result(1)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(new_timing.timestamps, [5.000123])
        self.assertEqual(result["submitted_count"], 1)
        self.assertEqual(result["unwritten_input_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
