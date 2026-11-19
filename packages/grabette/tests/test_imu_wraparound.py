"""BMI088Capture must keep timestamps monotonic across a SENSORTIME wrap.

SENSORTIME is a 24-bit counter (39.0625 us/tick) that wraps every ~655 s at an
arbitrary point, so any recording can span a wrap. Before the fix, samples
after the wrap jumped back ~655 s and were dropped by the cts >= 0 filter.
"""
import time

from grabette.hardware.bmi088 import SENSORTIME_RESOLUTION_US
from grabette.hardware.imu import BMI088Capture
from grabette.hardware.sync import SyncManager

WRAP = 0x1000000


class FakeBMI088:
    """Counter that tracks real time and starts just before the wrap."""

    def __init__(self, ticks_before_wrap: int):
        self._t0 = time.monotonic()
        self._start = WRAP - ticks_before_wrap

    def read_sensortime(self) -> int:
        elapsed_ticks = int((time.monotonic() - self._t0) * 1e6 / SENSORTIME_RESOLUTION_US)
        return (self._start + elapsed_ticks) % WRAP

    def read_accel_with_time(self):
        return (0.0, 9.81, 0.0), self.read_sensortime()

    def read_gyro(self):
        return (0.0, 0.0, 0.0)


def test_timestamps_monotonic_across_wrap():
    sync = SyncManager()
    sync.start()
    cap = BMI088Capture(sync, sample_rate_hz=200)
    # ~100 ms before the wrap; capture runs ~300 ms so it crosses it.
    cap._bmi088 = FakeBMI088(ticks_before_wrap=int(0.1e6 / SENSORTIME_RESOLUTION_US))

    cap.start_capture()
    time.sleep(0.3)
    samples = cap.stop()

    cts = [s["cts"] for s in samples.accel]
    # 200 Hz for ~0.3 s; the old code kept only the ~20 pre-wrap samples.
    assert len(cts) > 40
    assert all(b > a for a, b in zip(cts, cts[1:]))
    # Timestamps stay on the sync clock: the span matches wall time.
    assert 250 < cts[-1] - cts[0] < 400
