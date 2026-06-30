"""Time synchronization manager for coordinating video and IMU capture.

Ported from grabette-capture/grabette_capture/sync.py.
"""

import time

# Linux clock id for CLOCK_BOOTTIME. Some Python builds don't expose the
# symbolic constant (e.g. it's absent on the dev machine), but clock_gettime
# accepts the raw id, so fall back to 7 (its value on Linux).
_CLOCK_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", 7)


class SyncManager:
    """Manages synchronized timestamps across capture streams.

    Uses time.monotonic() as the common clock reference. All timestamps
    are recorded relative to the start time, ensuring both video and IMU
    streams share the same t=0 reference point.

    At start() it also captures a CLOCK_BOOTTIME reference taken back-to-back
    with the monotonic one. This lets hardware capture timestamps that are
    expressed in CLOCK_BOOTTIME (e.g. picamera2's ``SensorTimestamp``) be
    placed on the exact same t=0 timeline via ``boottime_ns_to_ms()`` — so
    the observation camera can be timestamped at its true capture instant
    instead of at frame-delivery time.
    """

    def __init__(self):
        self._start_time: float | None = None
        self._start_boottime: float | None = None

    @property
    def is_started(self) -> bool:
        return self._start_time is not None

    def start(self) -> None:
        # Read both clocks as close together as possible. CLOCK_BOOTTIME and
        # CLOCK_MONOTONIC differ only by time spent in system suspend, which is
        # zero during a recording session but generally non-zero in absolute
        # terms — so the offset must be captured, not assumed to be 0.
        self._start_time = time.monotonic()
        self._start_boottime = time.clock_gettime(_CLOCK_BOOTTIME)

    def get_timestamp_ms(self) -> float:
        if self._start_time is None:
            raise RuntimeError("SyncManager not started. Call start() first.")
        return (time.monotonic() - self._start_time) * 1000.0

    def monotonic_s_to_ms(self, monotonic_s: float) -> float:
        """Convert a CLOCK_MONOTONIC seconds stamp to ms relative to start.

        Counterpart of boottime_ns_to_ms for hardware capture timestamps that
        depthai already maps onto the host monotonic clock (pkt.getTimestamp()).
        Returns a value on the same t=0 timeline as get_timestamp_ms() — so the
        OAK can be dated at its capture instant instead of at queue-drain
        (delivery) time, matching the Arducam's sensor-capture timeline.
        """
        if self._start_time is None:
            raise RuntimeError("SyncManager not started. Call start() first.")
        return (monotonic_s - self._start_time) * 1000.0

    def boottime_ns_to_ms(self, boottime_ns: int) -> float:
        """Convert a CLOCK_BOOTTIME nanosecond stamp to ms relative to start.

        Returns a value on the same t=0 timeline as ``get_timestamp_ms()``,
        so a hardware sensor timestamp (CLOCK_BOOTTIME) lands on the shared
        host clock without the frame-delivery latency that a stamp taken in
        the callback would carry.
        """
        if self._start_boottime is None:
            raise RuntimeError("SyncManager not started. Call start() first.")
        return boottime_ns / 1e6 - self._start_boottime * 1000.0

    def reset(self) -> None:
        self._start_time = None
        self._start_boottime = None
