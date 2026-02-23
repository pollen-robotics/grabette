"""Time synchronization manager for coordinating video and IMU capture.

Ported from grabette-capture/grabette_capture/sync.py.
"""

import time


class SyncManager:
    """Manages synchronized timestamps across capture streams.

    Uses time.monotonic() as the common clock reference. All timestamps
    are recorded relative to the start time, ensuring both video and IMU
    streams share the same t=0 reference point.
    """

    def __init__(self):
        self._start_time: float | None = None

    @property
    def is_started(self) -> bool:
        return self._start_time is not None

    def start(self) -> None:
        self._start_time = time.monotonic()

    def get_timestamp_ms(self) -> float:
        if self._start_time is None:
            raise RuntimeError("SyncManager not started. Call start() first.")
        return (time.monotonic() - self._start_time) * 1000.0

    def reset(self) -> None:
        self._start_time = None
