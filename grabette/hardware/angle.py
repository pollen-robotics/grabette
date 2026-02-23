"""Angle sensor capture using AS5600 magnetic rotary position sensors over I2C.

Ported from grabette-capture/grabette_capture/angle.py.
"""

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .sync import SyncManager

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path.home() / ".grabette" / "angle_calibration.json"


@dataclass
class AngleSamples:
    """Collected angle samples from capture session."""
    samples: list[dict] = field(default_factory=list)


class AngleCapture:
    """Captures angle data from two AS5600 magnetic rotary position sensors.

    Each AS5600 is on a separate I2C bus (they have the same fixed address 0x36).
    """

    DEFAULT_SAMPLE_RATE_HZ = 100
    AS5600_ADDRESS = 0x36
    ANGLE_REGISTER = 0x0C

    def __init__(
        self,
        sync_manager: SyncManager,
        i2c_bus_1: int = 4,
        i2c_bus_2: int = 5,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    ):
        self.sync = sync_manager
        self.i2c_bus_1 = i2c_bus_1
        self.i2c_bus_2 = i2c_bus_2
        self.sample_rate_hz = sample_rate_hz

        self._samples = AngleSamples()
        self._running = False
        self._thread: threading.Thread | None = None
        self._i2c_1 = None
        self._i2c_2 = None

        self._offset_1_deg = 0.0
        self._offset_2_deg = 0.0
        self._load_calibration()

    def _load_calibration(self) -> None:
        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE) as f:
                    data = json.load(f)
                self._offset_1_deg = data.get("sensor_1_offset_deg", 0.0)
                self._offset_2_deg = data.get("sensor_2_offset_deg", 0.0)
            except Exception:
                pass

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360
        return angle_deg

    def init_sensors(self) -> None:
        from adafruit_extended_bus import ExtendedI2C

        if self._offset_1_deg != 0.0 or self._offset_2_deg != 0.0:
            logger.info("Angle: calibration offsets: %.1f, %.1f",
                        self._offset_1_deg, self._offset_2_deg)

        self._i2c_1 = ExtendedI2C(self.i2c_bus_1)
        self._i2c_2 = ExtendedI2C(self.i2c_bus_2)
        logger.info("Angle sensors initialized at %d Hz", self.sample_rate_hz)

    def _read_angle_raw(self, i2c) -> float:
        result = bytearray(2)
        i2c.writeto_then_readfrom(self.AS5600_ADDRESS, bytes([self.ANGLE_REGISTER]), result)
        raw = ((result[0] & 0x0F) << 8) | result[1]
        return raw * 360.0 / 4096.0

    def _capture_loop(self) -> None:
        error_count = 0
        read_count = 0
        sample_interval = 1.0 / self.sample_rate_hz

        while self._running:
            loop_start = time.monotonic()
            read_count += 1

            try:
                ts = self.sync.get_timestamp_ms()
                raw1 = self._read_angle_raw(self._i2c_1)
                raw2 = self._read_angle_raw(self._i2c_2)
                cal1 = self._normalize_angle(raw1 - self._offset_1_deg)
                cal2 = self._normalize_angle(raw2 - self._offset_2_deg)

                self._samples.samples.append({
                    "cts": ts,
                    "value": [math.radians(cal1), math.radians(cal2)],
                })
            except Exception:
                error_count += 1

            elapsed = time.monotonic() - loop_start
            sleep_time = sample_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Angle: %d reads, %d errors", read_count, error_count)

    def start_capture(self) -> None:
        if self._running:
            raise RuntimeError("Angle capture already running")
        if self._i2c_1 is None or self._i2c_2 is None:
            raise RuntimeError("Sensors not initialized. Call init_sensors() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before angle capture")

        self._samples = AngleSamples()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> AngleSamples:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._i2c_1 is not None:
            self._i2c_1.deinit()
            self._i2c_1 = None
        if self._i2c_2 is not None:
            self._i2c_2.deinit()
            self._i2c_2 = None

        return self._samples

    @property
    def sample_count(self) -> int:
        return len(self._samples.samples)
