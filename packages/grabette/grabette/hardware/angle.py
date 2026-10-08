"""Angle sensor capture using AS5600L magnetic rotary position sensors over I2C.

V1 hardware used AS5600 (fixed address 0x36); V2 HAT uses AS5600L which has
the same register layout but defaults to address 0x40 and supports user-
programmable addresses (so multiple sensors can share one I2C bus). For now
we keep one sensor per bus and use the default 0x40.

Robot-frame output convention: 0 = fingers fully open, positive = closing.
The per-sensor sign that turns raw magnet rotation into this convention is
read from gripette.config.settings (proximal_sign / distal_sign, derived
from `hand`).
"""

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import settings
from .sync import SyncManager

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path.home() / ".grabette" / "angle_calibration.json"
# Reads averaged per sensor when calibrating, for a stable zero.
CALIBRATION_SAMPLES = 20


def load_calibration() -> dict | None:
    """The saved offsets, or None when this device was never calibrated.

    A file that cannot be read or lacks an offset counts as missing: zero
    offsets are not a calibration, they are raw magnet angles, and an episode
    recorded with them has a gripper channel that means nothing."""
    try:
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        for key in ("sensor_1_offset_deg", "sensor_2_offset_deg"):
            float(data[key])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return data


def save_calibration(offset_1_deg: float, offset_2_deg: float) -> dict:
    """Write the offsets read at the fully-open pose; returns what was saved."""
    calibration = {
        "sensor_1_offset_deg": round(offset_1_deg, 6),
        "sensor_2_offset_deg": round(offset_2_deg, 6),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(calibration, f, indent=2)
    tmp.replace(CALIBRATION_FILE)
    return calibration


def read_raw_averaged(read_deg, n: int = CALIBRATION_SAMPLES) -> float:
    """Average N reads of `read_deg()` (degrees), as a circular mean so a
    magnet sitting on the 360/0 wrap does not average to 180."""
    sin_sum = 0.0
    cos_sum = 0.0
    for _ in range(n):
        rad = math.radians(read_deg())
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)
        time.sleep(0.01)
    return math.degrees(math.atan2(sin_sum / n, cos_sum / n)) % 360.0


@dataclass
class AngleSamples:
    """Collected angle samples from capture session."""
    samples: list[dict] = field(default_factory=list)


class AngleCapture:
    """Captures angle data from two AS5600L magnetic rotary position sensors.

    Each AS5600L is on a separate I2C bus (both at default address 0x40).

    V2 hardware (rgbd branch): hardware I2C peripherals on the BCM2711.
        - Bus 1 (distal):   /dev/i2c-3 (GPIO 4/5),  dtoverlay=i2c3,pins_4_5
        - Bus 2 (proximal): /dev/i2c-4 (GPIO 8/9),  dtoverlay=i2c4,pins_8_9

    Register layout matches the original AS5600 (RAW ANGLE at 0x0C-0x0D,
    ANGLE at 0x0E-0x0F).
    """

    DEFAULT_SAMPLE_RATE_HZ = 100
    AS5600_ADDRESS = 0x40  # AS5600L default; AS5600 (non-L) was 0x36
    ANGLE_REGISTER = 0x0C
    # A sensor with no successful read for this long is reported unplugged.
    # Several 50 Hz idle reads (or 100 Hz capture reads): one glitch is not a
    # disconnection, a second of silence is.
    STALE_S = 1.0
    # Per-sensor signs are read from settings.distal_sign / proximal_sign,
    # derived from settings.hand. See gripette/grabette/config.py for the
    # right/left → sign mapping.

    def __init__(
        self,
        sync_manager: SyncManager,
        i2c_bus_1: int = 3,
        i2c_bus_2: int = 4,
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
        # time.monotonic() of the last successful read, distal then proximal —
        # the only proof a sensor is there: opening its bus succeeds without one.
        self._last_ok = [0.0, 0.0]
        self._load_calibration()

    def _load_calibration(self) -> None:
        data = load_calibration()
        if data is not None:
            self._offset_1_deg = float(data["sensor_1_offset_deg"])
            self._offset_2_deg = float(data["sensor_2_offset_deg"])

    def calibrate(self) -> dict:
        """Take the current pose as zero (fingers fully open), save it and
        apply it at once — no restart needed. Blocking (~0.4 s of I2C reads):
        call it from an executor. Not while capturing."""
        if self._running:
            raise RuntimeError("Angle capture is running")
        if self._i2c_1 is None or self._i2c_2 is None:
            raise RuntimeError("Sensors not initialized. Call init_sensors() first.")
        raw1 = read_raw_averaged(lambda: self._read_angle_raw(self._i2c_1))
        raw2 = read_raw_averaged(lambda: self._read_angle_raw(self._i2c_2))
        calibration = save_calibration(raw1, raw2)
        self._offset_1_deg = calibration["sensor_1_offset_deg"]
        self._offset_2_deg = calibration["sensor_2_offset_deg"]
        logger.info("Angle: calibrated, offsets: %.1f, %.1f",
                    self._offset_1_deg, self._offset_2_deg)
        return calibration

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
        # Opening /dev/i2c-N says nothing about the sensor behind it: read each
        # one once, so an unplugged sensor fails here and not silently later.
        for name, bus, i2c in (("distal", self.i2c_bus_1, self._i2c_1),
                               ("proximal", self.i2c_bus_2, self._i2c_2)):
            try:
                self._read_angle_raw(i2c)
            except OSError as e:
                self._close()
                raise OSError(e.errno, f"{name} sensor does not answer on "
                                       f"/dev/i2c-{bus} ({e.strerror or e})") from e
        logger.info("Angle sensors initialized at %d Hz", self.sample_rate_hz)

    def _read_angle_raw(self, i2c) -> float:
        result = bytearray(2)
        i2c.writeto_then_readfrom(self.AS5600_ADDRESS, bytes([self.ANGLE_REGISTER]), result)
        self._last_ok[0 if i2c is self._i2c_1 else 1] = time.monotonic()
        raw = ((result[0] & 0x0F) << 8) | result[1]
        return raw * 360.0 / 4096.0

    def unresponsive(self) -> list[str]:
        """The sensors that have not answered for STALE_S ("" = none)."""
        now = time.monotonic()
        return [name for name, t in zip(("distal", "proximal"), self._last_ok)
                if now - t > self.STALE_S]

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
                # i2c_bus_1 is the DISTAL sensor, _2 is PROXIMAL — see init.
                cal1 = self._normalize_angle(raw1 - self._offset_1_deg) * settings.distal_sign
                cal2 = self._normalize_angle(raw2 - self._offset_2_deg) * settings.proximal_sign

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

        self._close()
        return self._samples

    def _close(self) -> None:
        if self._i2c_1 is not None:
            self._i2c_1.deinit()
            self._i2c_1 = None
        if self._i2c_2 is not None:
            self._i2c_2.deinit()
            self._i2c_2 = None

    @property
    def sample_count(self) -> int:
        return len(self._samples.samples)
