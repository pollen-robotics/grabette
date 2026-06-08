"""IMU capture using BMI088 over I2C with hardware timestamps.

Ported from grabette-capture/grabette_capture/imu_bmi088.py.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

from .bmi088 import (
    BMI088,
    ACC_ODR_200,
    ACC_RANGE_6G,
    GYRO_ODR_200_BW_23,
    GYRO_RANGE_2000,
    SENSORTIME_RESOLUTION_US,
    DEFAULT_ACCEL_ADDR,
    DEFAULT_GYRO_ADDR,
)
from .sync import SyncManager

logger = logging.getLogger(__name__)


@dataclass
class IMUSamples:
    """Collected IMU samples from a capture session."""
    accel: list[dict] = field(default_factory=list)
    gyro: list[dict] = field(default_factory=list)


class BMI088Capture:
    """Captures IMU data from BMI088 sensor with hardware timestamps.

    Units:
        - Accelerometer: m/s² (includes gravity, ~9.8 when stationary)
        - Gyroscope: rad/s
    """

    DEFAULT_SAMPLE_RATE_HZ = 200
    DEFAULT_I2C_BUS = 3

    def __init__(
        self,
        sync_manager: SyncManager,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        i2c_bus: int = DEFAULT_I2C_BUS,
        accel_addr: int = DEFAULT_ACCEL_ADDR,
        gyro_addr: int = DEFAULT_GYRO_ADDR,
    ):
        self.sync = sync_manager
        self.sample_rate_hz = sample_rate_hz
        self.i2c_bus = i2c_bus
        self.accel_addr = accel_addr
        self.gyro_addr = gyro_addr

        self._samples = IMUSamples()
        self._running = False
        self._thread: threading.Thread | None = None
        self._bmi088: BMI088 | None = None
        self._i2c = None
        self._sensortime_offset_ms: float = 0

        # Two-point calibration for clock drift correction
        self._cal_start_sync_ms: float = 0
        self._cal_start_sensortime: int = 0
        self._cal_end_sync_ms: float | None = None
        self._cal_end_sensortime: int | None = None

    def init_sensor(self) -> None:
        """Initialize BMI088 sensor over I2C."""
        from adafruit_extended_bus import ExtendedI2C

        logger.info("Connecting to BMI088 on I2C bus %d (accel=0x%02X, gyro=0x%02X)",
                     self.i2c_bus, self.accel_addr, self.gyro_addr)

        self._i2c = ExtendedI2C(self.i2c_bus)

        # Select ODR based on sample rate
        from .bmi088 import ACC_ODR_100, GYRO_ODR_100_BW_32, ACC_ODR_400, GYRO_ODR_400_BW_47
        if self.sample_rate_hz <= 100:
            accel_odr, gyro_odr = ACC_ODR_100, GYRO_ODR_100_BW_32
        elif self.sample_rate_hz <= 200:
            accel_odr, gyro_odr = ACC_ODR_200, GYRO_ODR_200_BW_23
        else:
            accel_odr, gyro_odr = ACC_ODR_400, GYRO_ODR_400_BW_47

        self._bmi088 = BMI088(
            self._i2c,
            accel_addr=self.accel_addr,
            gyro_addr=self.gyro_addr,
            accel_range=ACC_RANGE_6G,
            gyro_range=GYRO_RANGE_2000,
            accel_odr=accel_odr,
            gyro_odr=gyro_odr,
        )
        self._bmi088.init()
        logger.info("BMI088 initialized at %d Hz", self.sample_rate_hz)

    def _capture_loop(self) -> None:
        error_count = 0
        consecutive_errors = 0
        read_count = 0
        late_count = 0
        sample_interval = 1.0 / self.sample_rate_hz
        next_sample_time = time.monotonic()
        last_sensortime = -1

        while self._running:
            read_count += 1
            current_time = time.monotonic()
            if current_time > next_sample_time + sample_interval:
                late_count += 1

            try:
                accel, sensortime = self._bmi088.read_accel_with_time()
                gyro = self._bmi088.read_gyro()

                if consecutive_errors > 0:
                    if consecutive_errors >= 5:
                        logger.warning("IMU: recovered after %d consecutive I2C errors", consecutive_errors)
                    consecutive_errors = 0

                if sensortime != last_sensortime:
                    timestamp_ms = self._sensortime_to_capture_ms(sensortime)
                    self._samples.accel.append({"cts": timestamp_ms, "value": list(accel)})
                    self._samples.gyro.append({"cts": timestamp_ms, "value": list(gyro)})
                    last_sensortime = sensortime
            except Exception as e:
                error_count += 1
                consecutive_errors += 1
                if consecutive_errors == 1:
                    logger.debug("IMU I2C error: %s", e)
                elif consecutive_errors == 10:
                    logger.error("IMU: 10 consecutive I2C errors — possible bus contention: %s", e)
                elif consecutive_errors == 100:
                    logger.error("IMU: 100 consecutive errors — sensor may be in bad state")

            next_sample_time += sample_interval
            sleep_time = next_sample_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Record final calibration point for drift correction.
        # Done inside the capture thread to avoid I2C bus contention.
        try:
            _, end_sensortime = self._bmi088.read_accel_with_time()
            self._cal_end_sensortime = end_sensortime
            self._cal_end_sync_ms = self.sync.get_timestamp_ms()
        except Exception:
            logger.warning("IMU: failed to read final calibration point for drift correction")

        error_pct = (error_count / read_count * 100) if read_count > 0 else 0
        logger.info("IMU: %d polls, %d errors (%.1f%%), %d late, %d accel, %d gyro samples",
                     read_count, error_count, error_pct, late_count,
                     len(self._samples.accel), len(self._samples.gyro))
        if error_pct > 1.0:
            logger.warning("IMU error rate %.1f%% — data may be unreliable", error_pct)

    def _sensortime_to_capture_ms(self, sensortime: int) -> float:
        sensortime_ms = (sensortime * SENSORTIME_RESOLUTION_US) / 1000.0
        return sensortime_ms + self._sensortime_offset_ms

    def start_capture(self) -> None:
        if self._running:
            raise RuntimeError("IMU capture already running")
        if self._bmi088 is None:
            raise RuntimeError("Sensor not initialized. Call init_sensor() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before IMU capture")

        self._samples = IMUSamples()

        current_sensortime = self._bmi088.read_sensortime()
        current_capture_ms = self.sync.get_timestamp_ms()
        current_sensortime_ms = (current_sensortime * SENSORTIME_RESOLUTION_US) / 1000.0
        self._sensortime_offset_ms = current_capture_ms - current_sensortime_ms

        # Store calibration start point for two-point drift correction
        self._cal_start_sync_ms = current_capture_ms
        self._cal_start_sensortime = current_sensortime
        self._cal_end_sync_ms = None
        self._cal_end_sensortime = None

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> IMUSamples:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Correct BMI088 oscillator drift using two-point linear rescaling.
        self._correct_clock_drift()

        self._samples.accel.sort(key=lambda s: s["cts"])
        self._samples.gyro.sort(key=lambda s: s["cts"])
        self._samples.accel = [s for s in self._samples.accel if s["cts"] >= 0]
        self._samples.gyro = [s for s in self._samples.gyro if s["cts"] >= 0]

        if self._i2c is not None:
            self._i2c.deinit()
            self._i2c = None
        self._bmi088 = None

        return self._samples

    def _correct_clock_drift(self) -> None:
        """Correct BMI088 oscillator drift via two-point linear rescaling.

        The BMI088 SENSORTIME register runs on an internal oscillator with
        ~±1% tolerance. A single-point offset (computed at start) causes
        timestamps to drift linearly from the true monotonic sync clock.

        By measuring (sensortime, sync_time) at both capture start and stop,
        we can linearly rescale all IMU timestamps to match the sync clock,
        eliminating oscillator drift entirely.
        """
        if (self._cal_end_sensortime is None or
                self._cal_end_sync_ms is None or
                len(self._samples.accel) < 2):
            return

        sync_span = self._cal_end_sync_ms - self._cal_start_sync_ms

        # Handle 24-bit sensortime wraparound (wraps every ~655s)
        sensortime_diff = self._cal_end_sensortime - self._cal_start_sensortime
        if sensortime_diff < 0:
            sensortime_diff += 0x1000000
        sensortime_span_ms = sensortime_diff * SENSORTIME_RESOLUTION_US / 1000.0

        if sensortime_span_ms <= 0 or sync_span <= 0:
            return

        correction = sync_span / sensortime_span_ms
        drift_pct = abs(1.0 - correction) * 100
        logger.info("IMU clock drift correction: factor=%.6f (%.3f%% oscillator drift)",
                     correction, drift_pct)

        # Rescale: true_cts = start + (cts - start) * correction
        start_ms = self._cal_start_sync_ms
        for s in self._samples.accel:
            s["cts"] = start_ms + (s["cts"] - start_ms) * correction
        for s in self._samples.gyro:
            s["cts"] = start_ms + (s["cts"] - start_ms) * correction

    @property
    def sample_count(self) -> tuple[int, int]:
        return len(self._samples.accel), len(self._samples.gyro)
