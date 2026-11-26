"""Hardware-agnostic IMU capture over I2C with drift-corrected timestamps.

The capture machinery (poll loop, SENSORTIME/TIMESTAMP unwrap, two-point
oscillator-drift correction against the sync clock) is identical across IMU
chips; only the register-level reads differ. So the chip sits behind the
``ImuDriver`` contract and ``ImuCapture`` drives whichever one is wired up.

Supported drivers: BMI088 (bmi088.py, 24-bit SENSORTIME) and LSM6DSV16X
(lsm6dsv16x.py, 32-bit TIMESTAMP). ``create_imu_driver`` builds either with the
ODR that matches a requested sample rate.

Originally a BMI088-only module (ported from
grabette-capture/grabette_capture/imu_bmi088.py); generalized when the grabette
HAT moved from the BMI088 to the LSM6DSV16X.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .sync import SyncManager

logger = logging.getLogger(__name__)


@runtime_checkable
class ImuDriver(Protocol):
    """What ``ImuCapture`` needs from an IMU chip.

    The driver reports a free-running hardware counter whose LSB is
    ``TIMESTAMP_RESOLUTION_US`` microseconds and which wraps at
    ``2 ** COUNTER_BITS``. ImuCapture unwraps it and rescales it onto the sync
    clock, so a driver only ever returns raw counter ticks.

    Accelerometer samples are m/s² (gravity included); gyroscope rad/s.
    """

    TIMESTAMP_RESOLUTION_US: float
    COUNTER_BITS: int
    # True if the chip has a hardware FIFO that ImuCapture should drain (regular,
    # ODR-clocked timestamps). FIFO drivers also provide read_fifo()/flush_fifo()
    # (duck-typed, called only when USES_FIFO is True). False drivers are polled.
    USES_FIFO: bool

    def init(self) -> None: ...
    def read_accel(self) -> tuple[float, float, float]: ...
    def read_gyro(self) -> tuple[float, float, float]: ...
    def read_accel_with_time(self) -> tuple[tuple[float, float, float], int]: ...
    def read_timestamp(self) -> int: ...
    def deinit(self) -> None: ...


def create_imu_driver(sensor: str, i2c, sample_rate_hz: int, **kwargs) -> ImuDriver:
    """Build the driver for ``sensor`` on an open I2C bus, with the ODR closest
    to ``sample_rate_hz``. Extra kwargs (addresses, full-scale) pass through."""
    sensor = sensor.lower()
    if sensor == "bmi088":
        from .bmi088 import (
            BMI088, ACC_ODR_100, ACC_ODR_200, ACC_ODR_400,
            GYRO_ODR_100_BW_32, GYRO_ODR_200_BW_23, GYRO_ODR_400_BW_47,
        )
        if sample_rate_hz <= 100:
            accel_odr, gyro_odr = ACC_ODR_100, GYRO_ODR_100_BW_32
        elif sample_rate_hz <= 300:
            # Nearest ODR: a 240 Hz fleet target (for the LSM6) is closer to the
            # BMI088's 200 Hz step than to 400, and 400 Hz polled over the shared
            # I2C bus returns empty accel data here — so round 101..300 to 200.
            accel_odr, gyro_odr = ACC_ODR_200, GYRO_ODR_200_BW_23
        else:
            accel_odr, gyro_odr = ACC_ODR_400, GYRO_ODR_400_BW_47
        return BMI088(i2c, accel_odr=accel_odr, gyro_odr=gyro_odr, **kwargs)

    if sensor == "lsm6dsv16x":
        from .lsm6dsv16x import LSM6DSV16X, ODR_120Hz, ODR_240Hz
        odr = ODR_120Hz if sample_rate_hz <= 120 else ODR_240Hz
        return LSM6DSV16X(i2c, odr=odr, **kwargs)

    raise ValueError(f"unknown IMU sensor {sensor!r} (expected 'bmi088' or 'lsm6dsv16x')")


@dataclass
class IMUSamples:
    """Collected IMU samples from a capture session."""
    accel: list[dict] = field(default_factory=list)
    gyro: list[dict] = field(default_factory=list)


def detect_imu_capture(
    sync_manager: SyncManager,
    candidates: list[tuple[str, int, dict]],
    sample_rate_hz: int,
) -> "ImuCapture | None":
    """Probe each (sensor, i2c_bus, kwargs) candidate in order and return an
    initialized ImuCapture for the first one that responds.

    Returns None when no IMU answers — the caller records without one. This
    never raises, so a device with old or no IMU hardware degrades gracefully
    and a mixed fleet (BMI088 vs LSM6DSV16X) needs no per-unit configuration.
    """
    for sensor, i2c_bus, kwargs in candidates:
        cap = ImuCapture(sync_manager, sensor=sensor, i2c_bus=i2c_bus,
                         sample_rate_hz=sample_rate_hz, **kwargs)
        try:
            cap.init_sensor()
        except Exception as e:
            logger.debug("No %s IMU on I2C bus %d: %s", sensor, i2c_bus, e)
            cap.close()
            continue
        logger.info("Detected %s IMU on I2C bus %d", sensor, i2c_bus)
        return cap
    logger.info("No IMU detected on any candidate bus")
    return None


class ImuCapture:
    """Captures IMU data from any ``ImuDriver`` with sync-clock timestamps.

    Construct either with a ready ``driver`` (dependency injection, used by
    tests), or with ``sensor`` + ``i2c_bus`` so ``init_sensor`` opens the bus
    and builds the driver via ``create_imu_driver``.

    Units: accelerometer m/s² (gravity included), gyroscope rad/s.
    """

    DEFAULT_SAMPLE_RATE_HZ = 200

    def __init__(
        self,
        sync_manager: SyncManager,
        driver: ImuDriver | None = None,
        sensor: str | None = None,
        i2c_bus: int | None = None,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        **driver_kwargs,
    ):
        if driver is None and (sensor is None or i2c_bus is None):
            raise ValueError("pass either a driver, or both sensor and i2c_bus")

        self.sync = sync_manager
        self.sample_rate_hz = sample_rate_hz
        self.sensor = sensor
        self.i2c_bus = i2c_bus
        self._driver_kwargs = driver_kwargs

        self.driver: ImuDriver | None = None
        self._i2c = None  # owned only when we open it ourselves
        self._res_us: float = 0.0
        self._mask: int = 0
        self._uses_fifo: bool = False
        if driver is not None:
            self._bind_driver(driver)

        self._samples = IMUSamples()
        self._running = False
        self._thread: threading.Thread | None = None
        self._offset_ms: float = 0

        # Two-point calibration for clock drift correction
        self._cal_start_sync_ms: float = 0
        self._cal_start_ticks: int = 0
        self._cal_end_sync_ms: float | None = None
        self._cal_end_ticks: int | None = None

        # Hardware counter wraps at 2**COUNTER_BITS at an arbitrary point, so
        # every raw reading goes through _unwrap() to stay monotonic.
        self._last_raw_ticks: int = 0
        self._unwrapped_ticks: int = 0

    def _bind_driver(self, driver: ImuDriver) -> None:
        self.driver = driver
        self._res_us = driver.TIMESTAMP_RESOLUTION_US
        self._mask = (1 << driver.COUNTER_BITS) - 1
        self._uses_fifo = getattr(driver, "USES_FIFO", False)

    def _unwrap(self, raw: int) -> int:
        """Extend a raw wrapping counter reading into a monotonic one.

        Valid as long as consecutive readings are less than one full counter
        period apart, which the poll loop guarantees.
        """
        self._unwrapped_ticks += (raw - self._last_raw_ticks) & self._mask
        self._last_raw_ticks = raw
        return self._unwrapped_ticks

    def init_sensor(self) -> None:
        """Open the I2C bus (if we own it), build the driver, and init it."""
        if self.driver is None:
            from adafruit_extended_bus import ExtendedI2C
            logger.info("Connecting to %s IMU on I2C bus %d", self.sensor, self.i2c_bus)
            self._i2c = ExtendedI2C(self.i2c_bus)
            self._bind_driver(
                create_imu_driver(self.sensor, self._i2c, self.sample_rate_hz,
                                  **self._driver_kwargs))
        self.driver.init()
        logger.info("IMU initialized at %d Hz", self.sample_rate_hz)

    def _capture_loop(self) -> None:
        if self._uses_fifo:
            self._fifo_capture_loop()
            return
        error_count = 0
        consecutive_errors = 0
        read_count = 0
        late_count = 0
        sample_interval = 1.0 / self.sample_rate_hz
        next_sample_time = time.monotonic()
        last_ticks = -1

        while self._running:
            read_count += 1
            current_time = time.monotonic()
            if current_time > next_sample_time + sample_interval:
                late_count += 1

            try:
                accel, ticks = self.driver.read_accel_with_time()
                gyro = self.driver.read_gyro()

                if consecutive_errors > 0:
                    if consecutive_errors >= 5:
                        logger.warning("IMU: recovered after %d consecutive I2C errors", consecutive_errors)
                    consecutive_errors = 0

                if ticks != last_ticks:
                    timestamp_ms = self._ticks_to_capture_ms(self._unwrap(ticks))
                    self._samples.accel.append({"cts": timestamp_ms, "value": list(accel)})
                    self._samples.gyro.append({"cts": timestamp_ms, "value": list(gyro)})
                    last_ticks = ticks
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
            self._cal_end_ticks = self._unwrap(self.driver.read_timestamp())
            self._cal_end_sync_ms = self.sync.get_timestamp_ms()
        except Exception:
            logger.warning("IMU: failed to read final calibration point for drift correction")

        error_pct = (error_count / read_count * 100) if read_count > 0 else 0
        logger.info("IMU: %d polls, %d errors (%.1f%%), %d late, %d accel, %d gyro samples",
                     read_count, error_count, error_pct, late_count,
                     len(self._samples.accel), len(self._samples.gyro))
        if error_pct > 1.0:
            logger.warning("IMU error rate %.1f%% — data may be unreliable", error_pct)

    def _fifo_capture_loop(self) -> None:
        """Drain the chip FIFO on a timer. Sample timestamps come from the
        hardware counter batched in the FIFO, so they are evenly ODR-spaced and
        regular even if a drain is late (bus contention) — and nothing is missed
        between drains. The drain cadence only bounds latency, not the rate."""
        error_count = 0
        consecutive_errors = 0
        drains = 0
        drain_interval = 0.02  # 20 ms; FIFO depth (512) buffers well past this
        next_drain = time.monotonic()

        while self._running:
            drains += 1
            try:
                for kind, val, raw in self.driver.read_fifo():
                    cts = self._ticks_to_capture_ms(self._unwrap(raw))
                    stream = self._samples.accel if kind == "accel" else self._samples.gyro
                    stream.append({"cts": cts, "value": list(val)})
                if consecutive_errors:
                    consecutive_errors = 0
            except Exception as e:
                error_count += 1
                consecutive_errors += 1
                if consecutive_errors == 1:
                    logger.debug("IMU FIFO read error: %s", e)
                elif consecutive_errors == 10:
                    logger.error("IMU: 10 consecutive FIFO errors — bus contention? %s", e)

            next_drain += drain_interval
            sleep_time = next_drain - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Final calibration point for drift correction (same counter as the FIFO).
        try:
            self._cal_end_ticks = self._unwrap(self.driver.read_timestamp())
            self._cal_end_sync_ms = self.sync.get_timestamp_ms()
        except Exception:
            logger.warning("IMU: failed to read final calibration point for drift correction")

        logger.info("IMU(FIFO): %d drains, %d errors, %d accel, %d gyro samples",
                     drains, error_count, len(self._samples.accel), len(self._samples.gyro))

    def _ticks_to_capture_ms(self, ticks: int) -> float:
        return (ticks * self._res_us) / 1000.0 + self._offset_ms

    def start_capture(self) -> None:
        if self._running:
            raise RuntimeError("IMU capture already running")
        if self.driver is None:
            raise RuntimeError("Sensor not initialized. Call init_sensor() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before IMU capture")

        self._samples = IMUSamples()

        # Clear any FIFO contents that accumulated since init, so the recording
        # starts from an empty buffer aligned with the offset computed below.
        if self._uses_fifo:
            self.driver.flush_fifo()

        current_ticks = self.driver.read_timestamp()
        current_capture_ms = self.sync.get_timestamp_ms()
        current_ticks_ms = (current_ticks * self._res_us) / 1000.0
        self._offset_ms = current_capture_ms - current_ticks_ms

        # The unwrapped counter starts at the raw reading, so the offset above
        # stays valid for unwrapped values.
        self._last_raw_ticks = current_ticks
        self._unwrapped_ticks = current_ticks

        # Store calibration start point for two-point drift correction
        self._cal_start_sync_ms = current_capture_ms
        self._cal_start_ticks = current_ticks
        self._cal_end_sync_ms = None
        self._cal_end_ticks = None

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> IMUSamples:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Correct oscillator drift using two-point linear rescaling.
        self._correct_clock_drift()

        self._samples.accel.sort(key=lambda s: s["cts"])
        self._samples.gyro.sort(key=lambda s: s["cts"])
        self._samples.accel = [s for s in self._samples.accel if s["cts"] >= 0]
        self._samples.gyro = [s for s in self._samples.gyro if s["cts"] >= 0]

        self.close()
        return self._samples

    def close(self) -> None:
        """Release the driver and the I2C bus (if we opened it). Safe to call
        when init only partially succeeded, and more than once."""
        if self.driver is not None:
            try:
                self.driver.deinit()
            except Exception:
                pass
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass
            self._i2c = None
        self.driver = None

    def _correct_clock_drift(self) -> None:
        """Correct IMU oscillator drift via two-point linear rescaling.

        An IMU's hardware counter runs on an internal oscillator with ~±1%
        tolerance, so a single start offset lets timestamps drift linearly off
        the monotonic sync clock. Measuring (counter, sync) at both capture
        start and stop lets us rescale every timestamp onto the sync clock,
        removing the drift — and absorbing any error in the nominal
        TIMESTAMP_RESOLUTION_US along the way.
        """
        if (self._cal_end_ticks is None or
                self._cal_end_sync_ms is None or
                len(self._samples.accel) < 2):
            return

        sync_span = self._cal_end_sync_ms - self._cal_start_sync_ms

        # Both points are on the unwrapped counter, so no wrap handling here.
        ticks_diff = self._cal_end_ticks - self._cal_start_ticks
        ticks_span_ms = ticks_diff * self._res_us / 1000.0

        if ticks_span_ms <= 0 or sync_span <= 0:
            return

        correction = sync_span / ticks_span_ms
        drift_pct = abs(1.0 - correction) * 100
        logger.info("IMU clock drift correction: factor=%.6f (%.3f%% oscillator drift)",
                     correction, drift_pct)

        # Rescale: true_cts = start + (cts - start) * correction
        start_ms = self._cal_start_sync_ms
        for s in self._samples.accel:
            s["cts"] = start_ms + (s["cts"] - start_ms) * correction
        for s in self._samples.gyro:
            s["cts"] = start_ms + (s["cts"] - start_ms) * correction

    def read_live(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """One-shot (accel, gyro) read for status display; None if no driver."""
        if self.driver is None:
            return None
        return self.driver.read_accel(), self.driver.read_gyro()

    @property
    def sample_count(self) -> tuple[int, int]:
        return len(self._samples.accel), len(self._samples.gyro)
