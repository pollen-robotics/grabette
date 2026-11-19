"""IMU capture + driver tests (hardware-free).

ImuCapture must keep timestamps monotonic across a hardware-counter wrap,
whatever the counter width: the BMI088's 24-bit SENSORTIME (39.0625 us/tick,
wraps ~655 s) and the LSM6DSV16X's 32-bit TIMESTAMP (~21.75 us/tick). Any
recording can start just before a wrap, so samples after it must not jump back
and get dropped by the cts >= 0 filter. Also covers the LSM6DSV16X register
math and ImuDriver conformance via fake I2C.
"""
import struct
import time

import pytest

from casquette.hardware.bmi088 import BMI088, SENSORTIME_RESOLUTION_US
from casquette.hardware.imu import ImuCapture, ImuDriver, create_imu_driver
from casquette.hardware.lsm6dsv16x import (
    LSM6DSV16X, TIMESTAMP_RESOLUTION_US, WHO_AM_I, OUTX_L_A, OUTX_L_G, TIMESTAMP0,
    FIFO_STATUS1, FIFO_DATA_OUT_TAG, FIFO_CTRL4, FIFO_TAG_ACCEL, FIFO_TAG_GYRO,
    FIFO_TAG_TIMESTAMP, _fifo_tag_ok,
)


def fifo_tag(tag, cnt=0):
    """Build a well-formed FIFO tag byte: [7:3]=sensor, [2:1]=cnt, [0]=even
    parity — as the LSM6DSV16X emits them (e.g. accel cnt0 -> 0x11)."""
    base = (tag << 3) | (cnt << 1)
    return base | (bin(base).count("1") & 1)
from casquette.hardware.sync import SyncManager


class FakeCounterDriver:
    """ImuDriver whose counter tracks real time and starts just before a wrap."""

    def __init__(self, counter_bits: int, res_us: float, ticks_before_wrap: int):
        self.COUNTER_BITS = counter_bits
        self.TIMESTAMP_RESOLUTION_US = res_us
        self._wrap = 1 << counter_bits
        self._t0 = time.monotonic()
        self._start = self._wrap - ticks_before_wrap

    def init(self) -> None:
        pass

    def read_timestamp(self) -> int:
        elapsed = int((time.monotonic() - self._t0) * 1e6 / self.TIMESTAMP_RESOLUTION_US)
        return (self._start + elapsed) % self._wrap

    def read_accel_with_time(self):
        return (0.0, 9.81, 0.0), self.read_timestamp()

    def read_gyro(self):
        return (0.0, 0.0, 0.0)

    def read_accel(self):
        return (0.0, 9.81, 0.0)

    def deinit(self) -> None:
        pass


@pytest.mark.parametrize("bits,res_us", [(24, SENSORTIME_RESOLUTION_US), (32, TIMESTAMP_RESOLUTION_US)])
def test_timestamps_monotonic_across_wrap(bits, res_us):
    sync = SyncManager()
    sync.start()
    # Start ~100 ms before the wrap; capture runs ~300 ms so it crosses it.
    driver = FakeCounterDriver(bits, res_us, ticks_before_wrap=int(0.1e6 / res_us))
    cap = ImuCapture(sync, driver=driver, sample_rate_hz=200)

    cap.start_capture()
    time.sleep(0.3)
    samples = cap.stop()

    cts = [s["cts"] for s in samples.accel]
    # 200 Hz for ~0.3 s; a bad unwrap would keep only the ~20 pre-wrap samples.
    assert len(cts) > 40
    assert all(b > a for a, b in zip(cts, cts[1:]))
    # Timestamps stay on the sync clock: the span matches wall time.
    assert 250 < cts[-1] - cts[0] < 400


class FakeI2C:
    """Minimal adafruit-style I2C returning canned register reads."""

    def __init__(self, regs: dict[int, bytes]):
        self.regs = regs
        self.writes: list[tuple[int, bytes]] = []

    def try_lock(self) -> bool:
        return True

    def unlock(self) -> None:
        pass

    def writeto(self, addr, data) -> None:
        self.writes.append((data[0], bytes(data[1:])))

    def writeto_then_readfrom(self, addr, out, inbuf) -> None:
        data = self.regs.get(out[0], bytes(len(inbuf)))
        inbuf[:] = data[: len(inbuf)].ljust(len(inbuf), b"\x00")

    def deinit(self) -> None:
        pass


def test_lsm6dsv16x_scaling_and_timestamp():
    regs = {
        WHO_AM_I: bytes([0x70]),
        OUTX_L_A: struct.pack("<hhh", 1000, 0, -2000),
        OUTX_L_G: struct.pack("<hhh", 100, -100, 0),
        TIMESTAMP0: bytes([0x10, 0x00, 0x00, 0x00]),
    }
    dev = LSM6DSV16X(FakeI2C(regs))
    dev.init()  # WHO_AM_I matches -> no raise

    ax, ay, az = dev.read_accel()
    assert ay == 0.0
    assert ax == pytest.approx(1000 * 0.244e-3 * 9.80665, rel=1e-6)   # 8 g
    assert az == pytest.approx(-2000 * 0.244e-3 * 9.80665, rel=1e-6)

    gx, gy, gz = dev.read_gyro()
    assert gx == pytest.approx(100 * 70e-3 * 3.14159265359 / 180.0, rel=1e-6)  # 2000 dps
    assert gz == 0.0

    assert dev.read_timestamp() == 0x10  # little-endian 32-bit assembly


class FakeFifoDriver:
    """FIFO driver whose entries carry exactly-ODR-spaced hardware timestamps,
    emitted for however much real time has elapsed since the last drain."""

    USES_FIFO = True
    COUNTER_BITS = 32
    TIMESTAMP_RESOLUTION_US = TIMESTAMP_RESOLUTION_US

    def __init__(self, hz=240):
        self._t0 = time.monotonic()
        self._n = 0
        self._period = int((1.0 / hz) * 1e6 / self.TIMESTAMP_RESOLUTION_US)
        self._hz = hz

    def init(self):
        pass

    def read_timestamp(self):
        return self._n * self._period

    def flush_fifo(self):
        self._n = 0
        self._t0 = time.monotonic()

    def read_fifo(self):
        target = int((time.monotonic() - self._t0) * self._hz)
        out = []
        while self._n < target:
            ts = self._n * self._period
            out.append(("gyro", (0.0, 0.0, 0.0), ts))
            out.append(("accel", (0.0, 9.81, 0.0), ts))
            self._n += 1
        return out

    def read_accel(self):
        return (0.0, 9.81, 0.0)

    def read_gyro(self):
        return (0.0, 0.0, 0.0)

    def deinit(self):
        pass


def test_fifo_capture_is_regular_and_full_rate():
    sync = SyncManager()
    sync.start()
    cap = ImuCapture(sync, driver=FakeFifoDriver(hz=240), sample_rate_hz=240)
    cap.start_capture()
    time.sleep(0.3)
    samples = cap.stop()

    cts = [s["cts"] for s in samples.accel]
    assert len(cts) > 50                                   # ~72 at 240 Hz over 0.3 s
    assert all(b > a for a, b in zip(cts, cts[1:]))        # monotonic
    assert len(samples.gyro) == len(samples.accel)         # paired streams
    dts = [b - a for a, b in zip(cts, cts[1:])]
    mean_dt = sum(dts) / len(dts)
    assert 3.8 < mean_dt < 4.6                             # ~4.17 ms at 240 Hz
    assert max(abs(d - mean_dt) for d in dts) < 0.5        # regular: tight spread


def test_lsm6dsv16x_wrong_whoami_raises():
    dev = LSM6DSV16X(FakeI2C({WHO_AM_I: bytes([0x6A])}))
    with pytest.raises(RuntimeError, match="WHO_AM_I"):
        dev.init()


def test_lsm6dsv16x_read_fifo_parses_burst():
    def entry(tag, data6):
        return bytes([fifo_tag(tag)]) + data6
    stream = (entry(FIFO_TAG_TIMESTAMP, bytes([0x10, 0, 0, 0, 0, 0]))   # ts = 16
              + entry(FIFO_TAG_ACCEL, struct.pack('<hhh', 1000, 0, -2000))
              + entry(FIFO_TAG_GYRO, struct.pack('<hhh', 100, -100, 0)))
    regs = {FIFO_STATUS1: bytes([3, 0]), FIFO_DATA_OUT_TAG: stream}  # 3 entries pending
    dev = LSM6DSV16X(FakeI2C(regs))

    ev = dev.read_fifo()
    assert [e[0] for e in ev] == ["accel", "gyro"]          # timestamp consumed, not emitted
    assert ev[0][2] == 16 and ev[1][2] == 16                # both tagged with ts=16
    assert ev[0][1][0] == pytest.approx(1000 * 0.244e-3 * 9.80665, rel=1e-6)
    assert ev[1][1][0] == pytest.approx(100 * 70e-3 * 3.14159265359 / 180.0, rel=1e-6)


def test_fifo_tag_ok_accepts_valid_rejects_corrupt():
    for tag in (FIFO_TAG_GYRO, FIFO_TAG_ACCEL, FIFO_TAG_TIMESTAMP):
        for cnt in range(4):
            assert _fifo_tag_ok(fifo_tag(tag, cnt))
    assert not _fifo_tag_ok(fifo_tag(FIFO_TAG_ACCEL) ^ 1)   # flipped parity bit
    assert not _fifo_tag_ok(3 << 3)                         # unknown sensor id (3)
    assert not _fifo_tag_ok(0x1F)                            # garbage data byte


def test_lsm6dsv16x_read_fifo_drops_corrupt_drain():
    # A burst that loses byte alignment: the middle entry's tag byte is a
    # misaligned data byte (unknown sensor + bad parity). The whole drain must
    # be discarded and the FIFO flushed to resync — no partial/mis-tagged data.
    good_ts = bytes([fifo_tag(FIFO_TAG_TIMESTAMP)]) + bytes([0x10, 0, 0, 0, 0, 0])
    corrupt = bytes([0x1F]) + bytes(6)
    good_acc = bytes([fifo_tag(FIFO_TAG_ACCEL)]) + struct.pack('<hhh', 1000, 0, -2000)
    regs = {FIFO_STATUS1: bytes([3, 0]), FIFO_DATA_OUT_TAG: good_ts + corrupt + good_acc}
    i2c = FakeI2C(regs)
    dev = LSM6DSV16X(i2c)

    assert dev.read_fifo() == []                            # whole drain dropped
    assert any(reg == FIFO_CTRL4 for reg, _ in i2c.writes)  # flush_fifo() resynced


def test_drivers_conform_to_protocol():
    assert isinstance(LSM6DSV16X(FakeI2C({})), ImuDriver)
    assert isinstance(BMI088(FakeI2C({})), ImuDriver)


def test_create_imu_driver_selects_sensor_and_odr():
    from casquette.hardware.lsm6dsv16x import ODR_120Hz, ODR_240Hz

    assert isinstance(create_imu_driver("lsm6dsv16x", FakeI2C({}), 240), LSM6DSV16X)
    assert create_imu_driver("lsm6dsv16x", FakeI2C({}), 240)._odr == ODR_240Hz
    assert create_imu_driver("lsm6dsv16x", FakeI2C({}), 100)._odr == ODR_120Hz
    assert isinstance(create_imu_driver("bmi088", FakeI2C({}), 200), BMI088)
    with pytest.raises(ValueError, match="unknown IMU sensor"):
        create_imu_driver("nope", FakeI2C({}), 200)


def test_settings_imu_candidates_default():
    from casquette.config import DEFAULT_IMU_CANDIDATES, Settings
    # The probe list lives in config now; its default matches the wired hardware.
    assert Settings().imu_candidates == DEFAULT_IMU_CANDIDATES
    assert ("bmi088", 1, {}) in Settings().imu_candidates


def test_create_imu_driver_bmi088_odr_and_kwargs():
    from casquette.hardware.bmi088 import ACC_ODR_200, ACC_ODR_400
    # A 240 Hz fleet target rounds to the nearest BMI088 step (200), not 400.
    assert create_imu_driver("bmi088", FakeI2C({}), 240)._accel_odr == ACC_ODR_200
    assert create_imu_driver("bmi088", FakeI2C({}), 400)._accel_odr == ACC_ODR_400
    # gyro_addr (0x68 on the strapped-low HAT) passes through to the driver.
    assert create_imu_driver("bmi088", FakeI2C({}), 240, gyro_addr=0x68)._gyro_addr == 0x68
