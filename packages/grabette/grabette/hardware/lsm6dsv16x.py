"""LSM6DSV16X IMU driver for I2C communication.

Pure Python driver for the STMicroelectronics LSM6DSV16X 6-axis IMU
(accelerometer + gyroscope), the sensor on the current grabette HAT. Unlike the
BMI088 it is a single I2C device (one address for both sub-sensors) and exposes
a 32-bit hardware TIMESTAMP counter rather than the BMI088's 24-bit SENSORTIME.

It conforms to the ImuDriver contract in imu.py (read_accel / read_gyro /
read_accel_with_time / read_timestamp + the TIMESTAMP_RESOLUTION_US and
COUNTER_BITS class attributes), so ImuCapture drives it unchanged.

References:
- LSM6DSV16X datasheet: https://www.st.com/resource/en/datasheet/lsm6dsv16x.pdf
- ST register map: https://github.com/STMicroelectronics/lsm6dsv16x-pid
"""

import time
from struct import unpack

# SA0 low on the HAT stack -> 0x6A. A breakout strapping SA0 high is 0x6B and
# can be selected via the address parameter.
DEFAULT_ADDR = 0x6A

WHO_AM_I = 0x0F
WHO_AM_I_VALUE = 0x70

# Control registers
CTRL1 = 0x10  # accel ODR [3:0] + operating mode [6:4]
CTRL2 = 0x11  # gyro  ODR [3:0] + operating mode [6:4]
CTRL3 = 0x12  # SW_RESET bit0, IF_INC bit2, BDU bit6, BOOT bit7
CTRL6 = 0x15  # gyro full-scale [3:0]
CTRL8 = 0x17  # accel full-scale [1:0]
FUNCTIONS_ENABLE = 0x50  # TIMESTAMP_EN bit6

# Data registers (IF_INC auto-increments across the 6 data bytes)
OUT_TEMP_L = 0x20
OUTX_L_G = 0x22  # gyro  X/Y/Z, 6 bytes
OUTX_L_A = 0x28  # accel X/Y/Z, 6 bytes
TIMESTAMP0 = 0x40  # 32-bit counter, 4 bytes little-endian

# FIFO. Hardware batches accel+gyro at the ODR with periodic timestamp entries,
# so draining it gives exactly-ODR-spaced samples with hardware timestamps —
# regular regardless of when (or how jittery) the I2C drain happens. Each FIFO
# entry is 7 bytes: a tag byte (sensor id in [7:3]) + 6 data bytes.
FIFO_CTRL1 = 0x07  # watermark threshold low (unused: we drain by count)
FIFO_CTRL2 = 0x08
FIFO_CTRL3 = 0x09  # BDR_GY [7:4] | BDR_XL [3:0]
FIFO_CTRL4 = 0x0A  # DEC_TS_BATCH [7:6] | FIFO_MODE [2:0]
FIFO_STATUS1 = 0x1B  # DIFF_FIFO (unread count) [7:0]
FIFO_STATUS2 = 0x1C  # [0] = DIFF_FIFO[8]; [5] overrun; [7] watermark
FIFO_DATA_OUT_TAG = 0x78  # tag byte; 0x79-0x7E hold the 6 data bytes

FIFO_MODE_STREAM = 0x06   # continuous (overwrite-oldest) mode
FIFO_TMSTMP_DEC_1 = 0x01  # a timestamp entry per batch slot
FIFO_TAG_GYRO = 0x01
FIFO_TAG_ACCEL = 0x02
FIFO_TAG_TIMESTAMP = 0x04
FIFO_MAX_ENTRIES = 512    # device FIFO depth; cap a single drain to this
FIFO_BURST_ENTRIES = 128  # entries per burst I2C read (128*7 = 896 bytes)

# CTRL3 bits
CTRL3_SW_RESET = 0x01
CTRL3_IF_INC = 0x04
CTRL3_BDU = 0x40

# Operating mode (high bits of CTRL1/CTRL2): 0x0 = high-performance.
XL_HIGH_PERFORMANCE_MD = 0x0
GY_HIGH_PERFORMANCE_MD = 0x0

# ODR codes (CTRL1/CTRL2 [3:0]). The grid has no 200 Hz; 240 is the closest.
ODR_120Hz = 0x06
ODR_240Hz = 0x07

# Accel full-scale (CTRL8 [1:0]) and mg/LSB sensitivity.
XL_FS_4g = 0x01
XL_FS_8g = 0x02
XL_SENSITIVITY_MG = {XL_FS_4g: 0.122, XL_FS_8g: 0.244}

# Gyro full-scale (CTRL6 [3:0]) and mdps/LSB sensitivity.
GY_FS_2000dps = 0x04
GY_SENSITIVITY_MDPS = {GY_FS_2000dps: 70.0}

# The 32-bit TIMESTAMP counter ticks at a nominal ~21.75 us/LSB. This only sets
# the initial counter->ms scale; ImuCapture's two-point drift correction then
# rescales every sample onto the sync clock, so small oscillator error here is
# removed and the exact nominal value is not critical.
TIMESTAMP_RESOLUTION_US = 21.75

GRAVITY = 9.80665
DEG_TO_RAD = 3.14159265359 / 180.0


def _fifo_tag_ok(tag_byte: int) -> bool:
    """True if a FIFO tag byte is well-formed.

    The tag byte is [7:3]=sensor id, [2:1]=2-bit sample counter, [0]=parity,
    and the whole byte carries EVEN parity (verified on hardware). A burst that
    loses I2C byte alignment — a BCM2835 clock-stretch glitch, or a marginal
    HAT connection — misreads a data byte as a tag: that fails even parity
    and/or carries a sensor id we never batch. Either way the entry, and the
    rest of the burst after it, cannot be trusted.
    """
    if bin(tag_byte).count("1") & 1:  # parity must be even
        return False
    return (tag_byte >> 3) in (FIFO_TAG_GYRO, FIFO_TAG_ACCEL, FIFO_TAG_TIMESTAMP)


class LSM6DSV16X:
    """LSM6DSV16X 6-axis IMU driver.

    Units:
        - Accelerometer: m/s² (includes gravity)
        - Gyroscope: rad/s
        - Timestamp: raw 32-bit counter ticks (TIMESTAMP_RESOLUTION_US per LSB)
    """

    # ImuDriver contract.
    TIMESTAMP_RESOLUTION_US = TIMESTAMP_RESOLUTION_US
    COUNTER_BITS = 32
    # Capture via the hardware FIFO (ImuCapture drains it) rather than polling
    # the data registers — the only way to get a regular rate on a shared bus.
    USES_FIFO = True

    def __init__(
        self,
        i2c,
        addr: int = DEFAULT_ADDR,
        accel_fs: int = XL_FS_8g,
        gyro_fs: int = GY_FS_2000dps,
        odr: int = ODR_240Hz,
    ):
        self._i2c = i2c
        self._addr = addr
        self._accel_fs = accel_fs
        self._gyro_fs = gyro_fs
        self._odr = odr
        # Most recent timestamp-tag value seen while draining the FIFO; carried
        # across drains and assigned to the accel/gyro entries that follow it.
        self._fifo_ts = 0

        # mg/LSB -> m/s² ; mdps/LSB -> rad/s
        self._accel_scale = XL_SENSITIVITY_MG[accel_fs] * 1e-3 * GRAVITY
        self._gyro_scale = GY_SENSITIVITY_MDPS[gyro_fs] * 1e-3 * DEG_TO_RAD

    def _write(self, reg: int, value: int) -> None:
        while not self._i2c.try_lock():
            pass
        try:
            self._i2c.writeto(self._addr, bytes([reg, value]))
        finally:
            self._i2c.unlock()

    def _read(self, reg: int, length: int) -> bytes:
        while not self._i2c.try_lock():
            pass
        try:
            result = bytearray(length)
            self._i2c.writeto_then_readfrom(self._addr, bytes([reg]), result)
            return bytes(result)
        finally:
            self._i2c.unlock()

    def init(self) -> None:
        """Reset, verify WHO_AM_I, and configure accel + gyro + timestamp."""
        who = self._read(WHO_AM_I, 1)[0]
        if who != WHO_AM_I_VALUE:
            raise RuntimeError(
                f"LSM6DSV16X not found (WHO_AM_I=0x{who:02X}, expected 0x{WHO_AM_I_VALUE:02X})")

        # Software reset, then re-assert BDU + register auto-increment.
        self._write(CTRL3, CTRL3_SW_RESET)
        time.sleep(0.05)
        self._write(CTRL3, CTRL3_BDU | CTRL3_IF_INC)
        time.sleep(0.005)

        # Full-scale ranges.
        self._write(CTRL8, self._accel_fs)
        self._write(CTRL6, self._gyro_fs)

        # ODR + high-performance mode for both sub-sensors.
        self._write(CTRL1, (XL_HIGH_PERFORMANCE_MD << 4) | self._odr)
        self._write(CTRL2, (GY_HIGH_PERFORMANCE_MD << 4) | self._odr)

        # Enable the hardware timestamp counter.
        self._write(FUNCTIONS_ENABLE, 0x40)

        # FIFO: batch accel + gyro at the ODR, plus periodic timestamp entries,
        # in continuous (stream) mode. ImuCapture drains this instead of polling.
        self._write(FIFO_CTRL3, (self._odr << 4) | self._odr)  # BDR_GY | BDR_XL
        self._write(FIFO_CTRL4, (FIFO_TMSTMP_DEC_1 << 6) | FIFO_MODE_STREAM)
        time.sleep(0.02)

    def flush_fifo(self) -> None:
        """Clear stale FIFO contents and re-anchor the timestamp reference.
        Called at capture start so the recording begins with a clean buffer."""
        self._write(FIFO_CTRL4, 0x00)  # bypass mode empties the FIFO
        self._write(FIFO_CTRL4, (FIFO_TMSTMP_DEC_1 << 6) | FIFO_MODE_STREAM)
        self._fifo_ts = self.read_timestamp()

    def _fifo_count(self) -> int:
        s = self._read(FIFO_STATUS1, 2)
        return s[0] | ((s[1] & 0x01) << 8)

    def read_fifo(self) -> list[tuple[str, tuple[float, float, float], int]]:
        """Drain pending FIFO entries. Returns an ordered list of
        (kind, (x,y,z), raw_ticks) where kind is 'accel' or 'gyro' and raw_ticks
        is the hardware timestamp counter value for that sample.

        Entries are read in bursts: a single multi-byte I2C read pops several
        7-byte entries, the device auto-wrapping the output address from
        FIFO_DATA_OUT_Z_H (0x7E) back to FIFO_DATA_OUT_TAG (0x78). Per-entry
        reads cannot keep up with the ~720 entries/s at 240 Hz and let the FIFO
        overflow; a burst drains the backlog in one or two transactions.

        Each entry's tag byte is integrity-checked (``_fifo_tag_ok``). If one
        fails, the burst has lost byte alignment (clock-stretch glitch or a
        marginal HAT connection), so the whole drain is discarded and the FIFO
        flushed to resync — we return nothing rather than emit mis-tagged,
        imbalanced or wrongly-timestamped samples. A healthy sensor never trips
        this; a marginal one loses that drain (~a few ms) and keeps the stream
        clean and monotonic instead of corrupting it.
        """
        n = min(self._fifo_count(), FIFO_MAX_ENTRIES)
        out: list[tuple[str, tuple[float, float, float], int]] = []
        done = 0
        while done < n:
            c = min(FIFO_BURST_ENTRIES, n - done)
            raw = self._read(FIFO_DATA_OUT_TAG, 7 * c)  # one transaction, c entries
            for k in range(c):
                off = k * 7
                tag_byte = raw[off]
                if not _fifo_tag_ok(tag_byte):
                    self.flush_fifo()  # lost alignment: drop drain + resync
                    return []
                tag = tag_byte >> 3
                d = raw[off + 1:off + 7]
                if tag == FIFO_TAG_TIMESTAMP:
                    self._fifo_ts = d[0] | (d[1] << 8) | (d[2] << 16) | (d[3] << 24)
                elif tag == FIFO_TAG_ACCEL:
                    x, y, z = unpack('<hhh', d)
                    out.append(("accel", (x * self._accel_scale, y * self._accel_scale,
                                          z * self._accel_scale), self._fifo_ts))
                elif tag == FIFO_TAG_GYRO:
                    x, y, z = unpack('<hhh', d)
                    out.append(("gyro", (x * self._gyro_scale, y * self._gyro_scale,
                                         z * self._gyro_scale), self._fifo_ts))
            done += c
        return out

    def read_accel(self) -> tuple[float, float, float]:
        ax, ay, az = unpack('<hhh', self._read(OUTX_L_A, 6))
        return (ax * self._accel_scale, ay * self._accel_scale, az * self._accel_scale)

    def read_gyro(self) -> tuple[float, float, float]:
        gx, gy, gz = unpack('<hhh', self._read(OUTX_L_G, 6))
        return (gx * self._gyro_scale, gy * self._gyro_scale, gz * self._gyro_scale)

    def read_timestamp(self) -> int:
        data = self._read(TIMESTAMP0, 4)
        return data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)

    def read_accel_with_time(self) -> tuple[tuple[float, float, float], int]:
        # Accel (0x28) and timestamp (0x40) are not contiguous, so two reads.
        return self.read_accel(), self.read_timestamp()

    def deinit(self) -> None:
        """No device-side teardown needed; the I2C bus is owned by the caller."""
