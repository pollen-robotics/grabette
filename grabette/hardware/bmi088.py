"""BMI088 IMU driver for I2C communication.

Pure Python implementation for Bosch BMI088 6-axis IMU (accelerometer + gyroscope).
The BMI088 has separate I2C addresses for accelerometer and gyroscope.

Ported from grabette-capture/grabette_capture/bmi088.py.

References:
- BMI088 Datasheet: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf
"""

import time
from struct import unpack

# Default I2C addresses (SD0 pin high)
DEFAULT_ACCEL_ADDR = 0x19
DEFAULT_GYRO_ADDR = 0x69

# Accelerometer registers
ACC_CHIP_ID = 0x00
ACC_DATA = 0x12
ACC_SENSORTIME_0 = 0x18
ACC_CONF = 0x40
ACC_RANGE = 0x41
ACC_PWR_CONF = 0x7C
ACC_PWR_CTRL = 0x7D
ACC_SOFTRESET = 0x7E

# Gyroscope registers
GYRO_CHIP_ID = 0x00
GYRO_DATA = 0x02
GYRO_RANGE = 0x0F
GYRO_BANDWIDTH = 0x10
GYRO_LPM1 = 0x11
GYRO_SOFTRESET = 0x14

# Accelerometer ODR values
ACC_ODR_12_5 = 0x05
ACC_ODR_25 = 0x06
ACC_ODR_50 = 0x07
ACC_ODR_100 = 0x08
ACC_ODR_200 = 0x09
ACC_ODR_400 = 0x0A
ACC_ODR_800 = 0x0B
ACC_ODR_1600 = 0x0C

# Accelerometer bandwidth
ACC_BWP_OSR4 = 0x00
ACC_BWP_OSR2 = 0x01
ACC_BWP_NORMAL = 0x02

# Accelerometer range values
ACC_RANGE_3G = 0x00
ACC_RANGE_6G = 0x01
ACC_RANGE_12G = 0x02
ACC_RANGE_24G = 0x03

ACC_SENSITIVITY = {
    ACC_RANGE_3G: 10920,
    ACC_RANGE_6G: 5460,
    ACC_RANGE_12G: 2730,
    ACC_RANGE_24G: 1365,
}

# Gyroscope range values
GYRO_RANGE_2000 = 0x00
GYRO_RANGE_1000 = 0x01
GYRO_RANGE_500 = 0x02
GYRO_RANGE_250 = 0x03
GYRO_RANGE_125 = 0x04

GYRO_SENSITIVITY = {
    GYRO_RANGE_2000: 16.384,
    GYRO_RANGE_1000: 32.768,
    GYRO_RANGE_500: 65.536,
    GYRO_RANGE_250: 131.072,
    GYRO_RANGE_125: 262.144,
}

# Gyroscope ODR/bandwidth combinations
GYRO_ODR_2000_BW_532 = 0x00
GYRO_ODR_2000_BW_230 = 0x01
GYRO_ODR_1000_BW_116 = 0x02
GYRO_ODR_400_BW_47 = 0x03
GYRO_ODR_200_BW_23 = 0x04
GYRO_ODR_100_BW_12 = 0x05
GYRO_ODR_200_BW_64 = 0x06
GYRO_ODR_100_BW_32 = 0x07

# Gyroscope power modes
GYRO_PM_NORMAL = 0x00
GYRO_PM_SUSPEND = 0x80
GYRO_PM_DEEP_SUSPEND = 0x20

# Sensortime resolution: 39.0625 us per LSB
SENSORTIME_RESOLUTION_US = 39.0625

GRAVITY = 9.80665


class BMI088:
    """BMI088 6-axis IMU driver.

    Units:
        - Accelerometer: m/s² (includes gravity)
        - Gyroscope: rad/s
        - Sensortime: microseconds
    """

    def __init__(
        self,
        i2c,
        accel_addr: int = DEFAULT_ACCEL_ADDR,
        gyro_addr: int = DEFAULT_GYRO_ADDR,
        accel_range: int = ACC_RANGE_6G,
        gyro_range: int = GYRO_RANGE_2000,
        accel_odr: int = ACC_ODR_200,
        gyro_odr: int = GYRO_ODR_200_BW_23,
    ):
        self._i2c = i2c
        self._accel_addr = accel_addr
        self._gyro_addr = gyro_addr
        self._accel_range = accel_range
        self._gyro_range = gyro_range
        self._accel_odr = accel_odr
        self._gyro_odr = gyro_odr

        self._accel_scale = GRAVITY / ACC_SENSITIVITY[accel_range]
        self._gyro_scale = (3.14159265359 / 180.0) / GYRO_SENSITIVITY[gyro_range]

    def _write_accel(self, reg: int, value: int) -> None:
        while not self._i2c.try_lock():
            pass
        try:
            self._i2c.writeto(self._accel_addr, bytes([reg, value]))
        finally:
            self._i2c.unlock()

    def _read_accel(self, reg: int, length: int) -> bytes:
        while not self._i2c.try_lock():
            pass
        try:
            result = bytearray(length)
            self._i2c.writeto_then_readfrom(self._accel_addr, bytes([reg]), result)
            return bytes(result)
        finally:
            self._i2c.unlock()

    def _write_gyro(self, reg: int, value: int) -> None:
        while not self._i2c.try_lock():
            pass
        try:
            self._i2c.writeto(self._gyro_addr, bytes([reg, value]))
        finally:
            self._i2c.unlock()

    def _read_gyro(self, reg: int, length: int) -> bytes:
        while not self._i2c.try_lock():
            pass
        try:
            result = bytearray(length)
            self._i2c.writeto_then_readfrom(self._gyro_addr, bytes([reg]), result)
            return bytes(result)
        finally:
            self._i2c.unlock()

    def init(self) -> None:
        """Initialize the BMI088 sensor."""
        # --- Initialize Accelerometer ---
        try:
            self._write_accel(ACC_SOFTRESET, 0xB6)
        except OSError:
            pass
        time.sleep(0.05)

        try:
            self._read_accel(ACC_CHIP_ID, 1)
        except OSError:
            pass
        time.sleep(0.01)

        self._write_accel(ACC_PWR_CTRL, 0x04)
        time.sleep(0.005)
        self._write_accel(ACC_PWR_CONF, 0x00)
        time.sleep(0.005)

        acc_conf = (ACC_BWP_NORMAL << 4) | self._accel_odr
        self._write_accel(ACC_CONF, acc_conf)
        time.sleep(0.001)
        self._write_accel(ACC_RANGE, self._accel_range)
        time.sleep(0.001)

        # --- Initialize Gyroscope ---
        try:
            self._write_gyro(GYRO_SOFTRESET, 0xB6)
        except OSError:
            pass
        time.sleep(0.05)

        try:
            self._read_gyro(GYRO_CHIP_ID, 1)
        except OSError:
            pass
        time.sleep(0.01)

        self._write_gyro(GYRO_RANGE, self._gyro_range)
        time.sleep(0.001)
        self._write_gyro(GYRO_BANDWIDTH, self._gyro_odr)
        time.sleep(0.001)
        self._write_gyro(GYRO_LPM1, GYRO_PM_NORMAL)
        time.sleep(0.03)

        # Verify chip IDs
        accel_id = self._read_accel(ACC_CHIP_ID, 1)[0]
        gyro_id = self._read_gyro(GYRO_CHIP_ID, 1)[0]
        if accel_id != 0x1E:
            raise RuntimeError(f"BMI088 accelerometer not found (got 0x{accel_id:02X}, expected 0x1E)")
        if gyro_id != 0x0F:
            raise RuntimeError(f"BMI088 gyroscope not found (got 0x{gyro_id:02X}, expected 0x0F)")

    def read_accel(self) -> tuple[float, float, float]:
        data = self._read_accel(ACC_DATA, 6)
        ax, ay, az = unpack('<hhh', data)
        return (ax * self._accel_scale, ay * self._accel_scale, az * self._accel_scale)

    def read_gyro(self) -> tuple[float, float, float]:
        data = self._read_gyro(GYRO_DATA, 6)
        gx, gy, gz = unpack('<hhh', data)
        return (gx * self._gyro_scale, gy * self._gyro_scale, gz * self._gyro_scale)

    def read_sensortime(self) -> int:
        data = self._read_accel(ACC_SENSORTIME_0, 3)
        return data[0] | (data[1] << 8) | (data[2] << 16)

    def read_sensortime_us(self) -> float:
        return self.read_sensortime() * SENSORTIME_RESOLUTION_US

    def read_accel_with_time(self) -> tuple[tuple[float, float, float], int]:
        data = self._read_accel(ACC_DATA, 9)
        ax, ay, az = unpack('<hhh', data[0:6])
        sensortime = data[6] | (data[7] << 8) | (data[8] << 16)
        return (
            (ax * self._accel_scale, ay * self._accel_scale, az * self._accel_scale),
            sensortime,
        )

    def read_all(self) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
        accel, sensortime = self.read_accel_with_time()
        gyro = self.read_gyro()
        return accel, gyro, sensortime
