from __future__ import annotations

from pydantic import BaseModel


class IMUSample(BaseModel):
    timestamp_ms: float
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]


class AngleSample(BaseModel):
    timestamp_ms: float
    proximal: float  # radians
    distal: float  # radians


class TactileSample(BaseModel):
    timestamp_ms: float
    address: int  # Modbus device address of the sensor
    cells: list[list[int]]  # rows x cols grid of raw 12-bit ADC values (0-4095), row-major


class CaptureStatus(BaseModel):
    is_capturing: bool = False
    is_starting: bool = False
    episode_id: str | None = None  # id of the episode being / just captured
    duration_seconds: float = 0.0
    frame_count: int = 0
    imu_sample_count: int = 0
    angle_sample_count: int = 0
    tactile_sample_count: int = 0


class SensorState(BaseModel):
    imu: IMUSample | None = None
    angle: AngleSample | None = None
    tactile: list[TactileSample] | None = None
    capture: CaptureStatus = CaptureStatus()


class DaemonStatus(BaseModel):
    state: str
    backend: str
    error: str | None = None
    sensor: SensorState = SensorState()
