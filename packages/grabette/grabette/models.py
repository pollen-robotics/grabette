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


class CaptureStatus(BaseModel):
    is_capturing: bool = False
    session_id: str | None = None
    duration_seconds: float = 0.0
    frame_count: int = 0
    imu_sample_count: int = 0
    angle_sample_count: int = 0


class SensorState(BaseModel):
    imu: IMUSample | None = None
    angle: AngleSample | None = None
    capture: CaptureStatus = CaptureStatus()


class DaemonStatus(BaseModel):
    state: str
    backend: str
    error: str | None = None
    sensor: SensorState = SensorState()
