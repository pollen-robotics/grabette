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
    is_starting: bool = False
    episode_id: str | None = None  # id of the episode being / just captured
    duration_seconds: float = 0.0
    frame_count: int = 0
    imu_sample_count: int = 0
    angle_sample_count: int = 0
    # Why a capture cannot start right now ("" = it can). Rides on the status the
    # dashboard already polls, so a device tied up by an upload reads as busy
    # instead of "Idle" — a device that looks free while it is not is how a
    # recording gets started on top of one.
    blocked_reason: str = ""


class SensorState(BaseModel):
    imu: IMUSample | None = None
    angle: AngleSample | None = None
    capture: CaptureStatus = CaptureStatus()


class DaemonStatus(BaseModel):
    state: str
    backend: str
    error: str | None = None
    sensor: SensorState = SensorState()
