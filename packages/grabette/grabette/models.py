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
    # Why a capture cannot start right now ("" = it can). Rides on the status the
    # dashboard already polls, so a device tied up by an upload reads as busy
    # instead of "Idle" — a device that looks free while it is not is how a
    # recording gets started on top of one.
    blocked_reason: str = ""
    # The gripper angle sensors were never calibrated. Also part of
    # blocked_reason; split out so the dashboard can offer the fix (the
    # "Calibrate my device" button) rather than only quote the refusal.
    needs_calibration: bool = False
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
