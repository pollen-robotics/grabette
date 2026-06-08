"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import math

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "GRIPPER_"}

    # gRPC server
    host: str = "0.0.0.0"
    port: int = 50051

    # Camera
    camera_resolution_w: int = 1296
    camera_resolution_h: int = 972
    jpeg_quality: int = 70

    # Motors (Feetech STS3215 on serial bus)
    motor_port: str = "/dev/serial0"
    motor_baudrate: int = 1_000_000
    motor_id_1: int = 1
    motor_id_2: int = 2

    # Motor position limits (radians). Commands outside these are rejected.
    motor1_min: float = math.radians(-85)   # -1.4835 rad
    motor1_max: float = 0.0
    motor2_min: float = math.radians(-116)  # -2.0245 rad
    motor2_max: float = 0.0

    # Logging
    log_level: str = "INFO"


settings = Settings()
