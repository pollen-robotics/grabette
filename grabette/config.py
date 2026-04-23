"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "GRABETTE_"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Backend
    backend: str = "auto"  # "auto", "mock", or "rpi"

    # Data
    data_dir: Path = Path.home() / "grabette-data"

    # Camera
    camera_fps: int = 46
    camera_resolution_w: int = 1296
    camera_resolution_h: int = 972

    # IMU
    imu_hz: int = 200

    # Angle sensors (AS5600 on I2C buses 4 & 5)
    angle_sensors: bool = True

    # OAK-D camera (rect_left + depth + IMU → session/oak/)
    oak_enabled: bool = False

    # UI
    ui_enabled: bool = True

    # Hardware button (Grove LED Button on GPIO22/23)
    button_enabled: bool = True

    # Logging
    log_level: str = "INFO"


settings = Settings()
