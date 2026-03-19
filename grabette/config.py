"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "GRABETTE_", "env_file": ".env"}

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
    imu_i2c_bus: int = 3  # Pi 4: hw bus 3, Pi Zero 2W: hw bus 1

    # Angle sensors (AS5600 on separate I2C buses, same address)
    angle_sensors: bool = True
    angle_i2c_bus_1: int = 4  # Pi 4: hw bus 4, Pi Zero 2W: sw bus 3
    angle_i2c_bus_2: int = 5  # Pi 4: hw bus 5, Pi Zero 2W: sw bus 4

    # UI
    ui_enabled: bool = True

    # Hardware button (Grove LED Button)
    button_enabled: bool = True
    button_led_pin: int = 22  # Pi 4: GPIO 22, Pi Zero 2W: GPIO 12 (PWM connector)
    button_pin: int = 23      # Pi 4: GPIO 23, Pi Zero 2W: GPIO 13 (PWM connector)

    # Logging
    log_level: str = "INFO"


settings = Settings()
