"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import uuid
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


def _stable_device_id() -> str:
    """Return a stable per-device id, persisted across restarts."""
    path = Path.home() / ".cache" / "grabette" / "device_id"
    if path.exists():
        return path.read_text().strip()
    did = f"grabette-{uuid.uuid4().hex[:8]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(did)
    return did


class Settings(BaseSettings):
    model_config = {
        "env_prefix": "GRABETTE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

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

    # OAK-D SR — default OFF to save battery. Toggle from the UI to enable.
    enable_oakd: bool = False
    # After a capture that auto-enabled the OAK-D, keep it warm this many
    # seconds before powering down — lets back-to-back recordings start
    # instantly instead of paying the cold-boot warmup each time.
    oakd_keepalive_s: float = 30.0

    # UI
    ui_enabled: bool = True

    # Hardware button (Grove LED Button on GPIO22/23)
    button_enabled: bool = True

    # Logging
    log_level: str = "INFO"

    # Fleet relay
    relay_url: str = "https://glannuzel-grabette-fleet.hf.space"
    relay_enabled: bool = True
    device_id: str = ""
    device_name: str = ""

    @field_validator("device_id", mode="before")
    @classmethod
    def _resolve_device_id(cls, v: str) -> str:
        return v or _stable_device_id()

    @field_validator("device_name", mode="before")
    @classmethod
    def _resolve_device_name(cls, v: str) -> str:
        if v:
            return v
        import socket
        return socket.gethostname()


settings = Settings()
