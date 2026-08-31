"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import uuid
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from casquette.fleet import spaces


def _stable_device_id() -> str:
    """Return a stable per-device id, persisted across restarts.

    Mirrors grabette's scheme but namespaced to casquette so the two daemons
    on a shared host don't collide. The fleet identifies a device by this id.
    """
    path = Path.home() / ".cache" / "casquette" / "device_id"
    if path.exists():
        return path.read_text().strip()
    did = f"casquette-{uuid.uuid4().hex[:8]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(did)
    return did


class Settings(BaseSettings):
    # extra="ignore" so a stray/mis-prefixed .env key doesn't kill startup.
    model_config = {
        "env_prefix": "CASQUETTE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Server
    host: str = "0.0.0.0"
    port: int = 8001

    # Backend
    backend: str = "auto"  # "auto", "mock", or "rpi"

    # Data
    data_dir: Path = Path.home() / "casquette-data"

    # Camera
    camera_fps: int = 46
    camera_resolution_w: int = 1296
    camera_resolution_h: int = 972
    # Fixed exposure (microseconds). 8 ms is short enough to freeze
    # typical head + hand motion for ArUco detection without going so
    # dark that the auto-gain has to amplify noise into the image.
    # Set to 0 to fall back to libcamera's auto-exposure (longer
    # exposures in dim scenes — visibly more motion blur).
    camera_exposure_us: int = 8000

    # IMU
    imu_hz: int = 200
    imu_i2c_bus: int = 1  # Pi Zero 2W: hw bus 1

    # Device identification. Empty device_id → resolved to a stable persisted
    # id (~/.cache/casquette/device_id); empty device_name → hostname.
    device_id: str = ""
    device_name: str = ""

    # Fleet relay — cloud-orchestrated multi-device sync via the fleet Space.
    # The device connects OUTBOUND to relay_url, authenticates with its local
    # HF token, and polls for group start/stop commands. Empty token / disabled
    # / unreachable → the device just runs solo (fleet is best-effort).
    # Derived from CASQUETTE_FLEET_ENV (see casquette.fleet.spaces): pointing a
    # device at the test deployment is one named env var, not a URL copied around.
    # CASQUETTE_RELAY_URL still overrides, as usual for a Settings field.
    relay_url: str = Field(default_factory=lambda: spaces.space_url("fleet"))
    relay_enabled: bool = True

    # Logging
    log_level: str = "INFO"

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
