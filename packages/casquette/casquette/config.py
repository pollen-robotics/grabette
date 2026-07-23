"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # extra="ignore" so that a stray / mis-prefixed entry in .env doesn't
    # kill startup. Pydantic-settings otherwise treats every .env key as a
    # candidate field and errors out on anything not declared here.
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

    # Device identification (for multi-device sync)
    device_id: str = ""

    # Peer list for /api/sync/{start,stop} fan-out. Comma-separated
    # device_id=url pairs, e.g.
    #     CASQUETTE_PEERS=grabette-1=http://rgrabette2.local:8000
    # On casquette use CASQUETTE_PEERS; on grabette use GRABETTE_PEERS.
    # Empty (default) → no peers, sync endpoints degrade to local-only.
    peers: str = ""

    # Logging
    log_level: str = "INFO"


settings = Settings()
