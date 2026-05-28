"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # extra="ignore" so that a stray / mis-prefixed entry in .env doesn't
    # kill startup. Pydantic-settings otherwise treats every .env key as a
    # candidate field and errors out on anything not declared here.
    model_config = {
        "env_prefix": "GRABETTE_",
        "env_file": ".env",
        "extra": "ignore",
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

    # Device identification (used as the 'self' entry in sync rig metadata).
    # Set via GRABETTE_DEVICE_ID env var, e.g. "grabette-1".
    device_id: str = ""

    # Multi-device sync
    # Peer list for /api/sync/{start,stop} fan-out. Format:
    #     GRABETTE_PEERS=casquette-1=http://casquette.local:8001,grabette-2=...
    # On grabette use GRABETTE_PEERS; on casquette use CASQUETTE_PEERS.
    # Empty (default) → no peers, sync endpoints degrade to local-only.
    peers: str = ""


settings = Settings()
