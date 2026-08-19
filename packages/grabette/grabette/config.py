"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
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

    # ------------------------------------------------------------------
    # Robot-frame angle convention (matches the gripette runtime):
    #   0 rad        = fingers fully open
    #   positive rad = closing
    # The AS5600L magnets rotate in directions determined by the physical
    # sensor mounting, so we apply a sign per channel to produce the
    # convention above. The sign per finger depends on which hand this
    # is (mirror builds); override individual signs only for asymmetric
    # hardware revisions.
    # ------------------------------------------------------------------
    hand: Literal["left", "right"] = "right"

    # Per-sensor sign for raw -> robot-frame mapping (+1 or -1). Derived
    # from `hand` in the model validator below unless explicitly set via
    # GRABETTE_PROXIMAL_SIGN / GRABETTE_DISTAL_SIGN.
    proximal_sign: int | None = None
    distal_sign: int | None = None

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
    relay_url: str = "https://pollen-robotics-grabette-fleet.hf.space"
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

    @model_validator(mode="after")
    def _derive_signs_from_hand(self):
        # V2 mechanical: the two AS5600L magnets rotate in OPPOSITE
        # directions when the fingers close (distal mounted upside-down
        # relative to proximal). So for a given hand the per-sensor signs
        # have OPPOSITE values. The right vs left build is the mirror
        # image, which negates both signs at once.
        #
        # Right hand: distal=+1, proximal=-1 → both go positive on close.
        # Left hand:  distal=-1, proximal=+1 → both go positive on close.
        # (The left configuration happens to match what the OLD code used
        # to call 'right' under the negative-closing convention. Renamed
        # consistently with the new positive=closing convention.)
        right_signs = {"distal": +1, "proximal": -1}
        left_signs = {"distal": -1, "proximal": +1}
        default = right_signs if self.hand == "right" else left_signs
        if self.distal_sign is None:
            self.distal_sign = default["distal"]
        if self.proximal_sign is None:
            self.proximal_sign = default["proximal"]
        return self


settings = Settings()
