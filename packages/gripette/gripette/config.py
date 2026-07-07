"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import model_validator
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

    # ------------------------------------------------------------------
    # Robot-frame convention (used by every API surface: gRPC, scripts,
    # client, limit checks):
    #   0 rad  = fully OPEN
    #   positive rad = CLOSING
    # The MotorController bridges robot frame <-> encoder frame using:
    #   read:  robot = (encoder - offset) * sign
    #   write: encoder = robot * sign + offset
    # ------------------------------------------------------------------

    # Which hand this gripette is built as. Determines the default
    # encoder-sign mapping (mirror-image mounting). Override individual
    # signs via GRIPPER_MOTOR1_SIGN / GRIPPER_MOTOR2_SIGN if needed for
    # an asymmetric hardware revision.
    hand: Literal["left", "right"] = "right"

    # Per-motor sign for robot <-> encoder mapping. Derived from `hand` in
    # the model validator below unless explicitly set. Values: +1 or -1.
    motor1_sign: int | None = None
    motor2_sign: int | None = None

    # Per-motor encoder offset (radians, encoder frame): the raw encoder
    # reading observed when the gripper is at robot-frame zero (fully
    # open). Written by scripts/calibrate_zero{_local,}.py. Defaults to 0
    # so an uncalibrated gripette still runs (just with a slight zero
    # error).
    motor1_offset: float = 0.0
    motor2_offset: float = 0.0

    # Motor position limits in ROBOT FRAME (radians; 0 = open, positive
    # = closing). Commands outside these are rejected.
    motor1_min: float = 0.0
    motor1_max: float = math.radians(85)    # +1.4835 rad
    motor2_min: float = 0.0
    motor2_max: float = math.radians(116)   # +2.0245 rad

    # Logging
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _derive_signs_from_hand(self):
        # +1 means "positive robot = positive encoder", i.e. the motor
        # rotates in the direction we call 'closing' on its own native
        # encoder axis. -1 is the mirror: positive robot needs a negative
        # encoder goal. The right/left split here is a CONVENTION based
        # on the v2 hardware; adjust the table if a future revision
        # changes the physical mounting.
        right_signs = (+1, +1)
        left_signs = (-1, -1)
        default = right_signs if self.hand == "right" else left_signs
        if self.motor1_sign is None:
            self.motor1_sign = default[0]
        if self.motor2_sign is None:
            self.motor2_sign = default[1]
        return self


settings = Settings()
