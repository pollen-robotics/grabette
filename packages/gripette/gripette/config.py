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

    # Camera — defaults match the validated deployment (previously carried
    # per-device in /etc/gripette/env): video pipeline, binned half-res,
    # 30 fps stream.
    camera_resolution_w: int = 648
    camera_resolution_h: int = 486
    jpeg_quality: int = 70
    # picamera2 pipeline: "still" = full-res sensor readout per frame (slow,
    # ~10 Hz ceiling on a Pi Zero); "video" = continuous binned sensor mode
    # (same full FOV on the RPi cameras, much faster capture). If you switch
    # an already-trained deployment's mode or resolution, verify the streamed
    # image still matches training with ood_check.py before trusting evals.
    camera_mode: Literal["still", "video"] = "video"
    # StreamState target rate (frames/s). Actual rate is capped by what
    # capture+JPEG-encode achieves on the hardware (see camera_mode).
    stream_hz: float = 30.0
    # Explicit mock camera (generated placeholder frames) for dev machines
    # without picamera2/hardware. NEVER enabled implicitly: on the robot a
    # broken camera must fail the boot self-check, not stream fake images
    # the policy would silently act on.
    mock_camera: bool = False

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
    # 93.5 deg is the MEASURED collision angle (rgripette-v2, moved by hand with
    # the distal open). The previous 85 deg came from CAD and rejected the last
    # ~8.5 deg of real travel — precisely the range a firm close needs, since a
    # full close is meant to drive INTO the stop and let the torque cap stop it.
    motor1_max: float = math.radians(93.5)  # +1.6319 rad
    motor2_min: float = 0.0
    # Left at 116 deg deliberately. A torque-capped stall measured ~102 deg, but
    # that is a lower bound on the real travel, not the travel itself; tightening
    # to 102 could reject reachable commands. A loose bound is harmless here — it
    # is a safety envelope, NOT the normaliser. The projection normalises on the
    # measured travel instead (see grasp_projection.REACHABLE_DISTAL).
    motor2_max: float = math.radians(116)   # +2.0245 rad

    # Hard ceiling on motor effort, fraction 0..1 of max torque. Enforced
    # server-side on every command, so no client can reach 100%.
    #
    # This is what makes the raised motor1_max safe. The limits above are now the
    # real collision angles rather than a conservative CAD margin, so a full close
    # is MEANT to drive into the stop and be halted by torque. At 100% torque that
    # same command would grind into collision. Previously an unset per-command
    # limit meant "leave it alone", i.e. full torque on a fresh boot — the
    # protection depended on every client remembering to pass a limit.
    #
    # 0.5 leaves ample grip force (field grasps worked at 0.25) while keeping a
    # wide margin below stall.
    torque_ceiling: float = 0.5

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
