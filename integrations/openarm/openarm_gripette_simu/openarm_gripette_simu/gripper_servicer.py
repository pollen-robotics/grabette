"""GripperServicer — simulated Gripette gRPC service.

Mimics the real Gripette API (gripper.proto) using MuJoCo simulation.
Camera frames are read from a cache rendered by the main thread.
Motor commands control the proximal/distal joints.
"""

import logging
import os
import time
import threading
import cv2
import numpy as np

from .proto import gripper_pb2, gripper_pb2_grpc

logger = logging.getLogger(__name__)

STREAM_HZ = 30  # matches the real Grabette stream / training data FPS
STREAM_INTERVAL = 1.0 / STREAM_HZ

# Bridges the proximal-joint convention between the dataset and the arm
# scene. Both are now in the POSITIVE-closing convention (real gripette
# and grabette runtimes were flipped to `0 = open, positive = closing`),
# so this is a no-op (+1.0) by default and may eventually be removed.
#
# LEGACY-DATASET WARNING: datasets recorded BEFORE the convention flip
# captured `proximal` as NEGATIVE-on-close. A policy trained on those
# emits negative proximal goals and consumes negative proximal states, so
# when evaluating a legacy model, launch the server with
#     PROXIMAL_CMD_SIGN=-1 python -m openarm_gripette_simu ...
# (found the hard way 2026-07-12: the June sim models — e.g.
# diffusion_grabette_simu_release — closed proximal NEGATIVE; after the
# flip their closes pinned at the open-stop and obs read 0.0 forever.)
PROXIMAL_CMD_SIGN = float(os.environ.get("PROXIMAL_CMD_SIGN", "+1.0"))


class GripperServicer(gripper_pb2_grpc.GripperServiceServicer):

    def __init__(self, sim, server, lock: threading.Lock, start_time: float):
        self._sim = sim
        self._server = server  # SimulationServer, for get_camera_frame()
        self._lock = lock
        self._start_time = start_time

    def _get_motor_positions(self):
        """Report the proximal/distal gripper actual POSITION (qpos, rad), in
        the robot-frame convention (0 = open, positive = closing), with the
        PROXIMAL_CMD_SIGN bridge applied so legacy-dataset eval flips can be
        toggled without touching call sites.

        Matches training, which records the realized joint position
        (mocap_state_8d -> observation.state). With the compliant over-close,
        command and actual diverge by ~0.09 rad; we feed the actual position so
        eval observations match the recorded data. (If we switch the dataset back
        to recording the command, switch this to get_actuator_ctrl.)"""
        pos = self._sim.get_joint_positions(["proximal", "distal"])
        return float(PROXIMAL_CMD_SIGN * pos[0]), float(pos[1])

    def StreamState(self, request, context):
        logger.info("StreamState: client connected")
        sequence = 0
        next_time = time.monotonic()

        while context.is_active():
            with self._lock:
                pos1, pos2 = self._get_motor_positions()

            # Read cached camera frame (rendered in main thread, no GL conflict)
            img = self._server.get_camera_frame()

            # Encode as JPEG
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, jpeg_data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])

            frame = gripper_pb2.GripperFrame(
                jpeg_data=jpeg_data.tobytes(),
                motor_state=gripper_pb2.MotorState(
                    motor1_position=pos1,
                    motor2_position=pos2,
                ),
                timestamp_ms=(time.monotonic() - self._start_time) * 1000.0,
                sequence=sequence,
            )
            yield frame
            sequence += 1

            next_time += STREAM_INTERVAL
            sleep_dur = next_time - time.monotonic()
            if sleep_dur > 0:
                time.sleep(sleep_dur)

        logger.info("StreamState: client disconnected after %d frames", sequence)

    def SendMotorCommand(self, request, context):
        try:
            # During the post-reset grace period the server holds the gripper
            # open regardless of incoming commands. Without this, the eval
            # client's policy — which can't predict a closed→open transition
            # because the training data never showed one — keeps streaming
            # closed commands and the gripper snaps shut milliseconds after
            # a manual reset. The grace period ends after a fixed duration,
            # by which time the policy queue has rolled forward through
            # observations of the new home pose.
            hold_until = getattr(self._server, "_gripper_hold_open_until", 0.0)
            if time.monotonic() < hold_until:
                with self._lock:
                    self._sim.set_joint_commands(np.array([0.0, 0.0]),
                                                  ["proximal", "distal"])
                return gripper_pb2.MotorCommandResponse(success=True)
            with self._lock:
                self._sim.set_joint_commands(
                    np.array([PROXIMAL_CMD_SIGN * request.motor1_goal,
                              request.motor2_goal]),
                    ["proximal", "distal"],
                )
            return gripper_pb2.MotorCommandResponse(success=True)
        except Exception as e:
            logger.exception("Motor command failed")
            return gripper_pb2.MotorCommandResponse(success=False, error=str(e))

    def ReadMotors(self, request, context):
        with self._lock:
            pos1, pos2 = self._get_motor_positions()
        return gripper_pb2.MotorState(motor1_position=pos1, motor2_position=pos2)

    def SetTorque(self, request, context):
        return gripper_pb2.TorqueResponse(success=True)

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return gripper_pb2.PingResponse(status="ok", uptime_seconds=uptime)
