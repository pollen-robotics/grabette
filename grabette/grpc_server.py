"""Recording-aware gRPC server integrated with grabette capture sessions.

The server runs continuously. Frames received while no recording is active
are silently discarded. Call start_recording() / stop_recording() to control
the save window.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent import futures
from pathlib import Path

logger = logging.getLogger(__name__)

# HandSide enum values from frames.proto
_RIGHT = 0
_LEFT = 1


class _RecordingState:
    """Thread-safe recording state shared between the servicer and the server manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recording = False
        self._camera_dir: Path | None = None
        self._r_hand_path: Path | None = None
        self._l_hand_path: Path | None = None
        self._total = 0
        self._r_hand_traj: list = []
        self._l_hand_traj: list = []
        # Activity tracking (updated regardless of recording state)
        self._last_camera_ts: float = 0.0
        self._last_r_hand_ts: float = 0.0
        self._last_l_hand_ts: float = 0.0

    def start_recording(self, session_dir: Path) -> None:
        with self._lock:
            camera_dir = session_dir / "grpc_camera_frames"
            camera_dir.mkdir(parents=True, exist_ok=True)
            self._camera_dir = camera_dir
            self._r_hand_path = session_dir / "r_hand_traj.json"
            self._l_hand_path = session_dir / "l_hand_traj.json"
            self._total = 0
            self._r_hand_traj = []
            self._l_hand_traj = []
            self._recording = True
        logger.info("gRPC recording started → %s", session_dir)

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
            self._flush_to_disk()
            total = self._total
        logger.info("gRPC recording stopped (%d frames received)", total)

    def save_camera_frame(self, timestamp_ms: int, jpeg_data: bytes) -> bool:
        """Save JPEG frame to disk. Returns False if not recording."""
        with self._lock:
            self._last_camera_ts = time.time()
            if not self._recording:
                return False
            self._total += 1
            n = self._total
            camera_dir = self._camera_dir
        filename = f"frame_{timestamp_ms:016d}_{n:06d}.jpg"
        (camera_dir / filename).write_bytes(jpeg_data)
        return True

    def append_hand_entry(self, entry: dict, side: int) -> bool:
        """Append a hand pose entry in memory. Returns False if not recording."""
        with self._lock:
            now = time.time()
            if side == _RIGHT:
                self._last_r_hand_ts = now
            elif side == _LEFT:
                self._last_l_hand_ts = now
            if not self._recording:
                return False
            if side == _RIGHT:
                self._r_hand_traj.append(entry)
            elif side == _LEFT:
                self._l_hand_traj.append(entry)
        return True

    def append_hand_entries_bulk(self, r_entries: list, l_entries: list) -> bool:
        """Bulk-append hand entries in memory. Returns False if not recording."""
        with self._lock:
            now = time.time()
            if r_entries:
                self._last_r_hand_ts = now
            if l_entries:
                self._last_l_hand_ts = now
            if not self._recording:
                return False
            self._r_hand_traj.extend(r_entries)
            self._l_hand_traj.extend(l_entries)
        return True

    def _flush_to_disk(self) -> None:
        """Write in-memory traj lists to disk. Must be called with self._lock held."""
        if self._r_hand_traj and self._r_hand_path:
            self._r_hand_path.write_text(json.dumps(self._r_hand_traj, indent=2))
        if self._l_hand_traj and self._l_hand_path:
            self._l_hand_path.write_text(json.dumps(self._l_hand_traj, indent=2))

    def flush_to_disk(self) -> None:
        """Write in-memory traj to disk regardless of recording state."""
        with self._lock:
            self._flush_to_disk()



    def get_activity(self, stale_s: float = 3.0) -> dict:
        """Return connection and per-stream activity status."""
        now = time.time()
        with self._lock:
            cam_age = (now - self._last_camera_ts) if self._last_camera_ts else None
            r_age = (now - self._last_r_hand_ts) if self._last_r_hand_ts else None
            l_age = (now - self._last_l_hand_ts) if self._last_l_hand_ts else None

        def _info(age):
            if age is None:
                return {"active": False, "last_seen_s": None}
            return {"active": age < stale_s, "last_seen_s": round(age, 1)}

        connected = any(
            age is not None and age < stale_s
            for age in [cam_age, r_age, l_age]
        )
        return {
            "connected": connected,
            "camera": _info(cam_age),
            "r_hand": _info(r_age),
            "l_hand": _info(l_age),
        }



def _frame_to_hand_entry(frame) -> dict:
    return {
        "timestamp_ms": frame.timestamp_ms,
        "side": frame.side,
        "pose": list(frame.pose.data),
    }


class GrpcServer:
    """Manages the gRPC server lifecycle, wired to grabette recording sessions."""

    def __init__(self, host: str = "0.0.0.0", port: int = 50051) -> None:
        self._host = host
        self._port = port
        self._server = None
        self._state: _RecordingState | None = None

    def start(self) -> bool:
        """Start the gRPC server. Returns True on success, False if grpc is unavailable."""
        try:
            import grpc
            from grabette.grpc_api import frames_pb2, frames_pb2_grpc
        except ImportError as exc:
            logger.warning("gRPC server disabled (import error): %s", exc)
            return False

        state = _RecordingState()
        self._state = state

        class _Servicer(frames_pb2_grpc.GrabetteServiceServicer):
            def SendCameraFrame(self, frame, context):
                state.save_camera_frame(frame.timestamp_ms, frame.jpeg_data)
                return frames_pb2.FrameResponse(success=True)

            def StreamCameraFrame(self, request_iterator, context):
                count = 0
                for frame in request_iterator:
                    if state.save_camera_frame(frame.timestamp_ms, frame.jpeg_data):
                        count += 1
                return frames_pb2.FrameResponse(
                    success=True, message=f"Saved {count} frames"
                )

            def SendHandFrame(self, frame, context):
                state.append_hand_entry(_frame_to_hand_entry(frame), frame.side)
                return frames_pb2.FrameResponse(success=True)

            def StreamHandFrame(self, request_iterator, context):
                try:
                    for frame in request_iterator:
                        state.append_hand_entry(_frame_to_hand_entry(frame), frame.side)
                finally:
                    state.flush_to_disk()
                return frames_pb2.FrameResponse(success=True)

            def SendAllFrames(self, all_frames, context):
                state.save_camera_frame(
                    all_frames.camera_frame.timestamp_ms,
                    all_frames.camera_frame.jpeg_data,
                )
                state.append_hand_entries_bulk(
                    [_frame_to_hand_entry(all_frames.r_hand_frame)],
                    [_frame_to_hand_entry(all_frames.l_hand_frame)],
                )
                return frames_pb2.FrameResponse(success=True)

            def StreamAllFrames(self, request_iterator, context):
                try:
                    for af in request_iterator:
                        state.save_camera_frame(
                            af.camera_frame.timestamp_ms, af.camera_frame.jpeg_data
                        )
                        state.append_hand_entries_bulk(
                            [_frame_to_hand_entry(af.r_hand_frame)],
                            [_frame_to_hand_entry(af.l_hand_frame)],
                        )
                finally:
                    state.flush_to_disk()
                return frames_pb2.FrameResponse(success=True)

        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        frames_pb2_grpc.add_GrabetteServiceServicer_to_server(_Servicer(), self._server)
        self._server.add_insecure_port(f"{self._host}:{self._port}")
        self._server.start()
        logger.info("gRPC server listening on %s:%d", self._host, self._port)
        return True

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop(grace=2)
            self._server = None
        logger.info("gRPC server stopped")

    def start_recording(self, session_dir: Path) -> None:
        if self._state is not None:
            self._state.start_recording(session_dir)

    def stop_recording(self) -> None:
        if self._state is not None:
            self._state.stop_recording()

    def get_status(self) -> dict:
        """Return gRPC server status and client activity."""
        if self._state is None:
            return {"enabled": False, "connected": False}
        return {"enabled": True, **self._state.get_activity()}
