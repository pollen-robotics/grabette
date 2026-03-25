"""Recording-aware gRPC server integrated with grabette capture sessions.

The server runs continuously. Frames received while no recording is active
are silently discarded. Call start_recording() / stop_recording() to control
the save window.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
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
        self._camera_dir: Path | None = None        # active write dir (RAM or persistent)
        self._camera_dir_final: Path | None = None  # persistent destination on SD card
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
        # Persistent destination on SD card (created now so it's ready for the move)
        camera_dir_final = session_dir / "grpc_camera_frames"
        camera_dir_final.mkdir(parents=True, exist_ok=True)

        # Write frames to RAM during recording to avoid I/O contention with the
        # H.264 encoder stream. Falls back to the persistent dir if /dev/shm is
        # unavailable (non-Linux environments).
        shm = Path("/dev/shm")
        if shm.is_dir():
            camera_dir = shm / f"grabette_{session_dir.name}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            logger.info("gRPC frames → RAM (%s), will move to %s on stop", camera_dir, camera_dir_final)
        else:
            camera_dir = camera_dir_final
            logger.info("gRPC frames → %s (/dev/shm unavailable)", camera_dir_final)

        with self._lock:
            self._camera_dir = camera_dir
            self._camera_dir_final = camera_dir_final
            self._r_hand_path = session_dir / "r_hand_traj.json"
            self._l_hand_path = session_dir / "l_hand_traj.json"
            self._total = 0
            self._r_hand_traj = []
            self._l_hand_traj = []
            self._recording = True
        logger.info("gRPC recording started → %s", session_dir)

    def stop_accepting(self) -> None:
        """Stop accepting new frames immediately, without doing any I/O.
        Call stop_recording() later (after RPI mux completes) for file operations."""
        with self._lock:
            self._recording = False

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False  # idempotent if stop_accepting() was called first
            self._flush_to_disk()
            total = self._total
        # Intentionally no file I/O here. Frames stay in /dev/shm until
        # mux_camera_frames_to_mp4() runs, so that the mux reads from RAM
        # and does not compete with the next recording's H.264 SD writes.
        logger.info("gRPC recording stopped (%d frames received)", total)

    def _move_frames_to_persistent(self, src: Path, dst: Path) -> None:
        """Move JPEG frames from RAM (/dev/shm) to persistent storage."""
        frames = sorted(src.glob("frame_*.jpg"))
        logger.info("gRPC: moving %d frames from RAM to persistent storage", len(frames))
        for f in frames:
            shutil.move(str(f), dst / f.name)
        try:
            src.rmdir()
        except OSError:
            pass

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

    def mux_camera_frames_to_mp4(self) -> None:
        """Assemble gRPC JPEG frames into an MP4 using ffmpeg.

        Reads frames from their current location (RAM if /dev/shm was used).
        Writes the MP4 directly to the persistent session directory.
        Only moves individual JPEG frames to persistent storage after the mux
        completes, so SD I/O does not overlap with an ongoing recording.
        """
        with self._lock:
            src_dir = self._camera_dir
            final_dir = self._camera_dir_final

        if src_dir is None or not src_dir.is_dir():
            return

        frames = sorted(src_dir.glob("frame_*.jpg"))
        if len(frames) < 2:
            logger.info("gRPC: not enough frames to create MP4 (%d)", len(frames))
            # Clean up empty RAM dir if needed
            if final_dir and src_dir != final_dir:
                try:
                    src_dir.rmdir()
                except OSError:
                    pass
            return

        # Parse timestamps from filenames: frame_{timestamp_ms:016d}_{n:06d}.jpg
        try:
            ts_list = [int(f.stem.split("_")[1]) for f in frames]
        except (IndexError, ValueError):
            logger.warning("gRPC: could not parse timestamps from frame filenames")
            return

        duration_ms = ts_list[-1] - ts_list[0]
        fps = (len(ts_list) - 1) / (duration_ms / 1000.0) if duration_ms > 0 else 30.0

        # Write concat list in the source dir (alongside the frames, for relative paths)
        concat_path = src_dir / "frames.txt"
        lines = []
        for i, f in enumerate(frames):
            lines.append(f"file '{f.name}'")
            duration_s = (ts_list[i + 1] - ts_list[i]) / 1000.0 if i + 1 < len(ts_list) else 1.0 / fps
            lines.append(f"duration {duration_s:.6f}")
        concat_path.write_text("\n".join(lines))

        # MP4 goes to the persistent session directory regardless of where frames are.
        # This means ffmpeg reads from RAM and writes one sequential file to SD —
        # much less disruptive to concurrent SD writes than reading many small files.
        output_dir = final_dir.parent if final_dir else src_dir.parent
        output_path = output_dir / "grpc_video.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-vf", "format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        concat_path.unlink(missing_ok=True)
        if result.returncode != 0:
            logger.warning("gRPC: ffmpeg failed: %s", result.stderr[-300:])
        else:
            logger.info("gRPC: video saved → %s (%.1f fps, %d frames)", output_path, fps, len(frames))

        # Move individual JPEG frames to persistent storage now that the mux is done.
        # Doing this after the mux ensures no concurrent SD read pressure during recording.
        if final_dir and src_dir != final_dir:
            self._move_frames_to_persistent(src_dir, final_dir)
            with self._lock:
                self._camera_dir = final_dir


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

        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
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

    def stop_accepting(self) -> None:
        """Stop the recording window immediately (no I/O). Call stop_recording() later."""
        if self._state is not None:
            self._state.stop_accepting()

    def stop_recording(self) -> None:
        if self._state is not None:
            self._state.stop_recording()
            threading.Thread(
                target=self._state.mux_camera_frames_to_mp4,
                daemon=True,
                name="grpc-mux",
            ).start()

    def get_status(self) -> dict:
        """Return gRPC server status and client activity."""
        if self._state is None:
            return {"enabled": False, "connected": False}
        return {"enabled": True, **self._state.get_activity()}
