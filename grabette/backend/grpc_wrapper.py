"""GrpcBackend — wraps any Backend and hooks the gRPC server into capture lifecycle."""

from __future__ import annotations

from pathlib import Path

from grabette.backend.base import Backend
from grabette.models import CaptureStatus, SensorState


class GrpcBackend(Backend):
    """Decorator that adds gRPC recording hooks to any Backend.

    This ensures the gRPC server starts/stops saving data whenever grabette
    starts/stops a capture — regardless of the trigger (REST API or physical button).
    """

    def __init__(self, inner: Backend, grpc_server) -> None:
        self._inner = inner
        self._grpc = grpc_server

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()

    def get_state(self) -> SensorState:
        return self._inner.get_state()

    async def start_capture(self, session_dir: Path) -> None:
        await self._inner.start_capture(session_dir)
        self._grpc.start_recording(session_dir)

    async def stop_capture(self) -> CaptureStatus:
        # Stop accepting gRPC frames immediately — the recording window must match
        # the RPI capture window, not the longer post-processing window (ffmpeg mux
        # + hardware reinit) that follows.
        self._grpc.stop_accepting()
        # RPI stops: IMU/camera halt, ffmpeg mux, hardware reinit (~3-5s total).
        status = await self._inner.stop_capture()
        # Only now do the I/O-heavy work: move frames from RAM + start gRPC mux.
        # Running after the RPI mux avoids two concurrent ffmpeg processes competing
        # for SD card bandwidth.
        self._grpc.stop_recording()
        return status

    def get_capture_status(self) -> CaptureStatus:
        return self._inner.get_capture_status()

    @property
    def is_capturing(self) -> bool:
        return self._inner.is_capturing

    def get_frame_jpeg(self) -> bytes | None:
        return self._inner.get_frame_jpeg()
