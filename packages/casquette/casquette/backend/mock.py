from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from casquette.backend.base import Backend
from casquette.config import settings
from casquette.models import CaptureStatus, IMUSample, SensorState
from casquette.output import write_imu_json

logger = logging.getLogger(__name__)

FPS = 50.0
IMU_HZ = 200


class MockBackend(Backend):
    def __init__(self) -> None:
        self._running = False
        self._start_time: float | None = None
        self._capturing = False
        self._capture_start: float | None = None
        self._capture_session_dir: Path | None = None
        self._capture_task: asyncio.Task | None = None
        self._frame_count = 0
        self._imu_sample_count = 0
        self._wall_clock_start: str | None = None

    async def start(self) -> None:
        self._running = True
        self._start_time = time.time()
        logger.info("MockBackend started")

    async def stop(self) -> None:
        if self._capturing:
            await self.stop_capture()
        self._running = False
        self._start_time = None
        logger.info("MockBackend stopped")

    def get_state(self) -> SensorState:
        now_ms = time.time() * 1000
        noise = lambda: random.gauss(0, 0.02)
        imu = IMUSample(
            timestamp_ms=now_ms,
            accel=(noise(), noise(), 9.81 + noise()),
            gyro=(noise(), noise(), noise()),
        )
        return SensorState(imu=imu, capture=self.get_capture_status())

    async def start_capture(self, session_dir: Path) -> None:
        if self._capturing:
            raise RuntimeError("Already capturing")
        self._capturing = True
        self._capture_start = time.time()
        self._capture_session_dir = session_dir
        self._frame_count = 0
        self._imu_sample_count = 0
        self._wall_clock_start = datetime.now(timezone.utc).isoformat()
        self._capture_task = asyncio.create_task(self._mock_capture_loop())
        logger.info("MockBackend capture started → %s", session_dir)

    async def _mock_capture_loop(self) -> None:
        try:
            while self._capturing:
                self._frame_count += int(FPS / 10)
                self._imu_sample_count += int(IMU_HZ / 10)
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    async def stop_capture(self) -> CaptureStatus:
        if not self._capturing:
            raise RuntimeError("Not capturing")
        self._capturing = False
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None

        status = self.get_capture_status()

        if self._capture_session_dir:
            self._write_mock_outputs(self._capture_session_dir, status)

        self._capture_start = None
        self._capture_session_dir = None
        self._frame_count = 0
        self._imu_sample_count = 0
        self._wall_clock_start = None
        logger.info("MockBackend capture stopped")
        return status

    def get_capture_status(self) -> CaptureStatus:
        duration = 0.0
        if self._capture_start:
            duration = time.time() - self._capture_start
        return CaptureStatus(
            is_capturing=self._capturing,
            session_id=self._capture_session_dir.name if self._capture_session_dir else None,
            duration_seconds=round(duration, 2),
            frame_count=self._frame_count,
            imu_sample_count=self._imu_sample_count,
        )

    @property
    def is_capturing(self) -> bool:
        return self._capturing

    def get_frame_jpeg(self) -> bytes | None:
        return self._generate_test_pattern()

    @staticmethod
    def _generate_test_pattern() -> bytes:
        import struct
        width, height = 160, 120
        colors = [
            (255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
            (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0),
        ]
        bar_width = width // len(colors)
        row_size = width * 3
        padding = (4 - row_size % 4) % 4
        padded_row_size = row_size + padding

        bmp_data = bytearray()
        file_size = 54 + padded_row_size * height
        bmp_data += b'BM'
        bmp_data += struct.pack('<I', file_size)
        bmp_data += struct.pack('<HH', 0, 0)
        bmp_data += struct.pack('<I', 54)
        bmp_data += struct.pack('<I', 40)
        bmp_data += struct.pack('<i', width)
        bmp_data += struct.pack('<i', -height)
        bmp_data += struct.pack('<HH', 1, 24)
        bmp_data += struct.pack('<I', 0)
        bmp_data += struct.pack('<I', padded_row_size * height)
        bmp_data += struct.pack('<ii', 2835, 2835)
        bmp_data += struct.pack('<II', 0, 0)
        for y in range(height):
            for x in range(width):
                color_idx = min(x // bar_width, len(colors) - 1)
                r, g, b = colors[color_idx]
                bmp_data += bytes([b, g, r])
            bmp_data += b'\x00' * padding
        return bytes(bmp_data)

    def _write_mock_outputs(self, session_dir: Path, status: CaptureStatus) -> None:
        n_samples = status.imu_sample_count or 100
        duration_ms = status.duration_seconds * 1000
        accel_samples = []
        gyro_samples = []
        for i in range(n_samples):
            t = (i / n_samples) * duration_ms
            accel_samples.append({"cts": t, "value": [0.0, 0.0, 9.81]})
            gyro_samples.append({"cts": t, "value": [0.0, 0.0, 0.0]})

        write_imu_json(accel_samples, gyro_samples, FPS, session_dir / "imu_data.json")

        (session_dir / "raw_video.mp4").write_bytes(b"MOCK_VIDEO")

        meta = {
            "duration_seconds": status.duration_seconds,
            "frame_count": status.frame_count,
            "imu_sample_count": status.imu_sample_count,
            "fps": FPS,
            "imu_hz": IMU_HZ,
            "backend": "mock",
            "device_id": settings.device_id,
            "wall_clock_start_utc": self._wall_clock_start,
        }
        sync_meta = self.get_sync_metadata()
        if sync_meta:
            meta["sync"] = sync_meta
        (session_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
