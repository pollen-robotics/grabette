"""Session management for capture data."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    duration_seconds: float = 0.0
    frame_count: int = 0
    imu_sample_count: int = 0
    has_video: bool = False
    has_imu: bool = False


class SessionManager:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path.home() / "grabette-data" / "sessions"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.data_dir / session_id

    def create_session(self) -> str:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_id

    def list_sessions(self) -> list[SessionInfo]:
        sessions = []
        for d in sorted(self.data_dir.iterdir(), reverse=True):
            if d.is_dir():
                sessions.append(self._get_info(d.name))
        return sessions

    def get_session(self, session_id: str) -> SessionInfo:
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        return self._get_info(session_id)

    def delete_session(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        shutil.rmtree(session_dir)

    def create_archive(self, session_id: str) -> Path:
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        archive_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(session_dir, arcname=session_id)
        return archive_path

    def _get_info(self, session_id: str) -> SessionInfo:
        session_dir = self._session_dir(session_id)
        video_path = session_dir / "raw_video.mp4"
        imu_path = session_dir / "imu_data.json"

        # Read metadata if it exists
        meta_path = session_dir / "metadata.json"
        duration = 0.0
        frame_count = 0
        imu_sample_count = 0
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            duration = meta.get("duration_seconds", 0.0)
            frame_count = meta.get("frame_count", 0)
            imu_sample_count = meta.get("imu_sample_count", 0)

        return SessionInfo(
            session_id=session_id,
            created_at=session_id,  # ID is the timestamp
            duration_seconds=duration,
            frame_count=frame_count,
            imu_sample_count=imu_sample_count,
            has_video=video_path.exists(),
            has_imu=imu_path.exists(),
        )
