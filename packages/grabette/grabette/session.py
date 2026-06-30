"""Session and episode management for capture data.

Sessions are named groups of episodes. Episodes are individual captures
(raw_video.mp4 + imu_data.json). The registry lives in sessions.json;
episode directories are flat under episodes/.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

logger = logging.getLogger(__name__)

UNASSIGNED_ID = "unassigned"


# ── Models ────────────────────────────────────────────────────────────

class EpisodeInfo(BaseModel):
    episode_id: str
    created_at: str
    duration_seconds: float = 0.0
    frame_count: int = 0
    imu_sample_count: int = 0
    angle_sample_count: int = 0
    has_video: bool = False
    has_imu: bool = False


class SessionInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: str
    episode_ids: list[str] = []
    episode_count: int = 0
    total_duration: float = 0.0


class SessionDetail(SessionInfo):
    episodes: list[EpisodeInfo] = []


# ── SessionManager ────────────────────────────────────────────────────

class SessionManager:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path.home() / "grabette-data"
        self.episodes_dir = self.data_dir / "episodes"
        self._registry_path = self.data_dir / "sessions.json"
        self._sessions: list[dict] = []
        self.active_session_id: str = UNASSIGNED_ID
        # Runtime capture session (not persisted)
        self._capture_session_active: bool = False
        self._capture_session_task_id: str | None = None
        self._capture_session_count: int = 0
        # Episode whose directory exists but which hasn't been filed into a
        # session yet — i.e. a capture in progress. (episode_id, session_id).
        # It's registered on stop, so a half-started/aborted capture never
        # shows up in the session or its count.
        self._pending_episode: tuple[str, str | None] | None = None

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self._load()
        self._migrate_legacy()
        self._ensure_unassigned()
        self._save()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if self._registry_path.exists():
            try:
                data = json.loads(self._registry_path.read_text())
                self._sessions = data.get("sessions", [])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt sessions.json, starting fresh")
                self._sessions = []
        else:
            self._sessions = []

    def _save(self) -> None:
        data = {"sessions": self._sessions}
        tmp = self._registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self._registry_path)

    # ── Migration ─────────────────────────────────────────────────────

    def _migrate_legacy(self) -> None:
        """Move old sessions/{id} dirs to episodes/{id} and register them."""
        legacy_dir = self.data_dir / "sessions"
        if not legacy_dir.is_dir():
            return

        migrated_ids = []
        for d in sorted(legacy_dir.iterdir()):
            if d.is_dir():
                dest = self.episodes_dir / d.name
                if not dest.exists():
                    shutil.move(str(d), str(dest))
                    migrated_ids.append(d.name)
                    logger.info("Migrated legacy session %s → episodes/", d.name)

        if migrated_ids:
            # Add migrated episodes to Unassigned
            unassigned = self._find_session(UNASSIGNED_ID)
            if unassigned is None:
                self._ensure_unassigned()
                unassigned = self._find_session(UNASSIGNED_ID)
            existing = set(unassigned["episode_ids"])
            for eid in migrated_ids:
                if eid not in existing:
                    unassigned["episode_ids"].append(eid)
            self._save()

        # Remove legacy dir if empty
        try:
            legacy_dir.rmdir()
            logger.info("Removed empty legacy sessions/ directory")
        except OSError:
            pass  # Not empty, leave it

    def _ensure_unassigned(self) -> None:
        if self._find_session(UNASSIGNED_ID) is None:
            self._sessions.insert(0, {
                "id": UNASSIGNED_ID,
                "name": "Unassigned",
                "description": "",
                "created_at": "20250101_000000",
                "episode_ids": [],
            })

    def _find_session(self, session_id: str) -> dict | None:
        for s in self._sessions:
            if s["id"] == session_id:
                return s
        return None

    # ── Episode operations ────────────────────────────────────────────

    def episode_dir(self, episode_id: str) -> Path:
        return self.episodes_dir / episode_id

    def create_episode(self, session_id: str | None = None) -> str:
        """Create a new episode directory for a capture about to start.

        The episode is NOT filed into a session yet — that happens in
        register_episode() once the capture finishes and its files are
        written, so an in-progress or aborted capture never appears in the
        session or its count. The intended target session is remembered and
        resolved at registration time.
        """
        episode_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ep_dir = self.episode_dir(episode_id)
        ep_dir.mkdir(parents=True, exist_ok=True)
        self._pending_episode = (episode_id, session_id)
        return episode_id

    def register_episode(self, episode_id: str | None = None) -> None:
        """File the just-finished episode into its session and bump the count.

        Called after stop_capture has written the episode's files. Resolves
        the target the same way create_episode used to: an active capture
        session's task wins, else the session_id passed to create_episode,
        else the active session, else Unassigned. Idempotent and a no-op if
        there's no pending episode.
        """
        pending = self._pending_episode
        self._pending_episode = None
        if pending is None:
            return
        eid, session_id = pending
        if episode_id is not None and episode_id != eid:
            eid = episode_id

        if self._capture_session_active and self._capture_session_task_id:
            target_id = self._capture_session_task_id
        else:
            target_id = session_id or self.active_session_id
        target = self._find_session(target_id) or self._find_session(UNASSIGNED_ID)
        if eid not in target["episode_ids"]:
            target["episode_ids"].append(eid)
        self.active_session_id = target["id"]
        if self._capture_session_active:
            self._capture_session_count += 1
        self._save()

    def discard_pending_episode(self) -> None:
        """Drop a pending (created-but-never-registered) episode, removing its
        directory. Used when a capture fails to start so no empty episode is
        left behind."""
        pending = self._pending_episode
        self._pending_episode = None
        if pending is None:
            return
        ep_dir = self.episode_dir(pending[0])
        if ep_dir.exists():
            shutil.rmtree(ep_dir, ignore_errors=True)

    # ── Capture session (runtime lock) ────────────────────────────────

    def start_capture_session(self, task_id: str) -> None:
        target = self._find_session(task_id)
        if target is None:
            raise FileNotFoundError(f"Session {task_id} not found")
        self._capture_session_active = True
        self._capture_session_task_id = task_id
        self._capture_session_count = 0
        self.active_session_id = task_id

    def stop_capture_session(self) -> None:
        self._capture_session_active = False
        self._capture_session_task_id = None
        self._capture_session_count = 0

    def get_capture_session_status(self) -> dict:
        task_name = None
        if self._capture_session_task_id:
            s = self._find_session(self._capture_session_task_id)
            if s:
                task_name = s.get("name")
        return {
            "active": self._capture_session_active,
            "task_id": self._capture_session_task_id,
            "task_name": task_name,
            "count": self._capture_session_count,
        }

    def get_episode(self, episode_id: str) -> EpisodeInfo:
        ep_dir = self.episode_dir(episode_id)
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode {episode_id} not found")
        return self._get_episode_info(episode_id)

    def delete_episode(self, episode_id: str) -> None:
        ep_dir = self.episode_dir(episode_id)
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode {episode_id} not found")

        # Remove from whichever session contains it
        for s in self._sessions:
            if episode_id in s["episode_ids"]:
                s["episode_ids"].remove(episode_id)
                break

        shutil.rmtree(ep_dir)
        self._save()

    def create_episode_archive(self, episode_id: str) -> Path:
        ep_dir = self.episode_dir(episode_id)
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode {episode_id} not found")
        archive_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(ep_dir, arcname=episode_id)
        return archive_path

    def create_episodes_zip(self, episode_ids: list[str]) -> Path:
        archive_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        with tarfile.open(archive_path, "w:gz") as tar:
            for episode_id in episode_ids:
                ep_dir = self.episode_dir(episode_id)
                if ep_dir.exists():
                    tar.add(ep_dir, arcname=episode_id)
        return archive_path

    def _get_episode_info(self, episode_id: str) -> EpisodeInfo:
        ep_dir = self.episode_dir(episode_id)
        video_path = ep_dir / "raw_video.mp4"
        imu_path = ep_dir / "imu_data.json"

        duration = 0.0
        frame_count = 0
        imu_sample_count = 0
        angle_sample_count = 0
        meta_path = ep_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            duration = meta.get("duration_seconds", 0.0)
            frame_count = meta.get("frame_count", 0)
            imu_sample_count = meta.get("imu_sample_count") or meta.get("oakd", {}).get("imu_samples", 0)
            angle_sample_count = meta.get("angle_sample_count", 0)

        return EpisodeInfo(
            episode_id=episode_id,
            created_at=episode_id,
            duration_seconds=duration,
            frame_count=frame_count,
            imu_sample_count=imu_sample_count,
            angle_sample_count=angle_sample_count,
            has_video=video_path.exists(),
            has_imu=imu_path.exists(),
        )

    # ── Session operations ────────────────────────────────────────────

    def create_session(self, name: str, description: str = "") -> str:
        session_id = uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._sessions.append({
            "id": session_id,
            "name": name,
            "description": description,
            "created_at": created_at,
            "episode_ids": [],
        })
        self._save()
        return session_id

    def get_session(self, session_id: str) -> SessionInfo:
        s = self._find_session(session_id)
        if s is None:
            raise FileNotFoundError(f"Session {session_id} not found")
        return self._to_session_info(s)

    def get_session_detail(self, session_id: str) -> SessionDetail:
        s = self._find_session(session_id)
        if s is None:
            raise FileNotFoundError(f"Session {session_id} not found")
        return self._to_session_detail(s)

    def update_session(self, session_id: str, name: str | None = None, description: str | None = None) -> SessionInfo:
        s = self._find_session(session_id)
        if s is None:
            raise FileNotFoundError(f"Session {session_id} not found")
        if session_id == UNASSIGNED_ID:
            raise ValueError("Cannot modify the Unassigned session")
        if name is not None:
            s["name"] = name
        if description is not None:
            s["description"] = description
        self._save()
        return self._to_session_info(s)

    def delete_session(self, session_id: str) -> None:
        if session_id == UNASSIGNED_ID:
            raise ValueError("Cannot delete the Unassigned session")
        s = self._find_session(session_id)
        if s is None:
            raise FileNotFoundError(f"Session {session_id} not found")

        # Move episodes back to Unassigned
        unassigned = self._find_session(UNASSIGNED_ID)
        for eid in s["episode_ids"]:
            if eid not in unassigned["episode_ids"]:
                unassigned["episode_ids"].append(eid)

        self._sessions.remove(s)
        self._save()

    def list_sessions(self) -> list[SessionDetail]:
        return [self._to_session_detail(s) for s in self._sessions]

    def move_episodes(self, episode_ids: list[str], target_session_id: str) -> None:
        target = self._find_session(target_session_id)
        if target is None:
            raise FileNotFoundError(f"Target session {target_session_id} not found")

        for eid in episode_ids:
            # Remove from current session
            for s in self._sessions:
                if eid in s["episode_ids"]:
                    s["episode_ids"].remove(eid)
                    break
            # Add to target
            if eid not in target["episode_ids"]:
                target["episode_ids"].append(eid)

        self._save()

    # ── Helpers ────────────────────────────────────────────────────────

    def _to_session_info(self, s: dict) -> SessionInfo:
        episodes = [
            self._get_episode_info(eid)
            for eid in s["episode_ids"]
            if self.episode_dir(eid).exists()
        ]
        return SessionInfo(
            id=s["id"],
            name=s["name"],
            description=s.get("description", ""),
            created_at=s["created_at"],
            episode_ids=s["episode_ids"],
            episode_count=len(episodes),
            total_duration=sum(e.duration_seconds for e in episodes),
        )

    def _to_session_detail(self, s: dict) -> SessionDetail:
        episodes = [
            self._get_episode_info(eid)
            for eid in s["episode_ids"]
            if self.episode_dir(eid).exists()
        ]
        return SessionDetail(
            id=s["id"],
            name=s["name"],
            description=s.get("description", ""),
            created_at=s["created_at"],
            episode_ids=s["episode_ids"],
            episode_count=len(episodes),
            total_duration=sum(e.duration_seconds for e in episodes),
            episodes=episodes,
        )
