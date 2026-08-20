"""Task, session and episode management for capture data.

Vocabulary (shared with grabette-fleet):
  * Task    — a type of action to perform. A named basket of episodes that
              realize that action. Persisted in tasks.json.
  * Session — a recording session: a run of several episodes recorded
              together for one task. On the device this is an ephemeral
              runtime lock (announce "upcoming episodes → task X" + count);
              the authoritative cross-device session record lives in the
              fleet. Not persisted here.
  * Episode — an individual capture (raw_video.mp4 + imu_data.json). Belongs
              to exactly one task, optionally recorded within a session.

Episode directories are flat under episodes/.
"""

from __future__ import annotations

import json

from grabette.hardware.episode_files import DCAM_IMU, metadata_stats, resolve
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

# How many loose episodes a single report describes. The inbox is meant to be
# exceptional, but a device recorded at the button for a week must not grow the
# register payload without bound — nor pay a metadata.json read per episode on
# every report (report_tasks deliberately reads none). The newest takes are the
# ones an operator triages; `total` is sent alongside so the UI can say
# "50 of 137" instead of quietly truncating.
UNASSIGNED_REPORT_LIMIT = 50


def local_identity() -> tuple[dict, list[str]]:
    """This device's own (members, signature), for a capture it made alone.

    A solo or offline recording gets no membership from the fleet, but the device
    needs none to describe itself: `hand` IS its slot (the fleet only derives the
    same value from the same field, see _device_slot there), and
    device_id/device_name are exactly what it registers under — so the result is
    byte-for-byte what the fleet would have sent for a one-member session.

    Without this, a solo episode reached the fleet with no roles at all: flagged
    "incomplete" in the task view and, worse, silently dropped from dataset
    generation (an episode with no roles is skipped). Stamping it here means the
    device answers that question at record time, from what it already knows,
    instead of depending on a later fill-devices repair that could only ever
    reconstruct this very information — and cannot run at all while offline.
    """
    from grabette.config import settings

    slot = settings.hand  # Literal["left","right"] → always a valid fleet slot
    return {slot: {"device_id": settings.device_id, "name": settings.device_name}}, [slot]


def episode_id_for(ts: datetime) -> str:
    """Canonical episode-id format, derived from a UTC instant.

    Exposed so a synchronized group start can derive the SAME episode id on
    every device from the shared T0, instead of each device stamping its own
    wall-clock time — which would drift apart by however long each device
    took to actually receive and process the start command.
    """
    return ts.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: str
    episode_ids: list[str] = []
    episode_count: int = 0
    total_duration: float = 0.0


class TaskDetail(TaskInfo):
    episodes: list[EpisodeInfo] = []


# ── TaskManager ────────────────────────────────────────────────────────

class TaskManager:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path.home() / "grabette-data"
        self.episodes_dir = self.data_dir / "episodes"
        self._registry_path = self.data_dir / "tasks.json"
        self._legacy_registry_path = self.data_dir / "sessions.json"
        self._tasks: list[dict] = []
        self.active_task_id: str = UNASSIGNED_ID
        # Runtime session (a recording run for a task) — not persisted.
        self._session_active: bool = False
        self._session_task_id: str | None = None
        self._session_count: int = 0
        # Episode whose directory exists but which hasn't been filed into a
        # task yet — i.e. a capture in progress.
        # (episode_id, task_id, members, signature). It's registered on stop, so
        # a half-started/aborted capture never shows up in the task or its count.
        # members/signature (from the fleet at start) let this device be the
        # durable source of truth for who recorded each episode.
        self._pending_episode: tuple[str, str | None, dict | None, list | None] | None = None
        # Bumped on every _save (i.e. every mutation). The relay watches it to
        # know when to re-report this device's tasks to the fleet.
        self._revision: int = 0

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self._load()
        self._migrate_legacy()
        self._ensure_unassigned()
        self._save()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        # Clean-cut rename sessions.json → tasks.json: adopt an existing
        # legacy registry once, then _save() persists it under the new name.
        path = self._registry_path if self._registry_path.exists() else self._legacy_registry_path
        if path.exists():
            try:
                data = json.loads(path.read_text())
                # Legacy files use the "sessions" key; new ones use "tasks".
                self._tasks = data.get("tasks", data.get("sessions", []))
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt task registry, starting fresh")
                self._tasks = []
        else:
            self._tasks = []

    def _save(self) -> None:
        # Every mutation goes through _save, so bumping here gives a cheap
        # change-signal: the relay reports tasks to the fleet only when this
        # revision moves (see RelayClient), instead of resending on every beat.
        self._revision += 1
        data = {"tasks": self._tasks}
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
                    logger.info("Migrated legacy capture %s → episodes/", d.name)

        if migrated_ids:
            # Add migrated episodes to Unassigned
            unassigned = self._find_task(UNASSIGNED_ID)
            if unassigned is None:
                self._ensure_unassigned()
                unassigned = self._find_task(UNASSIGNED_ID)
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
        if self._find_task(UNASSIGNED_ID) is None:
            self._tasks.insert(0, {
                "id": UNASSIGNED_ID,
                "name": "Unassigned",
                "description": "",
                "created_at": "20250101_000000",
                "episode_ids": [],
            })

    def _find_task(self, task_id: str) -> dict | None:
        for t in self._tasks:
            if t["id"] == task_id:
                return t
        return None

    # ── Episode operations ────────────────────────────────────────────

    def episode_dir(self, episode_id: str) -> Path:
        return self.episodes_dir / episode_id

    def create_episode(
        self,
        task_id: str | None = None,
        episode_id: str | None = None,
        members: dict | None = None,
        signature: list | None = None,
    ) -> str:
        """Create a new episode directory for a capture about to start.

        episode_id defaults to the current wall-clock time, but a caller
        doing a synchronized group start should pass episode_id_for(T0)
        explicitly — every device in the group then creates a directory with
        the SAME name, even though each one actually creates it whenever it
        happens to process the start command (which can differ by up to the
        fleet poll interval), because they all derive it from the same T0
        rather than from their own local creation time.

        The episode is NOT filed into a task yet — that happens in
        register_episode() once the capture finishes and its files are
        written, so an in-progress or aborted capture never appears in the
        task or its count. The intended target task is remembered and
        resolved at registration time.
        """
        episode_id = episode_id or episode_id_for(datetime.now(timezone.utc))
        ep_dir = self.episode_dir(episode_id)
        ep_dir.mkdir(parents=True, exist_ok=True)
        self._pending_episode = (episode_id, task_id, members, signature)
        return episode_id

    def register_episode(self, episode_id: str | None = None) -> None:
        """File the just-finished episode into its task and bump the count.

        Called after stop_capture has written the episode's files. Resolves
        the target: an active session's task wins, else the task_id passed to
        create_episode, else the active task, else Unassigned. Idempotent and
        a no-op if there's no pending episode.
        """
        pending = self._pending_episode
        self._pending_episode = None
        if pending is None:
            return
        eid, task_id, members, signature = pending
        if episode_id is not None and episode_id != eid:
            eid = episode_id

        # No membership from the fleet ⇒ this device recorded alone (button press
        # outside a session, or the fleet was unreachable). Describe the episode
        # with our own role rather than leaving it role-less. Kept separate from
        # `signature` below: a self-reported signature must never overwrite one
        # the fleet gave us.
        own_signature: list[str] | None = None
        if not members:
            members, own_signature = local_identity()

        if self._session_active and self._session_task_id:
            target_id = self._session_task_id
        else:
            target_id = task_id or self.active_task_id
        target = self._find_task(target_id) or self._find_task(UNASSIGNED_ID)
        if eid not in target["episode_ids"]:
            target["episode_ids"].append(eid)
        # Persist who recorded this episode (role → {device_id, name}) and the
        # task's device signature, captured from the fleet at start. This makes
        # the device the durable source of truth: relogging it under a new HF
        # account still surfaces the task and names its (possibly offline) peers.
        if signature:
            target["device_signature"] = list(signature)
        elif own_signature and target["id"] != UNASSIGNED_ID and not target.get("device_signature"):
            # Self-reported: fill a signature only where there is none, never
            # replace one. A solo press into an existing bimanual task must not
            # rewrite it to ["left"] — that would misdescribe every episode
            # already recorded there, and would bypass the fleet's own rule that
            # a signature is locked once the task has recordings. Unassigned is
            # excluded: it is a bucket holding episodes of any provenance, so a
            # signature there would be a claim about nothing (and it is never
            # reported to the fleet anyway).
            target["device_signature"] = list(own_signature)
        if members:
            target.setdefault("episode_members", {})[eid] = members
        # active_task_id is deliberately NOT moved onto the task we just filed
        # into: it is the operator's EXPLICIT local selection (PUT
        # /api/tasks/active, start_session), never a side effect of recording.
        # Making it sticky is what hid episodes: a fleet-driven session records
        # into its group task through this method, and closing that session on
        # the fleet never touches the device (the fleet dispatches start/stop
        # commands, not start_session/stop_session). The device therefore kept
        # pointing at that task, so the next physical-button press — which the
        # fleet answers "solo" — silently appended a new episode to it instead
        # of leaving it in Unassigned, where an untriaged recording belongs.
        if self._session_active:
            self._session_count += 1
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

    # ── Session (runtime recording-run lock) ──────────────────────────

    def start_session(self, task_id: str) -> None:
        target = self._find_task(task_id)
        if target is None:
            raise FileNotFoundError(f"Task {task_id} not found")
        self._session_active = True
        self._session_task_id = task_id
        self._session_count = 0
        self.active_task_id = task_id

    def stop_session(self) -> None:
        # Drop the selection with the lock: once the session is over, a physical
        # button press must record a solo episode into Unassigned — where it is
        # visibly untriaged — instead of quietly extending the task the session
        # was recording for. Only when a session was actually running, so a
        # stray /api/session/stop doesn't wipe an explicit local selection.
        if self._session_active:
            self.active_task_id = UNASSIGNED_ID
        self._session_active = False
        self._session_task_id = None
        self._session_count = 0

    def get_session_status(self) -> dict:
        task_name = None
        if self._session_task_id:
            t = self._find_task(self._session_task_id)
            if t:
                task_name = t.get("name")
        return {
            "active": self._session_active,
            "task_id": self._session_task_id,
            "task_name": task_name,
            "count": self._session_count,
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

        # Remove from whichever task contains it. If that empties a named task,
        # drop the task too — an empty task has no reason to exist (and, being
        # unreported, the fleet couldn't otherwise reach it to delete it). This
        # is what makes orphan cleanup that removes a task's last episode also
        # remove the now-empty task, instead of leaving a 0-episode ghost.
        for t in self._tasks:
            if episode_id in t["episode_ids"]:
                t["episode_ids"].remove(episode_id)
                t.get("episode_members", {}).pop(episode_id, None)
                if t["id"] != UNASSIGNED_ID and not t["episode_ids"]:
                    self._tasks.remove(t)
                break

        shutil.rmtree(ep_dir)
        self._save()

    def _apply_episode_members(self, episode_id: str, members: dict,
                               device_signature: list | None = None) -> bool:
        """Merge members into the episode's entry on the task that holds it,
        WITHOUT saving. Backfills the task's device signature if it has none yet.
        Returns False if the episode isn't on this device."""
        if not members or not episode_id:
            return False
        for t in self._tasks:
            if episode_id in t.get("episode_ids", []):
                t.setdefault("episode_members", {}).setdefault(episode_id, {}).update(members)
                if device_signature and not t.get("device_signature"):
                    t["device_signature"] = list(device_signature)
                return True
        return False

    def set_episode_members(self, episode_id: str, members: dict,
                            device_signature: list | None = None) -> bool:
        """Backfill who recorded an episode (role → {device_id, name}) on the task
        that holds it — used to repair episodes recorded before members were
        persisted. MERGES into any existing members (so partial repairs from
        different devices accumulate). No-op (returns False) if the episode isn't
        on this device."""
        if self._apply_episode_members(episode_id, members, device_signature):
            self._save()
            return True
        return False

    def set_episodes_members(self, entries: list[dict],
                             device_signature: list | None = None) -> int:
        """Batch backfill: entries = [{"episode_id": str, "members": {...}}, …].
        Applies every entry present on this device and saves ONCE. Returns how
        many episodes were updated locally (absent ones are skipped)."""
        n = 0
        for e in entries or []:
            if self._apply_episode_members(e.get("episode_id"), e.get("members") or {},
                                           device_signature):
                n += 1
        if n:
            self._save()
        return n

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
        # A real capture writes dcam_imu.json (or legacy oakd_imu.json); the mock
        # backend writes imu_data.json. Checking only the latter reported
        # has_imu=False for every real recording (issue #79). A Gemini 305
        # episode has none of them — that camera has no IMU — which is a
        # legitimate absence, not a fault.
        imu_present = (
            resolve(ep_dir, DCAM_IMU).exists()
            or (ep_dir / "imu_data.json").exists()
        )

        duration = 0.0
        frame_count = 0
        imu_sample_count = 0
        angle_sample_count = 0
        meta_path = ep_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            duration = meta.get("duration_seconds", 0.0)
            frame_count = meta.get("frame_count", 0)
            imu_sample_count = (meta.get("imu_sample_count")
                                or metadata_stats(meta).get("imu_samples", 0))
            angle_sample_count = meta.get("angle_sample_count", 0)

        return EpisodeInfo(
            episode_id=episode_id,
            created_at=episode_id,
            duration_seconds=duration,
            frame_count=frame_count,
            imu_sample_count=imu_sample_count,
            angle_sample_count=angle_sample_count,
            has_video=video_path.exists(),
            has_imu=imu_present,
        )

    def revision(self) -> int:
        """Monotonic counter bumped on every task-registry change. The relay
        re-reports tasks to the fleet only when this moves (cheap change-signal)."""
        return self._revision

    def report_tasks(self) -> list[dict]:
        """Snapshot of this device's recorded tasks for the fleet to aggregate.

        The device is the durable source of truth for tasks and who recorded
        each episode. On connect it hands this to the fleet, which merges the
        reports from all connected devices by task name — so any operator sees
        the tasks and can name each episode's peers, even ones now offline.

        Skips Unassigned and empty tasks (a task with no episodes has no reason
        to exist).

        Episodes are GROUPED by their membership combination (a task's episodes
        almost always share the same devices), so the members map is sent once
        per group instead of once per episode — keeping the report compact even
        for tasks with a long recording history.
        """
        out: list[dict] = []
        for t in self._tasks:
            if t["id"] == UNASSIGNED_ID or not t.get("episode_ids"):
                continue
            members_by_ep = t.get("episode_members", {})
            groups: dict[tuple, dict] = {}
            for eid in t["episode_ids"]:
                m = members_by_ep.get(eid, {})
                key = tuple(sorted((role, w.get("device_id")) for role, w in m.items()))
                g = groups.get(key)
                if g is None:
                    g = groups[key] = {"members": m, "episode_ids": []}
                g["episode_ids"].append(eid)
            out.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "device_signature": t.get("device_signature", []),
                "groups": list(groups.values()),
            })
        return out

    def report_unassigned(self) -> dict:
        """Loose episodes — recorded outside any task — for the fleet's inbox.

        Sent on a channel of its OWN, deliberately not as a task (report_tasks
        skips Unassigned, and keeps doing so). Tasks are cross-device and merged
        by NAME on the fleet, which is exactly wrong here: a loose episode was
        recorded alone and belongs to this device only, so merging every device's
        stray takes under one name would make them indistinguishable. Staying off
        that channel also means these episodes can never leak into task selection
        or dataset generation, which both walk the reported tasks.

        Each entry carries enough to triage without downloading anything: who
        recorded it (always this device — see local_identity), how long the take
        was, and whether it has video. That's the difference between a 1s misfire
        and a real recording, which is the whole decision the operator makes.

        Episodes whose directory is gone are skipped: nothing can be done with an
        entry that no longer has data behind it.
        """
        t = self._find_task(UNASSIGNED_ID)
        if t is None:
            return {"total": 0, "episodes": []}
        members_by_ep = t.get("episode_members", {})
        # An inbox episode with no stored membership predates the self-stamp, or was
        # migrated from the old layout. Its role is not unknown: we hold the files,
        # so OUR slot is what we contributed — the same inference the fleet already
        # makes for task episodes (_online_episode_reporters: "the reporting device
        # IS a member, and its role is its slot"). Asserting only our own role never
        # over-claims: the fleet unions members across devices.
        own_members = local_identity()[0]
        # Episode ids are UTC timestamps, so a plain sort is chronological; the
        # tail is therefore the most recent (legacy non-timestamp ids just sort
        # lexicographically, which is stable and good enough for a cap).
        eids = sorted(eid for eid in t["episode_ids"] if self.episode_dir(eid).exists())
        episodes = []
        for eid in eids[-UNASSIGNED_REPORT_LIMIT:]:
            info = self._get_episode_info(eid)
            episodes.append({
                "episode_id": eid,
                "members": members_by_ep.get(eid) or own_members,
                "duration_seconds": info.duration_seconds,
                "has_video": info.has_video,
            })
        return {"total": len(eids), "episodes": episodes}

    # ── Task operations ────────────────────────────────────────────────

    def create_task(self, name: str, description: str = "") -> str:
        task_id = uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._tasks.append({
            "id": task_id,
            "name": name,
            "description": description,
            "created_at": created_at,
            "episode_ids": [],
        })
        self._save()
        return task_id

    def get_or_create_task(self, name: str) -> str:
        """Resolve a task by exact name match, creating one if none exists.

        Task ids are per-device UUIDs with no meaning across devices, so a
        shared task name (e.g. from a fleet-dispatched group capture) is the
        only stable join key: each device independently resolves the same
        name to its own local task, created on first use.
        """
        for t in self._tasks:
            if t["name"] == name:
                return t["id"]
        return self.create_task(name)

    def get_task(self, task_id: str) -> TaskInfo:
        t = self._find_task(task_id)
        if t is None:
            raise FileNotFoundError(f"Task {task_id} not found")
        return self._to_task_info(t)

    def get_task_detail(self, task_id: str) -> TaskDetail:
        t = self._find_task(task_id)
        if t is None:
            raise FileNotFoundError(f"Task {task_id} not found")
        return self._to_task_detail(t)

    def update_task(self, task_id: str, name: str | None = None, description: str | None = None) -> TaskInfo:
        t = self._find_task(task_id)
        if t is None:
            raise FileNotFoundError(f"Task {task_id} not found")
        if task_id == UNASSIGNED_ID:
            raise ValueError("Cannot modify the Unassigned task")
        if name is not None:
            t["name"] = name
        if description is not None:
            t["description"] = description
        self._save()
        return self._to_task_info(t)

    def delete_task(self, task_id: str) -> None:
        if task_id == UNASSIGNED_ID:
            raise ValueError("Cannot delete the Unassigned task")
        t = self._find_task(task_id)
        if t is None:
            raise FileNotFoundError(f"Task {task_id} not found")

        # Move episodes back to Unassigned, WITH their membership. The task dict
        # — and the episode_members it carries — is dropped just below, so
        # leaving them behind wouldn't merely strand them: it would destroy the
        # record of who recorded each episode, which nothing on the device could
        # rebuild afterwards. (delete_task_by_name has no such need: it deletes
        # the episodes themselves.)
        unassigned = self._find_task(UNASSIGNED_ID)
        members = t.get("episode_members", {})
        for eid in t["episode_ids"]:
            if eid not in unassigned["episode_ids"]:
                unassigned["episode_ids"].append(eid)
            if members.get(eid):
                unassigned.setdefault("episode_members", {})[eid] = members[eid]

        self._tasks.remove(t)
        self._save()

    # ── Fleet-driven ops, keyed by task NAME ───────────────────────────
    # The fleet aggregates tasks across devices by name (it doesn't know local
    # ids), so edit/delete it dispatches reference the task by name. All are
    # idempotent (absent → no-op) so a command that reaches a device which never
    # had the task, or already applied it, still succeeds.

    def _find_task_by_name(self, name: str) -> dict | None:
        for t in self._tasks:
            if t["id"] != UNASSIGNED_ID and t.get("name") == name:
                return t
        return None

    def rename_task(self, name: str, new_name: str | None = None,
                    description: str | None = None,
                    device_signature: list | None = None) -> bool:
        """Update a task's name/description (and optionally backfill its device
        signature) by name. Returns False if absent."""
        t = self._find_task_by_name(name)
        if t is None:
            return False
        if new_name:
            t["name"] = new_name
        if description is not None:
            t["description"] = description
        if device_signature is not None:
            t["device_signature"] = list(device_signature)
        self._save()
        return True

    def delete_task_by_name(self, name: str) -> bool:
        """Delete a task AND its recorded episodes (files + registry) by name.
        Unlike delete_task(), episodes are removed, not moved to Unassigned —
        the fleet's "delete task" means the recordings go too. Returns False if
        the task isn't present on this device."""
        t = self._find_task_by_name(name)
        if t is None:
            return False
        members = t.get("episode_members", {})
        for eid in list(t.get("episode_ids", [])):
            ep_dir = self.episode_dir(eid)
            if ep_dir.exists():
                shutil.rmtree(ep_dir, ignore_errors=True)
            members.pop(eid, None)
        self._tasks.remove(t)
        self._save()
        return True

    def list_tasks(self) -> list[TaskDetail]:
        return [self._to_task_detail(t) for t in self._tasks]

    def move_episodes(self, episode_ids: list[str], target_task_id: str) -> dict:
        """Refile episodes into another task, carrying their membership along.

        An episode's members (role → {device_id, name}) live on the dict of the
        task that holds it, so moving the ids alone would strand them: the target
        would report the episode role-less — flagged incomplete by the fleet, and
        left out of any dataset — while a dead entry piled up on the source. This
        is the path that files a loose Unassigned recording into a real task, so
        the role stamped at record time has to survive it.

        Returns {"moved", "skipped", "shared"}:
          * moved   — episode ids now filed in the target (already-there included,
                      so a repeated call reports the same end state).
          * skipped — ids this device knows nothing about (see below).
          * shared  — [{"episode_id", "peers"}] for episodes recorded WITH another
                      device. Refiling only this device's copy leaves the peer
                      filing the same episode under the old task, which the fleet
                      then sees as a split. Reported, not refused: reorganising
                      both devices is legitimate, and a hard refusal here would
                      block doing it on either side. Callers surface it as a
                      warning.
        """
        target = self._find_task(target_task_id)
        if target is None:
            raise FileNotFoundError(f"Target task {target_task_id} not found")
        from grabette.config import settings  # our own id, to tell peers apart

        own = settings.device_id
        moved: list[str] = []
        skipped: list[str] = []
        shared: list[dict] = []
        filed: list[dict] = []  # membership of each episode filed here, for the signature
        for eid in episode_ids:
            source = next((t for t in self._tasks if eid in t["episode_ids"]), None)
            if source is None and not self.episode_dir(eid).exists():
                # Nothing here holds it and there are no files either. A fleet
                # dispatch reaches every device, and most never recorded a given
                # episode — registering the id for them would invent an episode
                # that doesn't exist, which the fleet would then report. (An
                # episode present on disk but in no task is NOT skipped: filing it
                # repairs a lost registry entry.)
                skipped.append(eid)
                continue
            if source is target:
                # Already filed here, so stop before touching anything: the
                # lookup found the TARGET as the holder, and removing its members
                # would strip them while the guard below skipped the re-add —
                # losing data on what is supposed to be a no-op.
                members = target.get("episode_members", {}).get(eid) or {}
            else:
                members = None
                if source is not None:
                    source["episode_ids"].remove(eid)
                    members = source.get("episode_members", {}).pop(eid, None)
                if eid not in target["episode_ids"]:
                    target["episode_ids"].append(eid)
                if members:
                    target.setdefault("episode_members", {})[eid] = members
                members = members or {}
            if not members:
                # Nothing recorded about who made it (predates the self-stamp, or
                # migrated). Filing it is the moment to assert the one role we know
                # for certain: ours. We hold the files, so our slot is what this
                # device contributed — never a claim about anyone else, and the
                # fleet unions members across devices, so a peer filing the same
                # episode adds its own role. Without this, filing an old episode
                # produced a task the fleet reports as role-less, hence unusable
                # for a dataset.
                members = local_identity()[0]
                target.setdefault("episode_members", {})[eid] = members
            moved.append(eid)
            filed.append(members)
            peers = sorted({w.get("name") or w.get("device_id") for w in members.values()
                            if w.get("device_id") and w.get("device_id") != own})
            if peers:
                shared.append({"episode_id": eid, "peers": peers})

        self._adopt_signature(target, filed)
        self._save()
        return {"moved": moved, "skipped": skipped, "shared": shared}

    def _adopt_signature(self, target: dict, filed: list[dict]) -> None:
        """Give a signature-less task the role set of the episodes just filed in.

        Without this, filing loose episodes into a fresh task leaves it with no
        signature at all — and a task with no signature exposes no usable roles,
        so dataset generation refuses it outright. Same conservative rules as
        register_episode: fill only what is absent (an existing signature
        describes the episodes already recorded there and must never be
        rewritten), and never on Unassigned, a bucket of mixed provenance.

        Only when every filed episode agrees on ONE role set: a batch mixing a
        left-only episode with a bimanual one would otherwise be described as
        ["left","right"], which the left-only episode does not satisfy.
        """
        if target["id"] == UNASSIGNED_ID or target.get("device_signature"):
            return
        role_sets = {tuple(sorted(m)) for m in filed if m}
        if len(role_sets) != 1:
            return
        target["device_signature"] = list(role_sets.pop())

    # ── Helpers ────────────────────────────────────────────────────────

    def _to_task_info(self, t: dict) -> TaskInfo:
        episodes = [
            self._get_episode_info(eid)
            for eid in t["episode_ids"]
            if self.episode_dir(eid).exists()
        ]
        return TaskInfo(
            id=t["id"],
            name=t["name"],
            description=t.get("description", ""),
            created_at=t["created_at"],
            episode_ids=t["episode_ids"],
            episode_count=len(episodes),
            total_duration=sum(e.duration_seconds for e in episodes),
        )

    def _to_task_detail(self, t: dict) -> TaskDetail:
        episodes = [
            self._get_episode_info(eid)
            for eid in t["episode_ids"]
            if self.episode_dir(eid).exists()
        ]
        return TaskDetail(
            id=t["id"],
            name=t["name"],
            description=t.get("description", ""),
            created_at=t["created_at"],
            episode_ids=t["episode_ids"],
            episode_count=len(episodes),
            total_duration=sum(e.duration_seconds for e in episodes),
            episodes=episodes,
        )
