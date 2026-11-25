"""Tasks and episodes on this device, for the dashboard's Episodes page.

Read-mostly: tasks are created and edited from the fleet. The dashboard lists
them, plays, checks, downloads and deletes episodes.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from casquette.task import get_task_manager

router = APIRouter(tags=["tasks"])

# Episode ids are timestamps (episode_id_for); refusing anything else keeps a
# crafted id from walking out of the episodes directory.
_EPISODE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# What one recording leaves behind once it is fully saved.
REQUIRED_FILES = ("metadata.json", "raw_video.mp4", "imu_data.json", "frame_timestamps.json")


def _episode_dir(episode_id: str) -> Path:
    if not _EPISODE_ID_RE.match(episode_id):
        raise HTTPException(status_code=400, detail="Invalid episode id")
    ep_dir = get_task_manager().episode_dir(episode_id)
    if not ep_dir.exists():
        raise HTTPException(status_code=404, detail="Episode not found")
    return ep_dir


def _is_busy(episode_id: str | None = None) -> bool:
    """A capture is running or scheduled (on this episode, when one is given)."""
    from casquette.app.main import get_daemon_instance
    from casquette.fleet.capture_scheduler import get_capture_scheduler

    pending = get_task_manager().pending_episode_id
    if episode_id is not None:
        return pending == episode_id
    daemon = get_daemon_instance()
    capturing = bool(daemon and daemon.state.value == "running" and daemon.backend.is_capturing)
    return capturing or get_capture_scheduler().is_scheduled() or pending is not None


def episode_check(ep_dir: Path) -> dict:
    """What this episode lacks, from what is on disk.

    `saving` while the video is still being muxed from its .h264 (a few seconds
    after a stop on the Pi Zero 2W): not a fault, just not finished.
    """
    missing = [f for f in REQUIRED_FILES if not (ep_dir / f).exists()]
    saving = "raw_video.mp4" in missing and any(ep_dir.glob("*.h264"))
    camera_ok = True
    meta_path = ep_dir / "metadata.json"
    if meta_path.exists():
        try:
            camera_ok = json.loads(meta_path.read_text()).get("camera_ok", True) is not False
        except (OSError, ValueError):
            missing.append("metadata.json (unreadable)")
    return {
        "missing": missing,
        "saving": saving,
        "camera_ok": camera_ok,
        "complete": not missing and camera_ok,
    }


@router.get("/api/tasks")
def list_tasks():
    tasks = []
    for t in get_task_manager().list_tasks():
        d = t.model_dump()
        for ep in d["episodes"]:
            ep.update(episode_check(get_task_manager().episode_dir(ep["episode_id"])))
        tasks.append(d)
    return tasks


def _archive_response(archive_path: Path, filename: str) -> StreamingResponse:
    """Stream the archive from an open handle and unlink it right away, so it
    is gone even if the download is cancelled."""
    f = archive_path.open("rb")
    size = os.fstat(f.fileno()).st_size
    archive_path.unlink()

    def chunks():
        with f:
            yield from iter(lambda: f.read(1 << 20), b"")

    return StreamingResponse(chunks(), media_type="application/gzip", headers={
        "Content-Length": str(size),
        "Content-Disposition": f'attachment; filename="{filename}"',
    })


@router.get("/api/episodes/download")
def download_episodes(episode_ids: list[str] = Query()):
    for eid in episode_ids:
        _episode_dir(eid)
    archive = get_task_manager().create_episodes_zip(episode_ids)
    name = f"{episode_ids[0]}.tar.gz" if len(episode_ids) == 1 else "episodes.tar.gz"
    return _archive_response(archive, name)


@router.get("/api/episodes/{episode_id}/check")
def check_episode(episode_id: str):
    return {"episode_id": episode_id, **episode_check(_episode_dir(episode_id))}


@router.get("/api/episodes/{episode_id}/video")
def stream_video(episode_id: str):
    video = _episode_dir(episode_id) / "raw_video.mp4"
    if not video.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video, media_type="video/mp4")


@router.delete("/api/episodes")
def delete_all_episodes():
    if _is_busy():
        raise HTTPException(status_code=409, detail="A recording is in progress")
    return {"deleted": get_task_manager().delete_all_episodes()}


@router.delete("/api/episodes/{episode_id}")
def delete_episode(episode_id: str):
    _episode_dir(episode_id)
    if _is_busy(episode_id):
        raise HTTPException(status_code=409, detail="This episode is still being recorded")
    get_task_manager().delete_episode(episode_id)
    return {"deleted": episode_id}
