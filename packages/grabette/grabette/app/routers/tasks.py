from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend
from grabette.capture_scheduler import get_capture_scheduler
from grabette.fleet_sync import notify_group_stop, request_group_start
from grabette.task import TaskManager, episode_id_for

router = APIRouter(tags=["tasks"])

_task_manager = TaskManager()


def get_task_manager() -> TaskManager:
    return _task_manager


# ── Task endpoints ─────────────────────────────────────────────────────


class CreateTaskRequest(BaseModel):
    name: str
    description: str = ""


class UpdateTaskRequest(BaseModel):
    name: str | None = None
    description: str | None = None


@router.get("/api/tasks")
def list_tasks(tm: TaskManager = Depends(get_task_manager)):
    return tm.list_tasks()


@router.post("/api/tasks")
def create_task(
    req: CreateTaskRequest,
    tm: TaskManager = Depends(get_task_manager),
):
    task_id = tm.create_task(req.name, req.description)
    return tm.get_task(task_id)


@router.get("/api/tasks/active")
def get_active_task(tm: TaskManager = Depends(get_task_manager)):
    return {"task_id": tm.active_task_id}


class SetActiveTaskRequest(BaseModel):
    task_id: str


@router.put("/api/tasks/active")
def set_active_task(
    req: SetActiveTaskRequest,
    tm: TaskManager = Depends(get_task_manager),
):
    if tm._find_task(req.task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    tm.active_task_id = req.task_id
    return {"task_id": tm.active_task_id}


@router.get("/api/tasks/{task_id}")
def get_task(task_id: str, tm: TaskManager = Depends(get_task_manager)):
    try:
        return tm.get_task_detail(task_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")


@router.put("/api/tasks/{task_id}")
def update_task(
    task_id: str,
    req: UpdateTaskRequest,
    tm: TaskManager = Depends(get_task_manager),
):
    try:
        return tm.update_task(task_id, req.name, req.description)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/tasks/{task_id}")
def delete_task(task_id: str, tm: TaskManager = Depends(get_task_manager)):
    try:
        tm.delete_task(task_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": task_id}


# ── Session endpoints (recording-run lock) ─────────────────────────────


@router.get("/api/session/status")
def session_status(tm: TaskManager = Depends(get_task_manager)):
    return tm.get_session_status()


class StartSessionRequest(BaseModel):
    task_id: str | None = None


@router.post("/api/session/start")
def start_session(
    req: StartSessionRequest = StartSessionRequest(),
    tm: TaskManager = Depends(get_task_manager),
):
    task_id = req.task_id or tm.active_task_id
    try:
        tm.start_session(task_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return tm.get_session_status()


@router.post("/api/session/stop")
def stop_session(tm: TaskManager = Depends(get_task_manager)):
    tm.stop_session()
    return {"active": False}


# ── Episode endpoints ─────────────────────────────────────────────────


class MoveEpisodesRequest(BaseModel):
    episode_ids: list[str]
    target_task_id: str


class StartCaptureRequest(BaseModel):
    task_id: str | None = None


@router.post("/api/episodes/start")
async def start_capture(
    req: StartCaptureRequest = StartCaptureRequest(),
    backend: Backend = Depends(get_backend),
    tm: TaskManager = Depends(get_task_manager),
):
    if backend.is_capturing:
        raise HTTPException(status_code=409, detail="Already capturing")
    scheduler = get_capture_scheduler()
    if scheduler.is_scheduled():
        raise HTTPException(status_code=409, detail="A start is already scheduled")

    # If this device is grouped, a start here behaves like the fleet "start
    # group recording" button: the GROUP's task wins and the start is
    # synchronized at the shared T0 — so we don't impose the locally-selected
    # task. Solo (no group) → the requested/active local task, immediately.
    target_task_id = req.task_id or tm.active_task_id
    sync = await request_group_start("")
    status = sync.get("status")
    if status == "refused":
        # Fleet knows this device is in a group session but declined (e.g. a
        # peer is offline). Refuse rather than silently record a half-rig
        # solo episode.
        raise HTTPException(status_code=409,
                            detail={"message": "group start refused (a peer may be offline)",
                                    "detail": sync.get("detail")})
    target = None
    if status == "scheduled":
        gname = sync.get("task_name") or ""
        if gname:
            target_task_id = tm.get_or_create_task(gname)
        target = datetime.fromisoformat(sync["scheduled_start_utc"])

    # A group-synchronized start derives the episode id from the shared T0
    # (not from local wall-clock creation time), so every device's episode
    # folder for this recording ends up with the SAME name even though each
    # one actually creates its directory at a different real-world moment.
    episode_id = tm.create_episode(target_task_id, episode_id=episode_id_for(target) if target else None)
    episode_dir = tm.episode_dir(episode_id)

    if target is not None:
        await scheduler.schedule(backend, tm, episode_dir, target)
        return {
            "episode_id": episode_id,
            "status": "scheduled",
            "start_at_utc": sync["scheduled_start_utc"],
            "peers": sync.get("peers", []),
        }

    try:
        await backend.start_capture(episode_dir)
    except Exception:
        tm.discard_pending_episode()
        raise
    return {"episode_id": episode_id, "status": "capturing"}


@router.post("/api/episodes/stop")
async def stop_capture(
    backend: Backend = Depends(get_backend),
    tm: TaskManager = Depends(get_task_manager),
):
    scheduler = get_capture_scheduler()
    try:
        outcome = await scheduler.cancel_or_wait(backend)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if outcome == "cancelled":
        tm.discard_pending_episode()
        return {"status": "cancelled"}
    if not backend.is_capturing:
        raise HTTPException(status_code=409, detail="Not capturing")
    status = await backend.stop_capture()
    # File the episode into its task only now that its data is written.
    tm.register_episode(getattr(status, "episode_id", None))
    asyncio.create_task(notify_group_stop())  # best-effort; must not delay this response
    return status


@router.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str, tm: TaskManager = Depends(get_task_manager)):
    try:
        return tm.get_episode(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Episode not found")


@router.get("/api/episodes/{episode_id}/download")
def download_episode(episode_id: str, tm: TaskManager = Depends(get_task_manager)):
    try:
        archive_path = tm.create_episode_archive(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Episode not found")
    return FileResponse(
        archive_path,
        media_type="application/gzip",
        filename=f"{episode_id}.tar.gz",
    )


class DownloadEpisodesRequest(BaseModel):
    episode_ids: list[str]


@router.post("/api/episodes/download")
def download_episodes(req: DownloadEpisodesRequest, tm: TaskManager = Depends(get_task_manager)):
    if not req.episode_ids:
        raise HTTPException(status_code=400, detail="No episode IDs provided")
    archive_path = tm.create_episodes_zip(req.episode_ids)
    filename = f"episodes_{req.episode_ids[0]}.tar.gz" if len(req.episode_ids) == 1 else "episodes.tar.gz"
    return FileResponse(archive_path, media_type="application/gzip", filename=filename)


@router.get("/api/episodes/{episode_id}/video")
def stream_video(episode_id: str, tm: TaskManager = Depends(get_task_manager)):
    video_path = tm.episode_dir(episode_id) / "raw_video.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@router.delete("/api/episodes/{episode_id}")
def delete_episode(episode_id: str, tm: TaskManager = Depends(get_task_manager)):
    try:
        tm.delete_episode(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"deleted": episode_id}


@router.post("/api/episodes/move")
def move_episodes(
    req: MoveEpisodesRequest,
    tm: TaskManager = Depends(get_task_manager),
):
    try:
        tm.move_episodes(req.episode_ids, req.target_task_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"moved": req.episode_ids, "target_task_id": req.target_task_id}
