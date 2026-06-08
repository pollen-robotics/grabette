from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend
from grabette.session import SessionManager

router = APIRouter(tags=["sessions"])

_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager


# ── Session endpoints ─────────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    name: str
    description: str = ""


class UpdateSessionRequest(BaseModel):
    name: str | None = None
    description: str | None = None


@router.get("/api/sessions")
def list_sessions(sm: SessionManager = Depends(get_session_manager)):
    return sm.list_sessions()


@router.post("/api/sessions")
def create_session(
    req: CreateSessionRequest,
    sm: SessionManager = Depends(get_session_manager),
):
    session_id = sm.create_session(req.name, req.description)
    return sm.get_session(session_id)


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str, sm: SessionManager = Depends(get_session_manager)):
    try:
        return sm.get_session_detail(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.put("/api/sessions/{session_id}")
def update_session(
    session_id: str,
    req: UpdateSessionRequest,
    sm: SessionManager = Depends(get_session_manager),
):
    try:
        return sm.update_session(session_id, req.name, req.description)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, sm: SessionManager = Depends(get_session_manager)):
    try:
        sm.delete_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": session_id}


# ── Episode endpoints ─────────────────────────────────────────────────


class MoveEpisodesRequest(BaseModel):
    episode_ids: list[str]
    target_session_id: str


@router.post("/api/episodes/start")
async def start_capture(
    backend: Backend = Depends(get_backend),
    sm: SessionManager = Depends(get_session_manager),
):
    if backend.is_capturing:
        raise HTTPException(status_code=409, detail="Already capturing")
    episode_id = sm.create_episode()
    episode_dir = sm.episode_dir(episode_id)
    await backend.start_capture(episode_dir)
    return {"episode_id": episode_id, "status": "capturing"}


@router.post("/api/episodes/stop")
async def stop_capture(backend: Backend = Depends(get_backend)):
    if not backend.is_capturing:
        raise HTTPException(status_code=409, detail="Not capturing")
    status = await backend.stop_capture()
    return status


@router.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str, sm: SessionManager = Depends(get_session_manager)):
    try:
        return sm.get_episode(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Episode not found")


@router.get("/api/episodes/{episode_id}/download")
def download_episode(episode_id: str, sm: SessionManager = Depends(get_session_manager)):
    try:
        archive_path = sm.create_episode_archive(episode_id)
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
def download_episodes(req: DownloadEpisodesRequest, sm: SessionManager = Depends(get_session_manager)):
    if not req.episode_ids:
        raise HTTPException(status_code=400, detail="No episode IDs provided")
    archive_path = sm.create_episodes_zip(req.episode_ids)
    filename = f"episodes_{req.episode_ids[0]}.tar.gz" if len(req.episode_ids) == 1 else "episodes.tar.gz"
    return FileResponse(archive_path, media_type="application/gzip", filename=filename)


@router.get("/api/episodes/{episode_id}/video")
def stream_video(episode_id: str, sm: SessionManager = Depends(get_session_manager)):
    video_path = sm.episode_dir(episode_id) / "raw_video.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@router.delete("/api/episodes/{episode_id}")
def delete_episode(episode_id: str, sm: SessionManager = Depends(get_session_manager)):
    try:
        sm.delete_episode(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"deleted": episode_id}


@router.post("/api/episodes/move")
def move_episodes(
    req: MoveEpisodesRequest,
    sm: SessionManager = Depends(get_session_manager),
):
    try:
        sm.move_episodes(req.episode_ids, req.target_session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"moved": req.episode_ids, "target_session_id": req.target_session_id}
