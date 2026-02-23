from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from grabette.app.dependencies import get_backend, get_daemon
from grabette.backend.base import Backend
from grabette.daemon import Daemon
from grabette.session import SessionManager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager


@router.post("/start")
async def start_capture(
    backend: Backend = Depends(get_backend),
    sm: SessionManager = Depends(get_session_manager),
):
    if backend.is_capturing:
        raise HTTPException(status_code=409, detail="Already capturing")
    session_id = sm.create_session()
    session_dir = sm._session_dir(session_id)
    await backend.start_capture(session_dir)
    return {"session_id": session_id, "status": "capturing"}


@router.post("/stop")
async def stop_capture(backend: Backend = Depends(get_backend)):
    if not backend.is_capturing:
        raise HTTPException(status_code=409, detail="Not capturing")
    status = await backend.stop_capture()
    return status


@router.get("")
def list_sessions(sm: SessionManager = Depends(get_session_manager)):
    return sm.list_sessions()


@router.get("/{session_id}")
def get_session(session_id: str, sm: SessionManager = Depends(get_session_manager)):
    try:
        return sm.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/download")
def download_session(session_id: str, sm: SessionManager = Depends(get_session_manager)):
    try:
        archive_path = sm.create_archive(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    return FileResponse(
        archive_path,
        media_type="application/gzip",
        filename=f"{session_id}.tar.gz",
    )


@router.delete("/{session_id}")
def delete_session(session_id: str, sm: SessionManager = Depends(get_session_manager)):
    try:
        sm.delete_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}
