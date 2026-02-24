"""Replay API — start/stop/pause/resume/seek session replay."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from grabette.app.dependencies import get_daemon
from grabette.config import settings
from grabette.daemon import Daemon

router = APIRouter(prefix="/api/replay", tags=["replay"])


class ReplayStartRequest(BaseModel):
    session_id: str


class ReplaySeekRequest(BaseModel):
    time_ms: float


@router.post("/start")
async def start_replay(body: ReplayStartRequest, daemon: Daemon = Depends(get_daemon)):
    session_dir = settings.data_dir / body.session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if not (session_dir / "imu_data.json").exists():
        raise HTTPException(status_code=400, detail="Session has no IMU data")
    try:
        await daemon.start_replay(str(session_dir), body.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return daemon.replay_status


@router.post("/stop")
async def stop_replay(daemon: Daemon = Depends(get_daemon)):
    await daemon.stop_replay()
    return {"active": False}


@router.post("/pause")
async def pause_replay(daemon: Daemon = Depends(get_daemon)):
    await daemon.replay_pause()
    return daemon.replay_status


@router.post("/resume")
async def resume_replay(daemon: Daemon = Depends(get_daemon)):
    await daemon.replay_resume()
    return daemon.replay_status


@router.post("/seek")
async def seek_replay(body: ReplaySeekRequest, daemon: Daemon = Depends(get_daemon)):
    await daemon.replay_seek(body.time_ms)
    return daemon.replay_status


@router.get("/status")
def replay_status(daemon: Daemon = Depends(get_daemon)):
    return daemon.replay_status
