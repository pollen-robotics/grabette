from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from grabette.app.routers.sessions import get_session_manager
from grabette.hf import HuggingFaceClient
from grabette.jobs import JobStatus, get_job_manager
from grabette.session import SessionManager

router = APIRouter(prefix="/api/hf", tags=["huggingface"])

_hf_client = HuggingFaceClient()


def get_hf_client() -> HuggingFaceClient:
    return _hf_client


class AuthRequest(BaseModel):
    token: str


class UploadRequest(BaseModel):
    repo_id: str


@router.post("/auth")
def set_auth(req: AuthRequest, hf: HuggingFaceClient = Depends(get_hf_client)):
    hf.set_token(req.token)
    if not hf.is_authenticated:
        raise HTTPException(status_code=401, detail="Invalid token")
    info = hf.get_user_info()
    return {"authenticated": True, "user": info}


@router.get("/auth")
def check_auth(hf: HuggingFaceClient = Depends(get_hf_client)):
    if not hf.is_authenticated:
        return {"authenticated": False}
    info = hf.get_user_info()
    return {"authenticated": True, "user": info}


@router.post("/upload/{episode_id}")
async def upload_episode(
    episode_id: str,
    req: UploadRequest,
    hf: HuggingFaceClient = Depends(get_hf_client),
    sm: SessionManager = Depends(get_session_manager),
):
    if not hf.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated with HuggingFace")

    try:
        sm.get_episode(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Episode not found")

    episode_dir = sm.episode_dir(episode_id)
    jm = get_job_manager()
    job = jm.create_job(f"upload:{episode_id}")

    async def _run_upload():
        try:
            jm.update_progress(job.job_id, 0.0, "Starting upload...")

            def progress_cb(pct: float, msg: str):
                jm.update_progress(job.job_id, pct, msg)

            url = await asyncio.to_thread(
                hf.upload_episode, episode_dir, req.repo_id, progress_cb
            )
            jm.complete_job(job.job_id, url)
        except Exception as e:
            jm.fail_job(job.job_id, str(e))

    asyncio.create_task(_run_upload())
    return {"job_id": job.job_id, "status": "started"}


@router.get("/jobs")
def list_jobs():
    jm = get_job_manager()
    return [
        {
            "job_id": j.job_id,
            "name": j.name,
            "status": j.status.value,
            "progress": j.progress,
            "message": j.message,
            "result": j.result,
            "error": j.error,
        }
        for j in jm.list_jobs()
    ]


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "error": job.error,
    }


@router.post("/slam/{episode_id}")
async def run_slam(
    episode_id: str,
    req: UploadRequest,
    hf: HuggingFaceClient = Depends(get_hf_client),
    sm: SessionManager = Depends(get_session_manager),
):
    """Upload an episode and trigger SLAM processing."""
    if not hf.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated with HuggingFace")

    try:
        sm.get_episode(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Episode not found")

    episode_dir = sm.episode_dir(episode_id)

    from grabette.slam import get_slam_orchestrator
    slam = get_slam_orchestrator()
    job_id = await slam.run_slam(episode_id, episode_dir, req.repo_id, hf)

    return {"job_id": job_id, "status": "started"}


@router.websocket("/upload/{episode_id}/ws")
async def upload_progress_ws(ws: WebSocket, episode_id: str):
    """Stream upload progress for an episode."""
    await ws.accept()
    jm = get_job_manager()
    try:
        while True:
            # Find the most recent upload job for this episode
            job = None
            for j in reversed(jm.list_jobs()):
                if j.name == f"upload:{episode_id}":
                    job = j
                    break
            if job:
                await ws.send_json({
                    "status": job.status.value,
                    "progress": job.progress,
                    "message": job.message,
                    "result": job.result,
                    "error": job.error,
                })
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
