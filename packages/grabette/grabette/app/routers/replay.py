"""Replay API — start/stop/pause/resume/seek session replay."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from grabette.app.dependencies import get_daemon
from grabette.config import settings
from grabette.daemon import Daemon

router = APIRouter(prefix="/api/replay", tags=["replay"])


class ReplayStartRequest(BaseModel):
    episode_id: str


class ReplaySeekRequest(BaseModel):
    time_ms: float


@router.post("/start")
async def start_replay(body: ReplayStartRequest, daemon: Daemon = Depends(get_daemon)):
    episode_dir = settings.data_dir / "episodes" / body.episode_id
    if not episode_dir.exists():
        raise HTTPException(status_code=404, detail="Episode not found")
    if not (episode_dir / "imu_data.json").exists():
        raise HTTPException(status_code=400, detail="Episode has no IMU data")
    try:
        await daemon.start_replay(str(episode_dir), body.episode_id)
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


# ── Video replay page ──────────────────────────────────────────────

REPLAY_VIDEO_HTML = """\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
body{margin:0;background:#000;overflow:hidden;display:flex;align-items:center;justify-content:center;height:100vh}
video{max-width:100%;max-height:100%;object-fit:contain}
#msg{color:#888;font:14px monospace;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%)}
</style>
</head><body>
<video id="v" muted></video>
<div id="msg">Loading...</div>
<script>
(function(){
  var params=new URLSearchParams(location.search);
  var eid=params.get('episode_id');
  var v=document.getElementById('v');
  var msg=document.getElementById('msg');
  if(!eid){msg.textContent='No episode';return;}
  v.src='/api/episodes/'+encodeURIComponent(eid)+'/video';
  v.load();
  var wasPlaying=false, synced=false;

  setInterval(function(){
    fetch('/api/replay/status').then(function(r){return r.ok?r.json():null;})
    .then(function(st){
      if(!st)return;
      if(!st.active){msg.textContent='Replay ended';v.pause();return;}
      msg.style.display='none';
      var target=st.time_ms/1000;
      // Sync on seek or large drift
      if(Math.abs(v.currentTime-target)>0.5||!synced){
        v.currentTime=target;
        synced=true;
      }
      if(st.playing&&v.paused){v.play().catch(function(){});}
      if(!st.playing&&!v.paused){v.pause();}
      wasPlaying=st.playing;
    }).catch(function(){});
  },200);
})();
</script>
</body></html>"""


@router.get("/video")
async def replay_video_page():
    """Serve the video replay player page (embedded in iframe)."""
    return HTMLResponse(content=REPLAY_VIDEO_HTML)
