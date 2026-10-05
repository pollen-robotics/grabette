"""Speaker volume and test, for the dashboard's volume control.

The volume is the codec's line-out level (see hardware/sound.py), shown as
0..100 %. It is persisted on every change, so the next start of the daemon
plays the cues at the level last picked here; GRABETTE_SOUND_VOLUME is only the
default it falls back to, and what un-muting goes back to.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from grabette.config import settings
from grabette.hardware import sound

router = APIRouter(prefix="/api/sound", tags=["sound"])


class VolumeRequest(BaseModel):
    volume: int = Field(ge=0, le=100)
    # Beep at the new level, so the change is heard. Off for a mute.
    beep: bool = True


def _status() -> dict:
    speaker = sound.get_speaker()
    return {
        "volume": round(speaker.volume * 100),
        "default": round(max(0.0, min(1.0, settings.sound_volume)) * 100),
        "available": speaker.is_available,
        "testing": speaker.is_testing,
    }


def _is_capturing() -> bool:
    try:
        from grabette.app.main import get_daemon_instance

        daemon = get_daemon_instance()
        if daemon is None or daemon.state.value != "running":
            return False
        cap = daemon.backend.get_state().capture
        return bool(cap.is_capturing or cap.is_starting)
    except Exception:
        return False


@router.get("")
def get_sound():
    return _status()


@router.put("/volume")
def set_volume(req: VolumeRequest):
    speaker = sound.get_speaker()
    speaker.set_volume(req.volume / 100)
    try:
        sound.save_volume(speaker.volume)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not save the volume: {e}")
    if req.beep:
        speaker.play_preview()
    return _status()


@router.post("/test")
def test_sounds():
    """Play the recording cues once, in the order a take produces them."""
    # The sequence opens with the "recording is live" cue: played mid-take it
    # would tell the operator a recording just started.
    if _is_capturing():
        raise HTTPException(status_code=409, detail="Not during a recording.")
    status = _status()
    if not status["available"]:
        raise HTTPException(status_code=409, detail="No speaker on this Grabette.")
    if status["volume"] == 0:
        raise HTTPException(status_code=409, detail="The speaker is muted.")
    if not sound.get_speaker().play_test_sequence():
        raise HTTPException(status_code=409, detail="A test is already playing.")
    return status
