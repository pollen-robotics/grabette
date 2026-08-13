"""Thin HTTP client wrapping the grabette REST API."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Read-response in 8 MB chunks. Large enough to keep syscall overhead low,
# small enough that a multi-GB download never puts >8 MB in RAM at a time.
# The old `r.content` reader buffered the whole response (>1 GB for a full
# dataset), which OOM'd on the 4 GB Pi and silently returned None.
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


class GrabetteClient:
    """Synchronous client for the grabette REST/WebSocket API.

    Used by both the local Gradio dashboard and the HF Spaces app.
    """

    def __init__(
        self,
        base_url: str | None = None,
        download_dir: str | Path | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("GRABETTE_API_URL")
            or "http://localhost:8000"
        )
        # Where downloaded episode archives are staged before Gradio serves
        # them to the browser. Passed explicitly by the local daemon so the
        # multi-GB tar.gz lands on the SD card, NOT on Pi OS's tmpfs /tmp
        # (which is small enough to fill after a couple of downloads and
        # brick the whole daemon with ENOSPC). Falls back to the OS temp
        # dir for callers that don't set it — fine on workstations / HF
        # Spaces where /tmp is a normal-sized filesystem.
        self._download_dir = Path(download_dir) if download_dir else Path(tempfile.gettempdir())
        self._http = httpx.Client(base_url=self.base_url, timeout=10.0)

    # -- Camera --

    def get_snapshot(self) -> bytes | None:
        try:
            r = self._http.get("/api/camera/snapshot")
            r.raise_for_status()
            return r.content
        except Exception:
            return None

    def get_depth_snapshot(self) -> bytes | None:
        try:
            r = self._http.get("/api/camera/depth")
            if r.status_code != 200:
                return None
            return r.content
        except Exception:
            return None

    def get_camera_status(self) -> dict | None:
        try:
            r = self._http.get("/api/camera/status")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # -- Sensor state --

    def get_state(self) -> dict | None:
        try:
            r = self._http.get("/api/state")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # -- Teleop --

    def get_teleop_status(self) -> dict | None:
        try:
            r = self._http.get("/api/teleop/status")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # -- OAK-D --

    def get_oakd_status(self) -> dict | None:
        try:
            r = self._http.get("/api/oakd/status")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def set_oakd(self, on: bool) -> dict:
        path = "/api/oakd/enable" if on else "/api/oakd/disable"
        try:
            r = self._http.post(path)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    # -- Capture --

    def start_capture(self, task_id: str | None = None) -> dict:
        try:
            body = {"task_id": task_id} if task_id else {}
            r = self._http.post("/api/episodes/start", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def stop_capture(self) -> dict:
        try:
            r = self._http.post("/api/episodes/stop")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def get_session_status(self) -> dict:
        try:
            r = self._http.get("/api/session/status")
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"active": False, "task_id": None, "task_name": None, "count": 0}

    def start_session(self, task_id: str | None = None) -> dict:
        try:
            body = {"task_id": task_id} if task_id else {}
            r = self._http.post("/api/session/start", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    def stop_session(self) -> dict:
        try:
            r = self._http.post("/api/session/stop")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_active_task(self) -> str | None:
        try:
            r = self._http.get("/api/tasks/active")
            r.raise_for_status()
            return r.json().get("task_id")
        except Exception:
            return None

    def set_active_task(self, task_id: str) -> dict:
        try:
            r = self._http.put("/api/tasks/active", json={"task_id": task_id})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    # -- Tasks --

    def list_tasks(self) -> list[dict]:
        try:
            r = self._http.get("/api/tasks")
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    # -- Episodes --

    def delete_episode(self, episode_id: str) -> dict:
        try:
            r = self._http.delete(f"/api/episodes/{episode_id}")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def download_episodes(self, episode_ids: list[str]) -> str | None:
        try:
            filename = "episodes.tar.gz" if len(episode_ids) > 1 else f"{episode_ids[0]}.tar.gz"
            self._download_dir.mkdir(parents=True, exist_ok=True)
            path = str(self._download_dir / filename)
            with self._http.stream(
                "POST",
                "/api/episodes/download",
                json={"episode_ids": episode_ids},
                timeout=None,
            ) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                        f.write(chunk)
            return path
        except Exception:
            logger.exception("download_episodes(%s) failed", episode_ids)
            return None

    def move_episodes(self, episode_ids: list[str], target_task_id: str) -> dict:
        try:
            r = self._http.post(
                "/api/episodes/move",
                json={"episode_ids": episode_ids, "target_task_id": target_task_id},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    # -- System --

    def get_system_info(self) -> dict | None:
        try:
            r = self._http.get("/api/system/info")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def shutdown(self) -> dict:
        try:
            r = self._http.post("/api/system/shutdown")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    # -- Replay --

    def replay_start(self, episode_id: str) -> dict:
        try:
            r = self._http.post("/api/replay/start", json={"episode_id": episode_id})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def replay_stop(self) -> dict:
        try:
            r = self._http.post("/api/replay/stop")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def replay_pause(self) -> dict:
        try:
            r = self._http.post("/api/replay/pause")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def replay_resume(self) -> dict:
        try:
            r = self._http.post("/api/replay/resume")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def replay_seek(self, time_ms: float) -> dict:
        try:
            r = self._http.post("/api/replay/seek", json={"time_ms": time_ms})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def replay_status(self) -> dict:
        try:
            r = self._http.get("/api/replay/status")
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"active": False, "episode_id": None, "time_ms": 0, "duration_ms": 0, "playing": False}

    # -- WiFi --

    def wifi_status(self) -> dict:
        try:
            r = self._http.get("/api/wifi/status", timeout=3.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"mode": "offline", "ssid": None, "ip": None}

