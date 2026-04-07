"""Thin HTTP client wrapping the grabette REST API."""

from __future__ import annotations

import os
import tempfile

import httpx


class GrabetteClient:
    """Synchronous client for the grabette REST/WebSocket API.

    Used by both the local Gradio dashboard and the HF Spaces app.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url
            or os.environ.get("GRABETTE_API_URL")
            or "http://localhost:8000"
        )
        self._http = httpx.Client(base_url=self.base_url, timeout=10.0)

    # -- Camera --

    def get_snapshot(self) -> bytes | None:
        try:
            r = self._http.get("/api/camera/snapshot")
            r.raise_for_status()
            return r.content
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

    def get_daemon_status(self) -> dict | None:
        try:
            r = self._http.get("/api/daemon/status")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # -- Capture --

    def start_capture(self) -> dict:
        try:
            r = self._http.post("/api/episodes/start")
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

    # -- Sessions --

    def list_sessions(self) -> list[dict]:
        try:
            r = self._http.get("/api/sessions")
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def create_session(self, name: str, description: str = "") -> dict:
        try:
            r = self._http.post(
                "/api/sessions",
                json={"name": name, "description": description},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def update_session(self, session_id: str, name: str | None = None, description: str | None = None) -> dict:
        body = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        try:
            r = self._http.put(f"/api/sessions/{session_id}", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def delete_session(self, session_id: str) -> dict:
        try:
            r = self._http.delete(f"/api/sessions/{session_id}")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

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

    def download_episode(self, episode_id: str) -> str | None:
        try:
            r = self._http.get(
                f"/api/episodes/{episode_id}/download",
                timeout=60.0,
            )
            r.raise_for_status()
            path = os.path.join(
                tempfile.gettempdir(), f"{episode_id}.tar.gz"
            )
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        except Exception:
            return None

    def download_episodes(self, episode_ids: list[str]) -> str | None:
        try:
            r = self._http.post(
                "/api/episodes/download",
                json={"episode_ids": episode_ids},
                timeout=120.0,
            )
            r.raise_for_status()
            filename = "episodes.tar.gz" if len(episode_ids) > 1 else f"{episode_ids[0]}.tar.gz"
            path = os.path.join(tempfile.gettempdir(), filename)
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        except Exception:
            return None

    def move_episodes(self, episode_ids: list[str], target_session_id: str) -> dict:
        try:
            r = self._http.post(
                "/api/episodes/move",
                json={"episode_ids": episode_ids, "target_session_id": target_session_id},
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

    # -- HuggingFace --

    def hf_check_auth(self) -> dict:
        try:
            r = self._http.get("/api/hf/auth")
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"authenticated": False}

    def hf_set_auth(self, token: str) -> dict:
        try:
            r = self._http.post("/api/hf/auth", json={"token": token})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"authenticated": False, "error": detail}
        except Exception as e:
            return {"authenticated": False, "error": str(e)}

    def hf_upload_episode(self, episode_id: str, repo_id: str) -> dict:
        try:
            r = self._http.post(
                f"/api/hf/upload/{episode_id}",
                json={"repo_id": repo_id},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def hf_get_job(self, job_id: str) -> dict | None:
        try:
            r = self._http.get(f"/api/hf/jobs/{job_id}")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def hf_list_jobs(self) -> list[dict]:
        try:
            r = self._http.get("/api/hf/jobs")
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

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

    # -- SLAM --

    def slam_run(self, episode_id: str, repo_id: str) -> dict:
        try:
            r = self._http.post(
                f"/api/hf/slam/{episode_id}",
                json={"repo_id": repo_id},
                timeout=30.0,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}
