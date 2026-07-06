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

    def get_daemon_status(self) -> dict | None:
        try:
            r = self._http.get("/api/daemon/status")
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

    def start_teleop(self) -> dict:
        try:
            r = self._http.post("/api/teleop/start")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    def stop_teleop(self) -> dict:
        try:
            r = self._http.post("/api/teleop/stop")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

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

    def start_capture(self, session_id: str | None = None) -> dict:
        try:
            body = {"session_id": session_id} if session_id else {}
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

    def get_capture_session_status(self) -> dict:
        try:
            r = self._http.get("/api/capture-session/status")
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"active": False, "task_id": None, "task_name": None, "count": 0}

    def start_capture_session(self, task_id: str | None = None) -> dict:
        try:
            body = {"task_id": task_id} if task_id else {}
            r = self._http.post("/api/capture-session/start", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    def stop_capture_session(self) -> dict:
        try:
            r = self._http.post("/api/capture-session/stop")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_active_session(self) -> str | None:
        try:
            r = self._http.get("/api/sessions/active")
            r.raise_for_status()
            return r.json().get("session_id")
        except Exception:
            return None

    def set_active_session(self, session_id: str) -> dict:
        try:
            r = self._http.put("/api/sessions/active", json={"session_id": session_id})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
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

    def shutdown(self) -> dict:
        try:
            r = self._http.post("/api/system/shutdown")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    # -- HuggingFace --

    def hf_check_auth(self) -> dict:
        try:
            r = self._http.get("/api/hf/auth")
            r.raise_for_status()
            return r.json() or {"authenticated": False}
        except Exception:
            return {"authenticated": False}

    def hf_get_namespaces(self) -> list[str]:
        """Return available namespaces (username + orgs) for the authenticated user."""
        result = self.hf_check_auth()
        if not result.get("authenticated"):
            return []
        return (result.get("user") or {}).get("namespaces", [])

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

    def hf_push_and_process(
        self,
        task_ids: list[str],
        target_repo: str,
        raw_repo: str,
        task_description: str,
        private: bool = False,
    ) -> dict:
        try:
            r = self._http.post(
                "/api/hf/push",
                json={
                    "task_ids": task_ids,
                    "target_repo": target_repo,
                    "raw_repo": raw_repo,
                    "task_description": task_description,
                    "private": private,
                },
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
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            return {"status": "running", "message": "poll error, retrying…", "progress": 0}
        except Exception:
            return {"status": "running", "message": "poll error, retrying…", "progress": 0}

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

    # -- WiFi --

    def wifi_status(self) -> dict:
        try:
            r = self._http.get("/api/wifi/status", timeout=3.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"mode": "offline", "ssid": None, "ip": None}

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
