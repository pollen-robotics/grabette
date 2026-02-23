"""Thin HTTP client wrapping the grabette REST API.

Standalone copy for HuggingFace Spaces deployment (no grabette package deps).
Keep in sync with grabette/ui/api_client.py.
"""

from __future__ import annotations

import os
import tempfile

import httpx


class GrabetteClient:
    """Synchronous client for the grabette REST API."""

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
            r = self._http.post("/api/sessions/start")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    def stop_capture(self) -> dict:
        try:
            r = self._http.post("/api/sessions/stop")
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

    def download_session(self, session_id: str) -> str | None:
        try:
            r = self._http.get(
                f"/api/sessions/{session_id}/download", timeout=60.0,
            )
            r.raise_for_status()
            path = os.path.join(tempfile.gettempdir(), f"{session_id}.tar.gz")
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        except Exception:
            return None

    def delete_session(self, session_id: str) -> dict:
        try:
            r = self._http.delete(f"/api/sessions/{session_id}")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
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
            return {"authenticated": False, "error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"authenticated": False, "error": str(e)}

    def hf_upload_session(self, session_id: str, repo_id: str) -> dict:
        try:
            r = self._http.post(
                f"/api/hf/upload/{session_id}", json={"repo_id": repo_id},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    def hf_list_jobs(self) -> list[dict]:
        try:
            r = self._http.get("/api/hf/jobs")
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    # -- SLAM --

    def slam_run(self, session_id: str, repo_id: str) -> dict:
        try:
            r = self._http.post(
                f"/api/hf/slam/{session_id}",
                json={"repo_id": repo_id},
                timeout=30.0,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}
