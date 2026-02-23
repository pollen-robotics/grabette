"""HuggingFace Hub integration for session upload and cloud SLAM."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._api = None

    def set_token(self, token: str) -> None:
        self._token = token
        self._api = None  # Reset API client

    @property
    def is_authenticated(self) -> bool:
        if not self._token:
            return False
        try:
            self._get_api()
            return True
        except Exception:
            return False

    def _get_api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self._token)
            # Verify token by calling whoami
            self._api.whoami()
        return self._api

    def get_user_info(self) -> dict | None:
        try:
            api = self._get_api()
            info = api.whoami()
            return {"username": info.get("name", ""), "email": info.get("email", "")}
        except Exception:
            return None

    def upload_session(
        self,
        session_dir: Path,
        repo_id: str,
        progress_callback=None,
    ) -> str:
        """Upload a session directory to HuggingFace Hub.

        Args:
            session_dir: Path to session directory containing raw_video.mp4 + imu_data.json
            repo_id: HuggingFace repo ID (e.g., "username/grabette-data")
            progress_callback: Optional callable(percent: float, message: str)

        Returns:
            URL of the uploaded data on HuggingFace Hub.
        """
        api = self._get_api()
        session_id = session_dir.name

        if progress_callback:
            progress_callback(0.0, "Creating repository...")

        # Create repo if it doesn't exist
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

        if progress_callback:
            progress_callback(10.0, "Uploading files...")

        # Upload the session directory
        api.upload_folder(
            folder_path=str(session_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=f"sessions/{session_id}",
        )

        if progress_callback:
            progress_callback(100.0, "Upload complete")

        url = f"https://huggingface.co/datasets/{repo_id}/tree/main/sessions/{session_id}"
        logger.info("Session %s uploaded to %s", session_id, url)
        return url
