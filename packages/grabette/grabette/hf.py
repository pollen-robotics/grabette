"""HuggingFace Hub integration for episode upload and cloud SLAM."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self) -> None:
        self._api = None
        self._cached_token: str | None = None

    def set_token(self, token: str) -> None:
        """Persist token to the standard HF token file (or clear it)."""
        self._api = None
        self._cached_token = None
        if token:
            from huggingface_hub.constants import HF_TOKEN_PATH

            token_path = Path(HF_TOKEN_PATH)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token)
        else:
            try:
                from huggingface_hub import logout

                logout()
            except Exception:  # noqa: BLE001
                pass

    @property
    def is_authenticated(self) -> bool:
        from huggingface_hub import get_token

        if not get_token():
            return False
        try:
            self._get_api()
            return True
        except Exception:
            return False

    def _get_api(self):
        from huggingface_hub import HfApi, get_token

        token = get_token()
        if token != self._cached_token:
            self._api = None
            self._cached_token = token
        if self._api is None:
            if not token:
                raise ValueError("No token available")
            self._api = HfApi(token=token)
            self._api.whoami()
        return self._api

    def get_user_info(self) -> dict | None:
        try:
            api = self._get_api()
            info = api.whoami()
            username = info.get("name", "")
            orgs = [o["name"] for o in info.get("orgs", []) if o.get("name")]
            namespaces = [username] + orgs if username else orgs
            return {"username": username, "email": info.get("email", ""), "namespaces": namespaces}
        except Exception:
            return None

    def upload_episode(
        self,
        episode_dir: Path,
        repo_id: str,
        progress_callback=None,
    ) -> str:
        """Upload an episode directory to HuggingFace Hub.

        Args:
            episode_dir: Path to episode directory containing raw_video.mp4 + imu_data.json
            repo_id: HuggingFace repo ID (e.g., "username/grabette-data")
            progress_callback: Optional callable(percent: float, message: str)

        Returns:
            URL of the uploaded data on HuggingFace Hub.
        """
        api = self._get_api()
        episode_id = episode_dir.name

        if progress_callback:
            progress_callback(0.0, "Creating repository...")

        # Create repo if it doesn't exist
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

        if progress_callback:
            progress_callback(10.0, "Uploading files...")

        # Upload the episode directory
        api.upload_folder(
            folder_path=str(episode_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=episode_id,
        )

        if progress_callback:
            progress_callback(100.0, "Upload complete")

        url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{episode_id}"
        logger.info("Episode %s uploaded to %s", episode_id, url)
        return url

    def delete_dataset(self, repo_id: str) -> None:
        api = self._get_api()
        api.delete_repo(repo_id, repo_type="dataset")
        logger.info("Deleted dataset %s", repo_id)
