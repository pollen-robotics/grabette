# Copied from grabette/hf.py (develop @089f1d6) for the casquette fleet
# prototype. Deviation: added get_hf_client() singleton (grabette keeps that in
# its app/routers/huggingface.py). The client itself is generic (stdlib +
# huggingface_hub only). Keep in sync until extracted to a shared package.
"""HuggingFace Hub client — episode upload to a shared raw dataset."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Must match RAW_DOWNLOAD_NAME_RE in grabette-fleet app.py.
RAW_DOWNLOAD_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*-rawdl-\d{8}-\d{6}-[0-9a-f]{8}")


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self) -> None:
        self._api = None
        self._cached_token: str | None = None

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
            api = HfApi(token=token)
            api.whoami()
            self._api = api
        return self._api

    def upload_episode(
        self,
        episode_dir: Path,
        repo_id: str,
        progress_callback=None,
        path_in_repo: str | None = None,
        private: bool = False,
    ) -> str:
        """Upload an episode directory to a HuggingFace dataset repo.

        path_in_repo defaults to the episode id (the dir name). For a
        multi-device raw dataset pass "{episode_id}/{role}" so each device's
        stream for the SAME episode lands in its own subfolder.
        """
        api = self._get_api()
        dest = path_in_repo or episode_dir.name

        # Create repo if missing. `private` takes effect only on first create
        # (exist_ok=True won't flip an existing repo); all a job's devices pass
        # the same value, so whichever creates it first sets it correctly.
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(episode_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=dest,
        )
        url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{dest}"
        logger.info("Episode %s uploaded to %s", episode_dir.name, url)
        return url

    def delete_raw_download_repos(self, repos: list[str]) -> dict:
        """Delete the fleet's raw-download repos — only this user's, with the exact
        download name shape, whatever the fleet asks."""
        api = self._get_api()
        user = api.whoami().get("name", "")
        out: dict = {"deleted": [], "refused": [], "failed": []}
        for repo in repos:
            ns, _, name = str(repo).partition("/")
            if ns != user or not RAW_DOWNLOAD_NAME_RE.fullmatch(name):
                out["refused"].append(repo)
                logger.warning("Refusing to delete %s: not a raw download of %s", repo, user)
                continue
            try:
                api.delete_repo(repo, repo_type="dataset", missing_ok=True)
                out["deleted"].append(repo)
                logger.info("Deleted raw download %s", repo)
            except Exception as e:  # noqa: BLE001
                out["failed"].append({"repo": repo, "error": str(e)})
                logger.warning("Could not delete raw download %s: %s", repo, e)
        return out


_hf_client: Optional[HuggingFaceClient] = None


def get_hf_client() -> HuggingFaceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = HuggingFaceClient()
    return _hf_client
