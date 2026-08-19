"""HuggingFace Hub integration for episode upload and cloud SLAM."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# --- socket timeouts on every Hub call ----------------------------------------
# huggingface_hub builds its httpx client with timeout=None — no connect, read or
# write deadline at any layer. A link that DEGRADES rather than drops therefore
# leaves an upload waiting on a socket that never delivers, forever, inside a
# thread nothing can interrupt. That is the whole reason a grabette could stay
# pinned as "uploading" until the fleet Space was restarted.
#
# These are per-SOCKET-OPERATION budgets, not deadlines for the whole transfer:
# httpx counts them between chunks, so a legitimately slow multi-gigabyte upload
# over a weak link keeps running as long as it keeps making progress, and only a
# genuine stall trips them. When one does trip, it surfaces as
# httpx.TimeoutException INSIDE the worker thread — which hf_hub's own
# http_backoff already retries on, and which lets the thread DIE instead of
# lingering. That distinction is the point: an abandoned upload thread is far
# worse on a Pi than a slow one (see _UPLOAD_EXECUTOR in app/main.py).
_HUB_CONNECT_TIMEOUT_S = 20.0
_HUB_READ_TIMEOUT_S = 60.0
_HUB_WRITE_TIMEOUT_S = 60.0   # no progress writing a chunk for this long = stalled
_HUB_POOL_TIMEOUT_S = 20.0

_client_factory_lock = threading.Lock()
_client_factory_installed = False


def install_hub_timeouts() -> None:
    """Give every huggingface_hub call a socket timeout. Idempotent, best effort.

    set_client_factory is process-wide and public API. Called from _get_api so it
    covers every Hub user on the device (episode upload, SLAM push, whoami)
    without each of them having to remember."""
    global _client_factory_installed
    with _client_factory_lock:
        if _client_factory_installed:
            return
        try:
            import httpx
            from huggingface_hub import set_client_factory
            from huggingface_hub.utils._http import hf_request_event_hook

            timeout = httpx.Timeout(
                connect=_HUB_CONNECT_TIMEOUT_S, read=_HUB_READ_TIMEOUT_S,
                write=_HUB_WRITE_TIMEOUT_S, pool=_HUB_POOL_TIMEOUT_S,
            )
            set_client_factory(lambda: httpx.Client(
                event_hooks={"request": [hf_request_event_hook]},
                follow_redirects=True, timeout=timeout,
            ))
            logger.info("Hub client timeouts installed (connect=%.0fs read=%.0fs "
                        "write=%.0fs)", _HUB_CONNECT_TIMEOUT_S, _HUB_READ_TIMEOUT_S,
                        _HUB_WRITE_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — never block Hub access over this
            # A future hf_hub could drop set_client_factory or move the event
            # hook. Losing the timeouts is a regression, not an outage: uploads
            # still work, they are just unbounded again (and app/main.py's own
            # attempt bound remains the backstop).
            logger.warning("Could not install Hub client timeouts: %s", e)
        _client_factory_installed = True


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

        install_hub_timeouts()
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
        path_in_repo: str | None = None,
        private: bool = False,
    ) -> str:
        """Upload an episode directory to HuggingFace Hub.

        Args:
            episode_dir: Path to the episode directory (the entire folder is uploaded)
            repo_id: HuggingFace repo ID (e.g., "username/grabette-data")
            progress_callback: Optional callable(percent: float, message: str)
            path_in_repo: Destination path inside the repo. Defaults to the
                episode id (the dir name). For a multi-device raw dataset pass
                "{episode_id}/{role}" so each device's stream for the SAME
                episode lands in its own subfolder instead of colliding.
            private: Whether the repository should be created as private

        Returns:
            URL of the uploaded data on HuggingFace Hub.
        """
        api = self._get_api()
        dest = path_in_repo or episode_dir.name

        if progress_callback:
            progress_callback(0.0, "Creating repository...")

        # Create repo if it doesn't exist. `private` takes effect only on the
        # FIRST create (exist_ok=True won't flip an existing repo's visibility);
        # since all of a job's devices pass the same value, whichever creates it
        # first sets it correctly.
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

        if progress_callback:
            progress_callback(10.0, "Uploading files...")

        # Upload the episode directory
        api.upload_folder(
            folder_path=str(episode_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=dest,
        )

        if progress_callback:
            progress_callback(100.0, "Upload complete")

        url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{dest}"
        logger.info("Episode %s uploaded to %s", episode_dir.name, url)
        return url
