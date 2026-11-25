# Copied from grabette/hf.py (develop @089f1d6) for the casquette fleet
# prototype. Deviations: added get_hf_client() singleton (grabette keeps that in
# its app/routers/huggingface.py), no upload-heartbeat marks, xet disabled (see
# below). The client itself is generic (stdlib + huggingface_hub only). Keep in
# sync until extracted to a shared package.
"""HuggingFace Hub client — episode upload to a shared raw dataset."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Upload over plain LFS HTTP, not hf_xet. On a Pi Zero 2 W (416 MB RAM) hf_xet's
# parallel chunking/compression drove the process past available memory within
# ~10 s of an upload starting, and the kernel OOM-killed the whole service: the
# fleet saw the device go offline and the upload was lost. LFS streams each file
# from disk instead. Must be set before huggingface_hub is imported (it reads it
# into constants at import) — every import of it in casquette goes through this
# module, lazily. setdefault: HF_HUB_DISABLE_XET=0 in the env re-enables xet.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logger = logging.getLogger(__name__)

# Above this size, already_uploaded compares by size only.
_HASH_TO_SKIP_MAX_BYTES = 10 * 1024 * 1024


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

    def already_uploaded(self, episodes: list[tuple[Path, str]], repo_id: str) -> set[str]:
        """path_in_repo of the episodes whose files are all already on the Hub.

        Avoids re-hashing them in upload_episodes. Files above
        _HASH_TO_SKIP_MAX_BYTES are compared by size only.
        """
        import hashlib

        from huggingface_hub.utils import (DEFAULT_IGNORE_PATTERNS, RepositoryNotFoundError,
                                           filter_repo_objects)

        api = self._get_api()
        try:
            remote = {f.path: f for f in api.list_repo_tree(repo_id, repo_type="dataset",
                                                             recursive=True)
                      if hasattr(f, "blob_id")}   # files only, not RepoFolder
        except RepositoryNotFoundError:
            return set()

        def same(local: Path, entry) -> bool:
            size = local.stat().st_size
            if size != entry.size:
                return False
            if size > _HASH_TO_SKIP_MAX_BYTES:
                return True
            data = local.read_bytes()
            if entry.lfs is not None:
                return hashlib.sha256(data).hexdigest() == entry.lfs.sha256
            return hashlib.sha1(b"blob %d\0" % size + data).hexdigest() == entry.blob_id

        done = set()
        for episode_dir, dest in episodes:
            root = Path(episode_dir).expanduser().resolve()
            files = {p.relative_to(root).as_posix(): p
                     for p in sorted(root.glob("**/*")) if p.is_file()}
            rels = list(filter_repo_objects(files.keys(), ignore_patterns=DEFAULT_IGNORE_PATTERNS))
            prefix = dest.strip("/")
            if rels and all(f"{prefix}/{rel}" in remote
                            and same(files[rel], remote[f"{prefix}/{rel}"]) for rel in rels):
                done.add(dest)
        return done

    def upload_episodes(
        self,
        episodes: list[tuple[Path, str]],
        repo_id: str,
        private: bool = False,
        commit_message: str | None = None,
    ) -> None:
        """Upload (episode_dir, path_in_repo) pairs in a single commit (Hub commit quota)."""
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects

        api = self._get_api()
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

        operations = []
        for episode_dir, dest in episodes:
            root = Path(episode_dir).expanduser().resolve()
            files = {p.relative_to(root).as_posix(): p
                     for p in sorted(root.glob("**/*")) if p.is_file()}
            for rel in filter_repo_objects(files.keys(),
                                           ignore_patterns=DEFAULT_IGNORE_PATTERNS):
                operations.append(CommitOperationAdd(
                    path_in_repo=f"{dest.strip('/')}/{rel}", path_or_fileobj=str(files[rel])))

        dests = [dest for _, dest in episodes]
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message
            or f"Upload {len(dests)} episode(s): {', '.join(dests[:5])}"
            + ("…" if len(dests) > 5 else ""),
        )
        logger.info("Uploaded %d episode(s) to %s in one commit", len(dests), repo_id)


_hf_client: Optional[HuggingFaceClient] = None


def get_hf_client() -> HuggingFaceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = HuggingFaceClient()
    return _hf_client
