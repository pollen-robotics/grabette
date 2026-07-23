"""SLAM job orchestration — upload raw dataset, trigger Space, poll, delete raw."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import httpx

from grabette.jobs import get_job_manager

logger = logging.getLogger(__name__)

_SLAM_SPACE_URL = os.environ.get(
    "GRABETTE_SLAM_SPACE_URL",
    "https://pollen-robotics-grabette-slam.hf.space",
).rstrip("/")

# Repo id used to read/restart the Space runtime (cannot be reliably derived from
# the .hf.space URL because org names may contain hyphens).
_SLAM_SPACE_REPO = os.environ.get(
    "GRABETTE_SLAM_SPACE_REPO",
    "pollen-robotics/grabette-slam",
)

# How long we wait for a dormant Space to come back up before giving up.
_SPACE_WAKE_TIMEOUT = float(os.environ.get("GRABETTE_SLAM_WAKE_TIMEOUT", "600"))

# HF returns these when a Space is asleep / still starting.
_SLEEPY_STATUS = {502, 503, 504}
# SpaceStage values that mean "the app is serving requests".
_READY_STAGES = {"RUNNING", "RUNNING_BUILDING"}
# SpaceStage values that mean the Space is broken and will never wake on its own.
_FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


class SlamOrchestrator:
    def __init__(self) -> None:
        self._jm = get_job_manager()

    @staticmethod
    def _space_stage(token: str) -> str | None:
        """Current SpaceStage (e.g. 'RUNNING', 'SLEEPING'), or None if unreadable.

        Unreadable means the token lacks access or the Hub is unreachable — in that
        case we fall back to detecting sleep from the 503 on the actual request.
        """
        from huggingface_hub import get_space_runtime
        try:
            return get_space_runtime(_SLAM_SPACE_REPO, token=token or None).stage
        except Exception as e:  # noqa: BLE001 — best-effort, never fatal
            logger.warning("Could not read SLAM Space runtime: %s", e)
            return None

    async def _wake_space(self, token: str, stage: str | None) -> None:
        """Best-effort nudge to bring a dormant Space back up.

        A SLEEPING Space wakes on incoming traffic, so a plain GET is enough.
        PAUSED/STOPPED Spaces need an explicit restart (requires write access).
        """
        if stage in ("PAUSED", "STOPPED"):
            try:
                from huggingface_hub import restart_space
                await asyncio.to_thread(
                    restart_space, _SLAM_SPACE_REPO, token=token or None
                )
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not restart SLAM Space: %s", e)
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                await http.get(_SLAM_SPACE_URL)  # traffic wakes a sleeping Space
        except Exception:  # noqa: BLE001 — the wait loop below will keep polling
            pass

    async def _trigger_slam(self, job_id: str, token: str, payload: dict) -> str:
        """POST /api/process, waking the Space and waiting for it if it's dormant.

        Returns the Space-side job id. Raises only if the Space stays unreachable
        past _SPACE_WAKE_TIMEOUT or is in a permanently failed stage — a dormant
        Space is no longer treated as a hard failure.
        """
        _headers = {"Authorization": f"Bearer {token}"} if token else {}
        deadline = time.monotonic() + _SPACE_WAKE_TIMEOUT
        notified = False

        async with httpx.AsyncClient(timeout=30, headers=_headers) as http:
            while True:
                stage = await asyncio.to_thread(self._space_stage, token)
                if stage in _FAILED_STAGES:
                    raise RuntimeError(
                        f"SLAM Space is in a failed state (stage={stage}); "
                        "check the Space logs on Hugging Face"
                    )
                if stage is not None and stage not in _READY_STAGES:
                    # Dormant / still starting — notify once, nudge, then wait.
                    if not notified:
                        self._jm.update_progress(
                            job_id, 52.0,
                            "SLAM Space is asleep — waking it up, this can take a "
                            "minute…",
                        )
                        await self._wake_space(token, stage)
                        notified = True
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"SLAM Space did not wake within "
                            f"{_SPACE_WAKE_TIMEOUT:.0f}s (last stage={stage})"
                        )
                    self._jm.update_progress(
                        job_id, 53.0, f"Waking SLAM Space… (stage={stage})"
                    )
                    await asyncio.sleep(5)
                    continue

                # Stage RUNNING, or runtime unreadable → attempt the call.
                try:
                    r = await http.post(f"{_SLAM_SPACE_URL}/api/process", json=payload)
                    r.raise_for_status()
                    if notified:
                        self._jm.update_progress(
                            job_id, 54.0, "SLAM Space is up — starting…"
                        )
                    return r.json()["job_id"]
                except httpx.HTTPStatusError as e:
                    if e.response.status_code not in _SLEEPY_STATUS:
                        raise
                except (
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.ReadTimeout,
                    httpx.RemoteProtocolError,
                ):
                    pass  # transient — Space likely waking; retry below

                # Got a sleep/transient signal from the request itself.
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"SLAM Space stayed unreachable for "
                        f"{_SPACE_WAKE_TIMEOUT:.0f}s"
                    )
                if not notified:
                    self._jm.update_progress(
                        job_id, 52.0,
                        "SLAM Space is asleep — waking it up, this can take a "
                        "minute…",
                    )
                    await self._wake_space(token, None)
                    notified = True
                else:
                    self._jm.update_progress(
                        job_id, 53.0, "Waking SLAM Space…"
                    )
                await asyncio.sleep(5)

    async def push_and_process(
        self,
        task_ids: list[str],
        raw_repo: str,
        target_repo: str,
        task_description: str,
        hf_client,
        session_manager,
        private: bool = False,
    ) -> str:
        """Upload all episodes from task_ids to raw_repo, trigger SLAM, poll, delete raw.

        Returns the job_id for tracking via /api/hf/jobs/{job_id}.
        """
        job = self._jm.create_job(f"push:{target_repo}")
        job_id = job.job_id

        async def _run() -> None:
            try:
                # Collect episode dirs for all selected tasks
                sessions = session_manager.list_sessions()
                session_map = {s.id: s for s in sessions}
                episode_dirs: list[Path] = []
                for tid in task_ids:
                    s = session_map.get(tid)
                    if not s:
                        continue
                    for ep in s.episodes:
                        ep_dir = session_manager.episode_dir(ep.episode_id)
                        episode_dirs.append(ep_dir)

                if not episode_dirs:
                    raise ValueError("No episodes found for the selected tasks")

                total = len(episode_dirs)
                self._jm.update_progress(job_id, 2.0, f"Uploading {total} episode(s)…")

                # Upload episodes one by one to the raw repo
                for i, ep_dir in enumerate(episode_dirs):
                    pct_base = 2.0 + (i / total) * 48.0

                    def _progress(pct: float, msg: str, base: float = pct_base) -> None:
                        self._jm.update_progress(job_id, base + pct * 0.48, msg)

                    await asyncio.to_thread(
                        hf_client.upload_episode, ep_dir, raw_repo, _progress, private,
                    )
                    self._jm.update_progress(
                        job_id, 2.0 + ((i + 1) / total) * 48.0,
                        f"Uploaded {i + 1}/{total}",
                    )

                # Trigger SLAM Space (waking it first if it's dormant)
                self._jm.update_progress(job_id, 52.0, "Triggering SLAM pipeline…")
                from huggingface_hub import get_token
                token = get_token() or ""
                _headers = {"Authorization": f"Bearer {token}"} if token else {}
                slam_job_id = await self._trigger_slam(
                    job_id,
                    token,
                    {
                        "source_repo": raw_repo,
                        "target_repo": target_repo,
                        "task": task_description,
                        "private": private,
                    },
                )

                # Poll Space status
                self._jm.update_progress(job_id, 55.0, "SLAM processing…")
                async with httpx.AsyncClient(timeout=10, headers=_headers) as http:
                    while True:
                        await asyncio.sleep(5)
                        try:
                            r = await http.get(
                                f"{_SLAM_SPACE_URL}/api/status/{slam_job_id}"
                            )
                            r.raise_for_status()
                            status = r.json()
                        except Exception:
                            continue
                        slam_status = status.get("status", "running")
                        log_tail = (status.get("log") or "").split("\n")
                        last_line = next(
                            (l for l in reversed(log_tail) if l.strip()), ""
                        )
                        if slam_status == "done":
                            result_url = status.get("result") or \
                                f"https://huggingface.co/datasets/{target_repo}"
                            self._jm.update_progress(job_id, 95.0, "Cleaning up…")
                            # Delete raw dataset
                            try:
                                await asyncio.to_thread(
                                    hf_client.delete_dataset, raw_repo
                                )
                            except Exception as e:
                                logger.warning("Could not delete raw dataset: %s", e)
                            self._jm.complete_job(job_id, result_url)
                            return
                        elif slam_status == "error":
                            raise RuntimeError(
                                status.get("error") or "SLAM pipeline failed"
                            )
                        elif slam_status == "not_found":
                            raise RuntimeError("SLAM job not found on Space")
                        else:
                            # Map Space progress (0→100) into 55→93 range
                            if last_line:
                                self._jm.update_progress(
                                    job_id, 55.0, f"SLAM: {last_line}"
                                )

            except Exception as e:
                logger.exception("push-and-process job %s failed: %s", job_id, e)
                self._jm.fail_job(job_id, str(e))

        asyncio.create_task(_run())
        return job_id


_slam_orchestrator = SlamOrchestrator()


def get_slam_orchestrator() -> SlamOrchestrator:
    return _slam_orchestrator
