"""SLAM job orchestration — upload raw dataset, trigger Space, poll, delete raw."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import httpx

from grabette.jobs import get_job_manager

logger = logging.getLogger(__name__)

_SLAM_SPACE_URL = os.environ.get(
    "GRABETTE_SLAM_SPACE_URL",
    "https://pollen-robotics-grabette-slam.hf.space",
).rstrip("/")


class SlamOrchestrator:
    def __init__(self) -> None:
        self._jm = get_job_manager()

    async def push_and_process(
        self,
        task_ids: list[str],
        raw_repo: str,
        target_repo: str,
        task_description: str,
        hf_client,
        session_manager,
        exclude_fail: bool = False,
        exclude_bad: bool = False,
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
                        hf_client.upload_episode, ep_dir, raw_repo, _progress,
                    )
                    self._jm.update_progress(
                        job_id, 2.0 + ((i + 1) / total) * 48.0,
                        f"Uploaded {i + 1}/{total}",
                    )

                # Trigger SLAM Space
                self._jm.update_progress(job_id, 52.0, "Triggering SLAM pipeline…")
                from huggingface_hub import get_token
                token = get_token() or ""
                _headers = {"Authorization": f"Bearer {token}"} if token else {}
                async with httpx.AsyncClient(timeout=30, headers=_headers) as http:
                    r = await http.post(
                        f"{_SLAM_SPACE_URL}/api/process",
                        json={
                            "source_repo": raw_repo,
                            "target_repo": target_repo,
                            "task": task_description,
                            "exclude_fail": exclude_fail,
                            "exclude_bad": exclude_bad,
                        },
                    )
                    r.raise_for_status()
                    slam_job_id = r.json()["job_id"]

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
                            quality = status.get("quality")
                            self._jm.update_progress(job_id, 95.0, "Cleaning up…")
                            # Delete raw dataset
                            try:
                                await asyncio.to_thread(
                                    hf_client.delete_dataset, raw_repo
                                )
                            except Exception as e:
                                logger.warning("Could not delete raw dataset: %s", e)
                            self._jm.complete_job(job_id, result_url, quality=quality)
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
