"""SLAM job orchestration — upload session, trigger processing, poll results."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from grabette.jobs import get_job_manager

logger = logging.getLogger(__name__)


class SlamOrchestrator:
    """Orchestrate SLAM processing: upload -> trigger -> poll -> results."""

    def __init__(self) -> None:
        self._jm = get_job_manager()

    async def run_slam(
        self,
        episode_id: str,
        episode_dir: Path,
        repo_id: str,
        hf_client,
    ) -> str:
        """Start a SLAM job. Returns the job ID for tracking.

        Args:
            episode_id: Capture episode identifier.
            episode_dir: Local path to episode data.
            repo_id: HuggingFace dataset repo (e.g. "user/grabette-data").
            hf_client: Authenticated HuggingFaceClient instance.
        """
        job = self._jm.create_job(f"slam:{episode_id}")
        job_id = job.job_id

        async def _run():
            try:
                # Step 1: Upload episode to HF
                self._jm.update_progress(job_id, 5.0, "Uploading episode...")

                def progress_cb(pct: float, msg: str):
                    mapped = 5.0 + pct * 0.45  # 5% -> 50%
                    self._jm.update_progress(job_id, mapped, msg)

                url = await asyncio.to_thread(
                    hf_client.upload_episode, episode_dir, repo_id, progress_cb,
                )

                # Step 2: Trigger SLAM processing (placeholder)
                self._jm.update_progress(
                    job_id, 55.0, "Triggering SLAM processing...",
                )

                # TODO: Replace with actual SLAM trigger once compute is set up.
                # Options:
                #   - HF Inference Endpoint running ORB-SLAM3 Docker
                #   - Dedicated HF Space with SLAM processing
                #   - External API call to SLAM service
                logger.warning(
                    "SLAM compute not yet configured. "
                    "Episode uploaded to: %s", url,
                )

                self._jm.update_progress(
                    job_id, 90.0,
                    "Episode uploaded; SLAM compute not yet configured",
                )
                self._jm.complete_job(job_id, url)

            except Exception as e:
                logger.exception("SLAM job %s failed: %s", job_id, e)
                self._jm.fail_job(job_id, str(e))

        asyncio.create_task(_run())
        return job_id


_slam_orchestrator = SlamOrchestrator()


def get_slam_orchestrator() -> SlamOrchestrator:
    return _slam_orchestrator
