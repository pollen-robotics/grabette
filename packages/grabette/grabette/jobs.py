"""Background job tracker with progress reporting."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: str | None = None
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create_job(self, name: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, name=name)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def update_progress(self, job_id: str, progress: float, message: str = "") -> None:
        job = self._jobs.get(job_id)
        if job:
            job.progress = progress
            job.message = message
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.RUNNING

    def complete_job(self, job_id: str, result: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.result = result

    def fail_job(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = error


_job_manager = JobManager()


def get_job_manager() -> JobManager:
    return _job_manager
