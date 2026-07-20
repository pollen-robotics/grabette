"""Unit tests for grabette.jobs.JobManager — background-job state machine.

Long uploads/conversions surface progress to the UI through JobManager. The
PENDING → RUNNING → COMPLETED/FAILED transitions and progress bookkeeping are
what the UI polls, so they need to behave exactly.
"""

from grabette.jobs import JobManager, JobStatus


def test_create_job_starts_pending_with_unique_ids():
    """A new job starts PENDING at 0%, and each gets a distinct id tracked by the manager."""
    mgr = JobManager()
    a = mgr.create_job("upload")
    b = mgr.create_job("convert")
    assert a.status == JobStatus.PENDING
    assert a.progress == 0.0
    assert a.job_id != b.job_id
    assert {j.job_id for j in mgr.list_jobs()} == {a.job_id, b.job_id}


def test_update_progress_promotes_pending_to_running():
    """The first progress update flips PENDING → RUNNING and records progress/message."""
    mgr = JobManager()
    job = mgr.create_job("upload")
    mgr.update_progress(job.job_id, 42.0, "halfway")
    assert job.status == JobStatus.RUNNING
    assert job.progress == 42.0
    assert job.message == "halfway"


def test_complete_job_sets_result_and_full_progress():
    """Completion sets COMPLETED, 100% progress, and stores the result."""
    mgr = JobManager()
    job = mgr.create_job("upload")
    mgr.complete_job(job.job_id, "repo/dataset")
    assert job.status == JobStatus.COMPLETED
    assert job.progress == 100.0
    assert job.result == "repo/dataset"


def test_fail_job_records_error():
    """Failure sets FAILED and stores the error message."""
    mgr = JobManager()
    job = mgr.create_job("upload")
    mgr.fail_job(job.job_id, "network down")
    assert job.status == JobStatus.FAILED
    assert job.error == "network down"


def test_unknown_job_id_is_a_no_op():
    """Mutating an unknown job id is a safe no-op, and get_job returns None."""
    mgr = JobManager()
    # None of these should raise on a missing id.
    mgr.update_progress("nope", 10.0)
    mgr.complete_job("nope", "x")
    mgr.fail_job("nope", "x")
    assert mgr.get_job("nope") is None
