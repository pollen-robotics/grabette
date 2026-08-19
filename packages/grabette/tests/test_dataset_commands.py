"""Relay dataset commands: upload_episodes + the Space result the fleet trusts.

Both incidents these cover were reported from the field:

  1. A conversion where every episode lacked oakd_calib_offline.json finished as
     "done" with no dataset. The device reported it as a success with no
     result_url, the fleet filled the blank with a guessed repo URL, and the
     operator got a green "Open <dataset>" link onto a 404.

  2. An upload over a degraded link never came back. The command stayed in the
     fleet's queue for that device, which therefore read as "uploading" to every
     operator, through reboots, until the fleet Space itself was restarted.

So: a Space that pushed nothing must be an error, and an upload attempt must be
bounded and retried rather than allowed to hang.
"""
import asyncio
import threading
import time

import pytest

from grabette.app import main


# --- 1. the Space said "done" but pushed nothing -----------------------------

def test_quality_summary_names_the_dominant_reason():
    quality = [
        {"name": "a", "errors": ["missing oakd_calib_offline.json"]},
        {"name": "b", "errors": ["missing oakd_calib_offline.json"]},
        {"name": "c", "errors": ["SLAM failed: no trajectory produced"]},
    ]

    summary = main._space_quality_summary(quality)

    # Counted, not listed, and the most frequent reason leads.
    assert summary.startswith("missing oakd_calib_offline.json (2 episode(s))")
    assert "SLAM failed: no trajectory produced (1 episode(s))" in summary


def test_excluded_episodes_are_split_back_into_recording_and_arm():
    # The Space labels a role-layout episode "20250101_120000/left". The fleet
    # keys everything by episode id, and "which arm" is what tells the operator
    # which grabette to go and look at — so the label has to come apart.
    out = main._space_excluded_episodes([
        {"name": "20250101_120000/left", "excluded": True,
         "errors": ["missing oakd_calib_offline.json"]},
        {"name": "20250101_120100", "excluded": True, "verdict": "FAIL", "errors": []},
        {"name": "20250101_120200/right", "excluded": False, "errors": ["a warning"]},
    ])

    assert out[0] == {"episode_id": "20250101_120000", "role": "left",
                      "reason": "missing oakd_calib_offline.json"}
    # No error text: still nameable, via the verdict.
    assert out[1]["episode_id"] == "20250101_120100" and out[1]["role"] == ""
    assert "FAIL" in out[1]["reason"]
    # Kept episodes are not exclusions, however flagged they are.
    assert len(out) == 2


def test_excluded_episodes_tolerate_junk():
    assert main._space_excluded_episodes(None) == []
    assert main._space_excluded_episodes(["nope", {"excluded": True}]) == []


def test_a_partial_conversion_forwards_the_episode_list(monkeypatch):
    # The count the operator was missing is built from this list, so a successful
    # build must carry it and not only the summary sentence.
    monkeypatch.setattr(main, "_wake_space", _awake)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "tok")

    res = _process_result({
        "status": "done", "result": "https://hf.co/datasets/u/ds",
        "quality": [{"name": "ep1/left", "excluded": True,
                     "errors": ["missing oakd_calib_offline.json"]}],
    })

    assert res["status"] == "ok"
    assert res["excluded"] == [{"episode_id": "ep1", "role": "left",
                               "reason": "missing oakd_calib_offline.json"}]


def test_quality_summary_tolerates_junk():
    # The Space's payload is data from another service; a shape change must not
    # turn a reportable failure into an exception inside the error path.
    assert main._space_quality_summary(None) == ""
    assert main._space_quality_summary([]) == ""
    assert main._space_quality_summary(["nope", {"errors": []}]) == ""


def _process_result(status_payload):
    """Drive process_dataset against a Space serving one canned /api/status."""
    class _Resp:
        def __init__(self, payload):
            self.status = 200
            self._payload = payload

        async def json(self):
            return self._payload

        async def text(self):
            return ""

        async def read(self):
            return b""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def get(self, url, **kw):
            return _Resp(status_payload)

        def post(self, url, **kw):
            return _Resp({"job_id": "job1"})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    import aiohttp
    orig_session, orig_sleep = aiohttp.ClientSession, asyncio.sleep
    orig_daemon = main._daemon
    main._daemon = object()  # process_dataset only needs the gate to pass
    aiohttp.ClientSession = lambda *a, **kw: _Session()
    asyncio.sleep = lambda *_a, **_kw: orig_sleep(0)
    try:
        return asyncio.run(main._handle_relay_command({
            "id": "cmd1", "type": "process_dataset",
            "args": {"space_url": "https://space.test", "source_repo": "u/raw",
                     "target_repo": "u/ds"},
        }))
    finally:
        aiohttp.ClientSession, asyncio.sleep = orig_session, orig_sleep
        main._daemon = orig_daemon


def test_done_without_a_dataset_is_an_error_not_a_success(monkeypatch):
    # THE regression: "done" + result=None is what produced the 404 link. The
    # device must not hand the fleet a success it can paper over.
    monkeypatch.setattr(main, "_wake_space", _awake)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "tok")

    res = _process_result({"status": "done", "result": None, "quality": [
        {"name": "20250101_120000/left", "errors": ["missing oakd_calib_offline.json"]},
    ]})

    assert res["status"] == "error"
    assert "no episode made it through" in res["message"].lower()
    # And it carries the reason, so the operator doesn't have to open the Space.
    assert "missing oakd_calib_offline.json" in res["message"]


def test_done_with_a_dataset_is_still_a_success(monkeypatch):
    monkeypatch.setattr(main, "_wake_space", _awake)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "tok")

    res = _process_result({"status": "done", "result": "https://hf.co/datasets/u/ds"})

    assert res == {"status": "ok", "result_url": "https://hf.co/datasets/u/ds"}


def test_partial_success_reports_what_was_excluded(monkeypatch):
    monkeypatch.setattr(main, "_wake_space", _awake)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "tok")

    res = _process_result({
        "status": "done", "result": "https://hf.co/datasets/u/ds",
        "quality": [{"name": "b", "errors": ["missing oakd_calib_offline.json"]}],
    })

    assert res["status"] == "ok"
    assert "excluded" in res["message"]


def test_space_error_is_enriched_with_the_reasons(monkeypatch):
    monkeypatch.setattr(main, "_wake_space", _awake)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "tok")

    res = _process_result({
        "status": "error", "error": "No recording has all the arms (left+right)",
        "quality": [{"name": "a/left", "errors": ["missing oakd_calib_offline.json"]}],
    })

    assert res["status"] == "error"
    assert "No recording has all the arms" in res["message"]
    assert "missing oakd_calib_offline.json" in res["message"]


async def _awake(*a, **kw):
    return None  # the Space is up


# --- 2. a bounded, retried episode upload ------------------------------------

class _Hf:
    """Records upload attempts; fails the first `fail_times` of them."""

    def __init__(self, fail_times=0, hang=False):
        self.calls = []
        self._fail_times = fail_times
        self._hang = hang

    def upload_episode(self, ep_dir, repo, cb, dest, private):
        self.calls.append(dest)
        if self._hang:
            import time
            time.sleep(0.5)  # outlives the patched attempt timeout, not the suite
        if len(self.calls) <= self._fail_times:
            raise ConnectionResetError("connection reset by peer")
        return f"https://hf.co/{repo}/{dest}"


def _upload(hf, cancelled=lambda: False, ep_dir="/tmp/ep"):
    return asyncio.run(main._upload_one_episode(
        hf, ep_dir, "u/raw", "ep1/left", False, cancelled))


def test_a_transient_failure_is_retried(monkeypatch):
    # One bad minute on the link must not cost every other device's completed
    # upload, which is what a single-attempt upload did.
    monkeypatch.setattr(main, "_UPLOAD_RETRY_BASE_S", 0.0)
    hf = _Hf(fail_times=2)

    _upload(hf)

    assert len(hf.calls) == 3


def test_retries_are_bounded_and_report_the_last_error(monkeypatch):
    monkeypatch.setattr(main, "_UPLOAD_RETRY_BASE_S", 0.0)
    hf = _Hf(fail_times=99)

    with pytest.raises(RuntimeError) as e:
        _upload(hf)

    assert len(hf.calls) == main._UPLOAD_MAX_ATTEMPTS
    assert "connection reset" in str(e.value)


def test_a_hung_attempt_times_out_instead_of_waiting_forever(monkeypatch):
    # THE regression: an upload that never returns left the device pinned as
    # "uploading" in the fleet forever. It must give up and report.
    monkeypatch.setattr(main, "_UPLOAD_ATTEMPT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(main, "_UPLOAD_RETRY_BASE_S", 0.0)
    monkeypatch.setattr(main, "_UPLOAD_MAX_ATTEMPTS", 1)

    with pytest.raises(RuntimeError) as e:
        _upload(_Hf(hang=True))

    assert "TimeoutError" in str(e.value) or "timeout" in str(e.value).lower()


def test_a_cancel_during_backoff_does_not_wait_it_out(monkeypatch):
    # A cancel is the operator trying to free the device; making them sit out a
    # retry delay is the opposite of what it is for.
    monkeypatch.setattr(main, "_UPLOAD_RETRY_BASE_S", 30.0)
    hf = _Hf(fail_times=99)

    with pytest.raises(main._CommandCancelled):
        _upload(hf, cancelled=lambda: len(hf.calls) >= 1)

    assert len(hf.calls) == 1


# --- 3. incomplete episodes are screened out BEFORE the upload ---------------

class _Tm:
    def __init__(self, root):
        self.root = root

    def episode_dir(self, eid):
        return self.root / eid


def _make_episode(root, eid, *, complete=True):
    from grabette.episode_check import REQUIRED_FILES

    ep = root / eid
    ep.mkdir(parents=True)
    for name in REQUIRED_FILES:
        if not complete and name == "oakd_calib_offline.json":
            continue
        (ep / name).write_bytes(b"x")
    (ep / "oakd_depth.mkv").write_bytes(b"x")
    return ep


def _run_upload(tmp_path, episode_ids, hf, monkeypatch):
    monkeypatch.setattr("grabette.app.routers.tasks.get_task_manager",
                        lambda: _Tm(tmp_path))
    monkeypatch.setattr("grabette.app.routers.huggingface.get_hf_client", lambda: hf)
    monkeypatch.setattr(main, "_daemon", object())
    monkeypatch.setattr(main, "_UPLOAD_RETRY_BASE_S", 0.0)
    return asyncio.run(main._handle_relay_command({
        "id": "cmd1", "type": "upload_episodes",
        "args": {"raw_repo": "u/raw", "role": "left", "episode_ids": episode_ids},
    }))


def test_an_episode_missing_its_calibration_is_never_uploaded(tmp_path, monkeypatch):
    # Pushing it is pure waste: the Space rejects it, and in a bimanual build it
    # drops the peer arm's good recording with it.
    _make_episode(tmp_path, "ep_ok")
    _make_episode(tmp_path, "ep_bad", complete=False)
    hf = _Hf()

    res = _run_upload(tmp_path, ["ep_ok", "ep_bad"], hf, monkeypatch)

    assert res["status"] == "ok"
    assert hf.calls == ["ep_ok/left"]
    assert res["incomplete"] == [
        {"episode_id": "ep_bad", "missing": ["oakd_calib_offline.json"]}]
    # And it SAYS so — a build quietly assembled from a subset is the failure
    # this accounting exists to prevent.
    assert "oakd_calib_offline.json" in res["message"]


def test_nothing_convertible_fails_immediately(tmp_path, monkeypatch):
    # THE regression: uploading zero episodes and letting the build discover it
    # has nothing to convert half an hour later, after every other device pushed.
    _make_episode(tmp_path, "ep_bad", complete=False)
    hf = _Hf()

    res = _run_upload(tmp_path, ["ep_bad"], hf, monkeypatch)

    assert res["status"] == "error"
    assert hf.calls == []
    assert "can be converted" in res["message"]
    assert "oakd_calib_offline.json (1)" in res["message"]


def test_a_complete_batch_uploads_untouched(tmp_path, monkeypatch):
    _make_episode(tmp_path, "ep1")
    _make_episode(tmp_path, "ep2")
    hf = _Hf()

    res = _run_upload(tmp_path, ["ep1", "ep2"], hf, monkeypatch)

    assert res["status"] == "ok"
    assert hf.calls == ["ep1/left", "ep2/left"]
    assert res["incomplete"] == [] and "message" not in res


def test_an_absent_episode_is_still_reported_as_missing(tmp_path, monkeypatch):
    # Absent locally (deleted, never recorded here) stays distinct from present
    # but unconvertible — they need different fixes.
    _make_episode(tmp_path, "ep1")
    hf = _Hf()

    res = _run_upload(tmp_path, ["ep1", "gone"], hf, monkeypatch)

    assert res["missing"] == ["gone"] and res["incomplete"] == []


# --- 4. a stalled upload must never touch the capture path -------------------
# The point of this section: on a Pi, a thread that keeps running while the
# device records competes for CPU with the H.264 encoders and the OAK-D
# drainers. asyncio.to_thread would put it in the DEFAULT executor — the very
# pool the daemon's 50 Hz sensor poll and the OAK-D bring-up use. It must not.

@pytest.fixture(autouse=True)
def _reset_stall_counter():
    main._stalled_uploads = 0
    yield
    main._stalled_uploads = 0


def test_uploads_run_off_the_default_executor(monkeypatch):
    seen = {}

    class _Named:
        def upload_episode(self, *a):
            seen["thread"] = threading.current_thread().name

    _upload(_Named())

    assert seen["thread"].startswith("hf-upload"), seen["thread"]


def test_a_stalled_upload_leaves_the_default_executor_free(monkeypatch):
    # The regression this guards: an abandoned upload holding a default-executor
    # worker starves the sensor poll and the OAK-D bring-up that share it.
    monkeypatch.setattr(main, "_UPLOAD_ATTEMPT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(main, "_UPLOAD_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(main, "_UPLOAD_RETRY_BASE_S", 0.0)

    async def _drive():
        with pytest.raises(RuntimeError):
            await main._upload_one_episode(_Hf(hang=True), "/tmp/ep", "u/raw",
                                           "ep1/left", False, lambda: False)
        # While that thread is still running, work handed to the DEFAULT
        # executor must still be served immediately.
        return await asyncio.wait_for(
            asyncio.to_thread(lambda: threading.current_thread().name), timeout=1.0)

    name = asyncio.run(_drive())

    assert not name.startswith("hf-upload")
    assert main._stalled_uploads == 1  # counted, not forgotten


def test_a_stalled_thread_gives_its_slot_back_when_it_finishes(monkeypatch):
    monkeypatch.setattr(main, "_UPLOAD_ATTEMPT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(main, "_UPLOAD_MAX_ATTEMPTS", 1)
    hf = _Hf(hang=True)  # sleeps 0.5s, i.e. finishes well after we give up

    with pytest.raises(RuntimeError):
        _upload(hf)
    assert main._stalled_uploads == 1

    time.sleep(1.0)  # the abandoned thread completes

    assert main._stalled_uploads == 0, "the done callback must release the slot"


def test_a_saturated_uploader_refuses_instead_of_queueing_silently(monkeypatch):
    # Queueing behind threads that cannot be interrupted looks exactly like the
    # original bug from the operator's seat: an upload that never progresses.
    monkeypatch.setattr(main, "_stalled_uploads", main._UPLOAD_WORKERS)
    hf = _Hf()

    with pytest.raises(RuntimeError) as e:
        _upload(hf)

    assert hf.calls == []
    assert "reboot" in str(e.value)


def test_repeated_stalls_leave_no_phantom_count(monkeypatch):
    # The count gates future uploads, so a leak of even one is a device that
    # slowly refuses to upload at all. Ordering the increment before the release
    # callback is what prevents it — a thread finishing in between would
    # otherwise decrement (clamped at 0) and let the increment linger forever.
    monkeypatch.setattr(main, "_UPLOAD_ATTEMPT_TIMEOUT_S", 0.02)
    monkeypatch.setattr(main, "_UPLOAD_MAX_ATTEMPTS", 1)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            _upload(_Hf(hang=True))
        time.sleep(0.7)  # let the abandoned thread finish and release its slot

    assert main._stalled_uploads == 0
