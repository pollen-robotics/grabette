"""The progress signal the upload watchdog is built on.

app/main.py abandons an upload only when this heartbeat goes stale, so a
heartbeat that is silently NOT wired turns the watchdog blind — and a blind
watchdog falls back to an elapsed-time bound, i.e. to the false positive the
whole design exists to avoid. Hence tests on the wiring itself, not just on the
watchdog that reads it.

The xet source is the one that matters. hf-xet is a hard dependency of
huggingface_hub on every machine we run on and xet is the default upload path,
yet it moves its bytes in a Rust stack, outside the httpx client whose socket
timeouts are the rest of this defence. Nothing else can see that transfer.
"""
import time

import pytest

from grabette import hf


@pytest.fixture(autouse=True)
def _fresh_heartbeat(monkeypatch):
    """Reset the module's install-once state; it is process-wide by design."""
    monkeypatch.setattr(hf, "_heartbeat_installed", False)
    monkeypatch.setattr(hf, "_heartbeat_sources", set())
    monkeypatch.setattr(hf, "_last_activity", 0.0)


# --- the clock ----------------------------------------------------------------

def test_an_unmarked_heartbeat_reports_no_age_rather_than_zero():
    # Zero would read as "activity right now", which is the opposite of the
    # truth and would make a wedged upload look healthy forever.
    assert hf.hub_activity_age_s() is None


def test_marking_makes_the_age_readable_and_small():
    hf.note_hub_activity()

    age = hf.hub_activity_age_s()

    assert age is not None and age < 1.0


def test_the_age_grows_with_silence():
    hf.note_hub_activity()
    first = hf.hub_activity_age_s()
    time.sleep(0.05)

    assert hf.hub_activity_age_s() > first


# --- completeness: a PARTIAL heartbeat must not be trusted --------------------

def test_one_source_is_not_enough():
    # The trap this guards: with only httpx wired, a xet transfer emits no marks
    # for its entire duration, so trusting staleness would abandon a healthy
    # upload — the one outcome the watchdog must never produce.
    hf._add_source("httpx")

    assert hf.heartbeat_sources() == {"httpx"}
    assert not hf.heartbeat_is_complete()


def test_the_other_single_source_is_not_enough_either():
    hf._add_source("xet")

    assert not hf.heartbeat_is_complete()


def test_both_sources_make_the_heartbeat_trustworthy():
    hf._add_source("httpx")
    hf._add_source("xet")

    assert hf.heartbeat_is_complete()


# --- the xet hook -------------------------------------------------------------

@pytest.fixture
def fake_xet(monkeypatch):
    """Stand-in hf_xet whose upload functions match the real signature.

    The real signature is asserted separately below, so this stays a fake without
    becoming a fiction: if hf_xet ever reorders its parameters, that other test
    fails rather than this one silently testing the wrong shape.
    """
    hf_xet = pytest.importorskip("hf_xet")
    calls = []

    def upload_files(file_paths, endpoint, token_info, token_refresher,
                     progress_updater, _repo_type, request_headers=None,
                     sha256s=None, skip_sha256=False):
        calls.append(progress_updater)
        return "committed"

    monkeypatch.setattr(hf_xet, "upload_files", upload_files)
    monkeypatch.setattr(hf_xet, "upload_bytes", upload_files)
    return hf_xet, calls


def test_the_real_hf_xet_signature_still_has_the_progress_argument():
    # The hook injects by POSITION, so this is the assumption that would let it
    # pass a callable where the Rust side expects a repo type. Asserted here, and
    # re-checked at install time, so a reordering degrades instead of corrupting.
    hf_xet = pytest.importorskip("hf_xet")

    index = hf._xet_progress_index(hf_xet.upload_files)

    assert index == 4, f"hf_xet.upload_files moved its progress argument to {index}"


def test_the_injected_updater_marks_the_heartbeat(fake_xet):
    hf_xet, calls = fake_xet

    hf.install_upload_heartbeat()
    hf_xet.upload_files(["f"], "ep", ("t", 1), None, None, "dataset")
    calls[0]("total", [])   # the Rust side reporting progress

    assert "xet" in hf.heartbeat_sources()
    assert hf.hub_activity_age_s() is not None, "the injected updater never marked"


def test_it_marks_even_when_progress_bars_are_off(fake_xet):
    # With bars disabled huggingface_hub builds no reporter and passes None. That
    # is the realistic way to lose this heartbeat — someone quieting tqdm in the
    # systemd unit — so chaining onto None is the case that matters most.
    hf_xet, calls = fake_xet

    hf.install_upload_heartbeat()
    hf_xet.upload_files(["f"], "ep", ("t", 1), None, None, "dataset")
    calls[0]("total", [])

    assert hf.hub_activity_age_s() is not None


def test_an_existing_reporter_still_receives_its_updates(fake_xet):
    # Marking must not cost the operator their progress bars.
    hf_xet, calls = fake_xet
    seen = []

    hf.install_upload_heartbeat()
    hf_xet.upload_files(["f"], "ep", ("t", 1), None,
                        lambda *a: seen.append(a), "dataset")
    calls[0]("total", ["items"])

    assert seen == [("total", ["items"])]


def test_the_progress_argument_is_hooked_when_passed_by_keyword(fake_xet):
    hf_xet, calls = fake_xet

    hf.install_upload_heartbeat()
    hf_xet.upload_files(["f"], "ep", ("t", 1), None, _repo_type="dataset",
                        progress_updater=None)
    calls[0]("total", [])

    assert hf.hub_activity_age_s() is not None


def test_installing_twice_does_not_stack_wrappers(fake_xet):
    hf_xet, _ = fake_xet

    hf.install_upload_heartbeat()
    wrapped_once = hf_xet.upload_files
    hf._heartbeat_installed = False   # force a second real attempt
    hf.install_upload_heartbeat()

    assert hf_xet.upload_files is wrapped_once


def test_a_renamed_progress_argument_is_reported_not_injected(monkeypatch):
    # A future hf_xet could reorder or rename it. Injecting blindly would hand a
    # callable to the wrong parameter; refusing to install just costs the source.
    hf_xet = pytest.importorskip("hf_xet")
    monkeypatch.setattr(hf_xet, "upload_files",
                        lambda paths, endpoint, renamed_cb: None)
    monkeypatch.setattr(hf_xet, "upload_bytes",
                        lambda paths, endpoint, renamed_cb: None)

    hf.install_upload_heartbeat()

    assert "xet" not in hf.heartbeat_sources()
    assert not hf.heartbeat_is_complete()


def test_a_missing_upload_function_is_reported_not_raised(monkeypatch):
    hf_xet = pytest.importorskip("hf_xet")
    monkeypatch.delattr(hf_xet, "upload_files")
    monkeypatch.delattr(hf_xet, "upload_bytes")

    hf.install_upload_heartbeat()   # must not raise

    assert "xet" not in hf.heartbeat_sources()


# --- the mark must mean "bytes moved", not "the library spoke" ----------------

class _Tick:
    """A stand-in for hf_xet's PyTotalProgressUpdate."""

    def __init__(self, processed=0, transferred=0):
        self.total_bytes_completion_increment = processed
        self.total_transfer_bytes_completion_increment = transferred


def test_a_tick_reporting_no_movement_does_not_mark(fake_xet):
    # hf_xet ticks on a timer (it reports a completion RATE), so a wedged
    # transfer keeps calling back. Marking on arrival would call it alive
    # forever — the original hang, one layer up.
    hf_xet, calls = fake_xet

    hf.install_upload_heartbeat()
    hf_xet.upload_files(["f"], "ep", ("t", 1), None, None, "dataset")
    for _ in range(5):
        calls[0](_Tick(), [])

    assert hf.hub_activity_age_s() is None, "a wedged transfer looked alive"


def test_a_transfer_increment_marks(fake_xet):
    hf_xet, calls = fake_xet

    hf.install_upload_heartbeat()
    hf_xet.upload_files(["f"], "ep", ("t", 1), None, None, "dataset")
    calls[0](_Tick(transferred=4096), [])

    assert hf.hub_activity_age_s() is not None


def test_a_hashing_increment_marks_too(fake_xet):
    # The local hashing phase advances total_bytes, not transfer_bytes. Missing
    # it would abandon every upload whose hashing outlasts the stall budget.
    hf_xet, calls = fake_xet

    hf.install_upload_heartbeat()
    hf_xet.upload_files(["f"], "ep", ("t", 1), None, None, "dataset")
    calls[0](_Tick(processed=1 << 20), [])

    assert hf.hub_activity_age_s() is not None


def test_an_unrecognised_payload_marks_rather_than_going_silent(fake_xet):
    # Direction of the fallback: not spotting a hang costs a device held until
    # the blind bound; withholding a mark kills a healthy upload.
    hf_xet, calls = fake_xet

    hf.install_upload_heartbeat()
    hf_xet.upload_files(["f"], "ep", ("t", 1), None, None, "dataset")
    calls[0](object(), [])

    assert hf.hub_activity_age_s() is not None


def test_the_real_increment_fields_still_exist():
    # _progress_moved falls back to marking unconditionally when it recognises
    # nothing, so a rename here would silently disable hang detection.
    hf_xet = pytest.importorskip("hf_xet")

    missing = [n for n in hf._XET_INCREMENTS
               if not hasattr(hf_xet.PyTotalProgressUpdate, n)]

    assert not missing, f"hf_xet renamed {missing}"
