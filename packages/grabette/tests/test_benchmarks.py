"""benchmarks/episode_report.py and compare_reports.py on synthetic episodes.

The benchmark tools are standalone scripts (stdlib, run with the system python
on the Pi), so they are loaded from their files rather than imported."""
import importlib.util
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


report_mod = _load("episode_report")
compare_mod = _load("compare_reports")
run_mod = _load("run_episodes")


def _episode(root: Path, eid: str, device: str = "grabette-test", *, seconds: float = 60.0,
             rgb_holes=(), startup_holes=0, angle_holes=(), depth_seq_jump=0,
             start_capture_ms=80.0, valid=True, frame_count_delta=0, angle_end_s=None,
             pts=None) -> Path:
    ep = root / eid / device
    ep.mkdir(parents=True)
    # RGB at 50 fps; a hole drops the frames inside it.
    step = 20.0
    rgb = [5 + i * step for i in range(int(seconds * 1000 / step))]
    drop = set()
    for k in range(startup_holes):            # one frame lost each, in the first 1.5 s
        drop.add(10 + 5 * k)
    for at_s, ms in rgb_holes:
        first = int(at_s * 1000 / step)
        drop.update(range(first, first + int(ms / step) - 1))
    rgb = [t for i, t in enumerate(rgb) if i not in drop]
    (ep / "frame_timestamps.json").write_text(json.dumps(rgb))
    # Depth at 30 fps, numbered; a seq jump = frames the host never received.
    depth = []
    seq = 100
    for i in range(int(seconds * 30)):
        if i == 300 and depth_seq_jump:
            seq += depth_seq_jump
        depth.append({"seq": seq, "device_us": i * 33333, "host_ms": 3 + i * 33.333
                      + (depth_seq_jump * 33.333 if depth_seq_jump and i >= 300 else 0)})
        seq += 1
    (ep / "dcam_left_timestamps.json").write_text(json.dumps({"samples": depth}))
    # Angles at 90 Hz with optional holes.
    ang, t = [], 0.5
    holes = dict(angle_holes)
    while t < (angle_end_s or seconds) * 1000:
        ang.append({"cts": t, "value": [0.0, 0.0]})
        t += holes.pop(next((h for h in holes if t >= h * 1000), None), 11.1) if holes else 11.1
    (ep / "angle_data.json").write_text(json.dumps({"samples": ang}))
    meta = {
        "duration_seconds": seconds, "frame_count": len(rgb) + frame_count_delta, "device_id": device,
        "video_timestamp_alignment": {"valid": valid, "unwritten_camera_frame_count": 0,
                                      "reordered_output_count": 0, "writer_queue_peak": 12},
        "dcam": {"left_frames": len(depth), "depth_frames": len(depth), "dropped_frames": 0},
        "sync": {"scheduled_start_utc": "2026-11-09T10:00:00.000000+00:00",
                 "capture_started_utc": "2026-11-09T10:00:00.003000+00:00",
                 "wake_skew_ms": 0.3, "start_capture_ms": start_capture_ms},
    }
    if pts:
        meta["video_timestamp_alignment"].update(pts_offset_source=pts[0], pts_offset_candidates=pts[1])
    (ep / "metadata.json").write_text(json.dumps(meta))
    return ep


def _report(root: Path, label="test") -> dict:
    eps = [report_mod.measure_episode(f) for f in report_mod.find_episodes([root])]
    devices = {}
    for e in eps:
        devices.setdefault(e["device"], []).append(e)
    return {"label": label, "overall": report_mod.summarize(eps),
            "devices": {d: report_mod.summarize(v) for d, v in devices.items()},
            "groups": report_mod.group_measures(eps), "chrono": None, "episodes": eps}


def test_episode_measures_find_each_kind_of_hole(tmp_path):
    ep = _episode(tmp_path, "20261109_100000", rgb_holes=[(20, 300)], startup_holes=3,
                  angle_holes=[(30, 1030)], depth_seq_jump=5, start_capture_ms=3400)
    m = report_mod.measure_episode(ep)
    assert m["episode"] == "20261109_100000" and m["device"] == "grabette-test"
    assert m["rgb"]["holes_startup"] == 3
    assert m["rgb"]["holes_after_startup"] == 1
    assert m["rgb"]["holes_over_250ms"] == 1
    assert m["rgb"]["frame_count_matches"] and m["rgb"]["alignment_valid"] is True
    assert m["depth"]["seq_lost"] == 5 and m["depth"]["holes"] == 1
    assert m["angle"]["holes_over_500ms"] == 1
    assert 85 < m["angle"]["rate_hz"] < 92
    assert m["start"]["start_capture_ms"] == 3400
    assert m["start"]["started_minus_scheduled_ms"] == 3.0


def test_summary_counts_invariants_and_late_starts(tmp_path):
    _episode(tmp_path, "20261109_100000", valid=False, start_capture_ms=3400)
    _episode(tmp_path, "20261109_100100", frame_count_delta=3)
    s = _report(tmp_path)["overall"]
    assert s["invariants"]["alignment_invalid"] == 1
    assert s["invariants"]["frame_count_mismatch"] == 1
    assert s["start"]["late_starts"] == {"count": 1, "episodes": 2}
    assert s["minutes"] == 2.0


def test_groups_pair_the_devices_of_an_episode(tmp_path):
    _episode(tmp_path, "20261109_100000", "left")
    _episode(tmp_path, "20261109_100000", "right")
    groups = _report(tmp_path)["groups"]
    assert groups == [{"episode": "20261109_100000", "devices": ["left", "right"],
                       "same_scheduled_start": True, "capture_started_spread_ms": 0.0}]


def test_compare_flags_a_significant_rise_in_angle_timeouts(tmp_path):
    for i in range(10):
        _episode(tmp_path / "base", f"20261109_10{i:02d}00", seconds=360)
        _episode(tmp_path / "cand", f"20261109_11{i:02d}00", seconds=360,
                 angle_holes=[(100, 1030), (200, 1030)])
    rows, regressed = compare_mod.compare(_report(tmp_path / "base"), _report(tmp_path / "cand"))
    row = next(r for r in rows if r["metric"].startswith("Angles: holes > 500"))
    assert row["verdict"] == "REGRESSION" and regressed


def test_compare_does_not_call_one_extra_event_a_regression(tmp_path):
    _episode(tmp_path / "base", "20261109_100000", seconds=600)
    _episode(tmp_path / "cand", "20261109_110000", seconds=600, angle_holes=[(100, 1030)])
    rows, regressed = compare_mod.compare(_report(tmp_path / "base"), _report(tmp_path / "cand"))
    row = next(r for r in rows if r["metric"].startswith("Angles: holes > 500"))
    assert row["verdict"] == "worse, not significant" and not regressed


def test_compare_flags_a_broken_invariant_only_when_the_baseline_kept_it(tmp_path):
    _episode(tmp_path / "base", "20261109_100000")
    _episode(tmp_path / "cand", "20261109_110000", valid=False)
    _, regressed = compare_mod.compare(_report(tmp_path / "base"), _report(tmp_path / "cand"))
    assert regressed
    _, regressed = compare_mod.compare(_report(tmp_path / "cand"), _report(tmp_path / "cand"))
    assert not regressed


@pytest.mark.parametrize("k,lam,expected", [(0, 2.0, 0.1353), (2, 2.0, 0.6767), (5, 0.5, 0.99998)])
def test_poisson_cdf(k, lam, expected):
    assert compare_mod.poisson_cdf(k, lam) == pytest.approx(expected, abs=1e-4)


def test_end_flags_a_stream_that_stopped_early(tmp_path):
    ok = report_mod.measure_episode(_episode(tmp_path, "20261109_100000"))
    assert ok["end"]["stopped_early"] == [] and ok["end"]["spread_ms"] < 100
    early = report_mod.measure_episode(_episode(tmp_path, "20261109_100100", angle_end_s=50))
    assert early["end"]["stopped_early"] == ["angle"]
    s = _report(tmp_path)["overall"]["end"]
    assert s["streams_stopped_early"] == {"count": 1, "episodes": 2}


def test_pts_offset_fields_are_counted(tmp_path):
    _episode(tmp_path, "20261109_100000", pts=("encoder_first_timestamp", 3))
    _episode(tmp_path, "20261109_100100", pts=("matched", 1))
    _episode(tmp_path, "20261109_100200")  # before ad4eb7b: no field
    rgb = _report(tmp_path)["overall"]["rgb"]
    assert rgb["pts_offset_ambiguous"] == 1 and rgb["pts_offset_matched"] == 1


def test_decode_mismatch_is_an_invariant(tmp_path, monkeypatch):
    ep = _episode(tmp_path, "20261109_100000")
    (ep / "raw_video.mp4").write_bytes(b"")
    n = len(json.loads((ep / "frame_timestamps.json").read_text()))
    monkeypatch.setattr(report_mod, "decoded_frames", lambda path: n - 2)  # two pictures dropped
    m = report_mod.measure_episode(ep, decode=True)
    assert m["decode"]["rgb"] == {"decoded": n - 2, "timestamps": n, "matches": False}
    s = report_mod.summarize([m])
    assert s["invariants"]["decoded_frame_mismatch"] == 1 and s["episodes_decoded"] == 1


def test_compare_skips_decode_when_one_side_did_not_decode(tmp_path, monkeypatch):
    _episode(tmp_path / "base", "20261109_100000")
    ep = _episode(tmp_path / "cand", "20261109_110000")
    (ep / "raw_video.mp4").write_bytes(b"")
    monkeypatch.setattr(report_mod, "decoded_frames", lambda path: 1)
    base = _report(tmp_path / "base")
    cand = _report(tmp_path / "cand")
    cand["overall"] = report_mod.summarize([report_mod.measure_episode(ep, decode=True)])
    rows, regressed = compare_mod.compare(base, cand)
    assert not any(r["metric"].startswith("Videos: decoded") for r in rows) and not regressed


def test_run_inputs_read_stop_calls_phases_and_clock(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({
        "label": "develop", "episodes": [{"episode_id": "20261109_100000", "stop_call_s": 2.4}],
        "clock": {"start": {"ntp_synchronized": True, "offset_ms": 0.4},
                  "end": {"ntp_synchronized": True, "offset_ms": -12.0}}}))
    (run / "daemon.log").write_text(
        "RpiBackend capture stopped [ms: angle_stop=12 camera_stop=1650 muxes_wallclock=300  total_awaited=312]\n"
        "RpiBackend capture stopped [ms: angle_stop=10 camera_stop=4020 muxes_wallclock=280  total_awaited=290]\n")
    ids, logs, entries = report_mod.run_inputs(run)
    assert ids == {"20261109_100000"} and entries["20261109_100000"]["stop_call_s"] == 2.4
    assert logs["stop_phases_ms"]["camera_stop"] == {"median": 2835, "max": 4020, "n": 2}
    assert logs["clock"] == {"ntp_synchronized": True, "offset_ms_abs_max": 12.0, "ok": False}
    report = {"logs": {"develop": logs}}
    assert compare_mod.clock_warnings(report, "Baseline")


@pytest.mark.parametrize("show,timesync,chrony,expected", [
    ("NTPSynchronized=yes\n", "  Offset: -512us\n", None, (True, -0.512)),
    ("NTPSynchronized=no\n", "  Offset: +1.5ms\n", None, (False, 1.5)),
    (None, None, "System time     : 0.002000000 seconds slow of NTP time\n", (None, -2.0)),
])
def test_parse_clock(show, timesync, chrony, expected):
    c = run_mod.parse_clock(show, timesync, chrony)
    assert (c["ntp_synchronized"], c["offset_ms"]) == expected


@pytest.mark.parametrize("temp,throttled,expected", [
    ("temp=82.3'C\n", "throttled=0x80008\n", (82.3, ["soft_temp_limit"], ["soft_temp_limit"])),
    ("temp=77.9'C\n", "throttled=0x0\n", (77.9, [], [])),
    ("temp=60.0'C\n", "throttled=0x50000\n", (60.0, [], ["under_voltage", "throttled"])),
    (None, None, (None, [], [])),
])
def test_parse_thermal(temp, throttled, expected):
    t = run_mod.parse_thermal(temp, throttled)
    assert (t["temp_c"], t["now"], t["since_boot"]) == expected


def _thermal(temp, now=(), since=()):
    return {"temp_c": temp, "now": list(now), "since_boot": list(since)}


def test_thermal_check_flags_heat_and_new_throttling():
    cool = {"thermal": {"start": _thermal(60), "end": _thermal(70)},
            "episodes": [{"thermal": _thermal(65)}]}
    assert report_mod.thermal_check(cool)["ok"]
    # Throttled since boot before the run started: not held against this run.
    old = {"thermal": {"start": _thermal(60, since=["soft_temp_limit"]),
                       "end": _thermal(70, since=["soft_temp_limit"])}, "episodes": []}
    assert report_mod.thermal_check(old)["ok"]
    hot = {"thermal": {"start": _thermal(70), "end": _thermal(79, since=["soft_temp_limit"])},
           "episodes": [{"thermal": _thermal(82.3, now=["soft_temp_limit"])}]}
    t = report_mod.thermal_check(hot)
    assert not t["ok"] and t["temp_c_max"] == 82.3
    assert t["throttled_now"] == ["soft_temp_limit"] and t["new_since_boot"] == ["soft_temp_limit"]


def test_compare_warns_about_heat():
    def rep(t):
        return {"logs": {"run": {"thermal": t}}}
    ok = {"temp_c_max": 70.0, "throttled_now": [], "new_since_boot": [], "ok": True}
    hot = {"temp_c_max": 82.3, "throttled_now": ["soft_temp_limit"], "new_since_boot": [], "ok": False}
    assert compare_mod.thermal_warnings(rep(ok), rep(ok)) == []
    w = compare_mod.thermal_warnings(rep(ok), rep(hot))
    assert any("Candidate" in x and "82.3" in x for x in w)
    assert any("same temperature" in x for x in w)  # 70 vs 82.3


def test_poisson_cdf_does_not_underflow_on_large_counts():
    assert compare_mod.poisson_cdf(800, 800.0) == pytest.approx(0.509, abs=0.01)
    verdict, _ = compare_mod.judge_rate({"count": 800, "minutes": 60}, {"count": 1600, "minutes": 60})
    assert verdict == "REGRESSION"
    verdict, _ = compare_mod.judge_rate({"count": 800, "minutes": 60}, {"count": 790, "minutes": 60})
    assert verdict == "ok"


def test_one_depth_burst_is_one_event_not_a_regression(tmp_path):
    _episode(tmp_path / "base", "20261109_100000", seconds=600)
    ep = _episode(tmp_path / "cand", "20261109_110000", seconds=600, depth_seq_jump=30)
    m = report_mod.measure_episode(ep)
    assert m["depth"]["seq_lost"] == 30 and m["depth"]["seq_loss_events"] == 1
    rows, _ = compare_mod.compare(_report(tmp_path / "base"), _report(tmp_path / "cand"))
    row = next(r for r in rows if r["metric"].startswith("Depth: losses"))
    assert row["verdict"] == "worse, not significant"


def test_an_invariant_broken_far_more_often_is_a_regression(tmp_path):
    for i in range(10):
        _episode(tmp_path / "base", f"20261109_10{i:02d}00", valid=(i != 0))
        _episode(tmp_path / "cand", f"20261109_11{i:02d}00", valid=False)
    rows, regressed = compare_mod.compare(_report(tmp_path / "base"), _report(tmp_path / "cand"))
    row = next(r for r in rows if r["metric"].startswith("RGB: episodes with invalid"))
    assert row["verdict"] == "REGRESSION" and regressed
    rows, regressed = compare_mod.compare(_report(tmp_path / "base"), _report(tmp_path / "base"))
    assert not regressed


def test_run_inputs_count_angle_read_errors_and_alignment_failures(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"episodes": [{"episode_id": "20261109_100000"}]}))
    (run / "daemon.log").write_text(
        "Angle: 4521 reads, 3 errors\nAngle: 900 reads, 0 errors\n"
        "Video timestamp alignment failed: no camera frame for output PTS\n")
    _, logs, _ = report_mod.run_inputs(run)
    assert logs["daemon"]["angle_read_errors"] == 3
    assert logs["daemon"]["alignment_error"] == 1


def test_a_run_without_episodes_is_refused_not_ignored(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"episodes": [{"error": "start: refused"}]}))
    _episode(tmp_path / "eps", "20261109_100000")
    monkeypatch.setattr("sys.argv", ["episode_report", str(tmp_path / "eps"), "--run", str(run)])
    with pytest.raises(SystemExit) as caught:
        report_mod.main()
    assert "no recorded episode" in str(caught.value)


def _fake_daemon(monkeypatch, *, start, capturing_after_stop=False, stop_fails=False):
    state = {"is_capturing": False}

    def call(base, path, method="POST"):
        if path == "/api/episodes/start":
            state.update(is_capturing=True, episode_id="20261109_100000")
            return start
        if path == "/api/episodes/stop":
            state["is_capturing"] = capturing_after_stop
            if stop_fails:
                raise run_mod.urllib.error.URLError("timed out")
            return {}
        if path == "/api/state":
            return {"capture": dict(state)}
        raise AssertionError(path)

    monkeypatch.setattr(run_mod, "call", call)
    run_mod.stop.clear()


def test_a_group_start_waiting_for_the_cameras_records_once_the_fleet_starts_it(monkeypatch):
    _fake_daemon(monkeypatch, start={"status": "waiting_camera"})
    entry = run_mod.record_one("http://x", 0, group=True)
    assert "error" not in entry and entry["episode_id"] == "20261109_100000"
    assert not run_mod.stop.is_set()


def test_a_failed_stop_waits_for_the_daemon_then_goes_on(monkeypatch):
    _fake_daemon(monkeypatch, start={"status": "capturing", "episode_id": "e"}, stop_fails=True)
    entry = run_mod.record_one("http://x", 0, group=False)
    assert entry["error"].startswith("stop:") and not run_mod.stop.is_set()


def test_a_daemon_still_capturing_after_a_failed_stop_ends_the_run(monkeypatch):
    _fake_daemon(monkeypatch, start={"status": "capturing", "episode_id": "e"},
                 stop_fails=True, capturing_after_stop=True)
    monkeypatch.setattr(run_mod, "STOP_WAIT_S", 0.3)
    run_mod.record_one("http://x", 0, group=False)
    assert run_mod.stop.is_set()
    run_mod.stop.clear()
