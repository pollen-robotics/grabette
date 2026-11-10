"""Record a benchmark session: episodes started and stopped through the daemon's API.

Stdlib only. On the Pi (or from a workstation with --api), daemon RUNNING,
nobody pressing buttons, dashboard closed (its camera preview loads the Pi):

    python3 benchmarks/run_episodes.py --label "develop @ 1707d4e"             # 30 episodes
    python3 benchmarks/run_episodes.py --durations 120 --total 3600            # 1 h of 2 min takes
    python3 benchmarks/run_episodes.py --group --label "pr-251 group"          # fleet group start

Episodes go to the device's ACTIVE task: select a throwaway one first, e.g.
    curl -s -X POST localhost:8000/api/tasks -H 'Content-Type: application/json' -d '{"name":"bench"}'
    curl -s -X PUT localhost:8000/api/tasks/active -H 'Content-Type: application/json' -d '{"task_id":"<id>"}'

Solo (default): the grabette must not be in an open fleet session, or every
start becomes a group start. --group: the grabettes ARE in a fleet session;
each start here is scheduled by the fleet for all of them at a shared t=0, and
each stop here stops them all. Run the script on ONE member only.

The default plan cycles 5 s, 30 s and 120 s takes: short ones exercise start
and stop, long ones the rare holes. Before each start it checks the free space
of the data folder (when the daemon is local) and stops below --min-free-gb.

Writes ~/benchmarks/<UTC start>_episodes/:
  run.json     plan; for every episode its id, requested length, start
               status, how long the start and stop calls took, and any error;
               this Pi's clock synchronisation at the start and the end
  daemon.log   the daemon's journal over the run, without HTTP access lines
  kernel.log   the kernel's (i2c, USB, mmc, under-voltage messages)
Feed the folder to episode_report.py --run. Ctrl-C stops the current episode
cleanly and still writes everything.

Clock: only this Pi's is recorded (timedatectl, or chronyc). With --group,
run `timedatectl timesync-status` on the other members too: an unsynchronised
clock shifts their capture_started_utc (+52 ms on 2026-11-05).
"""

import argparse
import re
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

stop = threading.Event()

IN_FLEET_SESSION = ("This grabette is in an open fleet session: a start here is a GROUP start "
                    "(every member records, into the session's task). Close the session or take "
                    "this grabette out of the group, or rerun with --group.")


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}", flush=True)


def call(base: str, path: str, method: str = "POST") -> dict:
    req = urllib.request.Request(base + path, data=b"{}" if method == "POST" else None, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read() or b"{}")


def http_detail(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        try:
            return e.read().decode(errors="replace")
        except Exception:
            return ""
    return ""


def is_local(api: str) -> bool:
    return urlparse(api).hostname in ("127.0.0.1", "localhost", "::1")


def journal(since_epoch: float, *args: str) -> str:
    try:
        r = subprocess.run(["journalctl", "--no-pager", "-o", "short-precise", "--utc",
                            "--since", f"@{int(since_epoch)}", *args],
                           capture_output=True, text=True, timeout=120)
        return r.stdout or r.stderr
    except (OSError, subprocess.SubprocessError) as e:
        return f"journalctl failed: {e}\n"


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


_UNITS_MS = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1000.0, "min": 60000.0}


def parse_clock(show: str | None, timesync: str | None, chrony: str | None) -> dict:
    """NTP state and offset from `timedatectl show`, `timedatectl
    timesync-status` (systemd-timesyncd) or `chronyc tracking`."""
    out: dict = {"ntp_synchronized": None, "offset_ms": None, "source": None}
    if show:
        m = re.search(r"^NTPSynchronized=(\w+)", show, re.M)
        if m:
            out["ntp_synchronized"] = m.group(1) == "yes"
    if timesync:
        m = re.search(r"^\s*Offset:\s*([+-]?[\d.]+)\s*(ns|us|µs|ms|s|min)\b", timesync, re.M)
        if m:
            out["offset_ms"] = round(float(m.group(1)) * _UNITS_MS[m.group(2)], 3)
            out["source"] = "timesyncd"
    if out["offset_ms"] is None and chrony:
        m = re.search(r"^System time\s*:\s*([\d.]+) seconds (fast|slow)", chrony, re.M)
        if m:
            v = float(m.group(1)) * 1000
            out["offset_ms"] = round(v if m.group(2) == "fast" else -v, 3)
            out["source"] = "chrony"
    return out


def clock_status() -> dict:
    return {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **parse_clock(_run(["timedatectl", "show"]), _run(["timedatectl", "timesync-status"]),
                          _run(["chronyc", "tracking"]))}


def free_gb(folder: Path) -> float | None:
    try:
        return shutil.disk_usage(folder).free / 1e9
    except OSError:
        return None


def wait_until(t_utc: datetime) -> None:
    stop.wait(max((t_utc - datetime.now(timezone.utc)).total_seconds(), 0))


def record_one(base: str, seconds: float, group: bool) -> dict:
    """One episode; returns what to keep in run.json. Sets `stop` when the run
    cannot go on (fleet session in solo mode)."""
    entry = {"requested_s": seconds, "start_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}
    t = time.monotonic()
    try:
        started = call(base, "/api/episodes/start")
        entry["start_call_s"] = round(time.monotonic() - t, 3)
    except (urllib.error.URLError, OSError, ValueError) as e:
        entry["error"] = f"start: {e} {http_detail(e)}".strip()
        if "group start refused" in entry["error"]:
            log(f"start refused by the fleet. {IN_FLEET_SESSION}")
            stop.set()
        return entry
    status = started.get("status")
    entry["status"] = status
    entry["episode_id"] = started.get("episode_id")
    if status == "capturing":
        stop.wait(seconds)
    elif status == "scheduled" and group:
        t0 = datetime.fromisoformat(started["start_at_utc"])
        entry["scheduled_start_utc"] = started["start_at_utc"]
        entry["peers"] = started.get("peers")
        wait_until(t0)
        stop.wait(seconds)
    else:
        # The fleet took the start over: cancel it, leave no recording behind.
        try:
            call(base, "/api/episodes/stop")
        except Exception:
            pass
        entry["error"] = f"start answered {started}"
        log(f"start answered {started}. {IN_FLEET_SESSION if not group else ''}")
        stop.set()
        return entry
    # How long the daemon takes to stop: it waits for every stream to stop
    # and the videos to be muxed (camera_stop reached 4-7 s on SD-card stalls).
    t = time.monotonic()
    try:
        call(base, "/api/episodes/stop")
        entry["stop_call_s"] = round(time.monotonic() - t, 3)
    except (urllib.error.URLError, OSError, ValueError) as e:
        entry["error"] = f"stop: {e} {http_detail(e)}".strip()
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--api", default="http://127.0.0.1:8000", help="daemon URL")
    p.add_argument("--durations", default="5,30,120",
                   help="episode lengths in s, cycled (default 5,30,120)")
    p.add_argument("--count", type=int, default=30, help="episodes to record (default 30)")
    p.add_argument("--total", type=float, help="instead of --count: record for this many seconds")
    p.add_argument("--pause", type=float, default=5, help="pause between episodes, s (default 5)")
    p.add_argument("--group", action="store_true", help="the grabettes are in a fleet session: group starts")
    p.add_argument("--min-free-gb", type=float, default=3.0,
                   help="stop when the data folder has less free space (local daemon only)")
    p.add_argument("--data-dir", type=Path, default=Path.home() / "grabette-data")
    p.add_argument("--label", default="", help="what is measured, e.g. 'develop @ 1707d4e'")
    p.add_argument("--out", type=Path, default=Path.home() / "benchmarks")
    args = p.parse_args()

    base = args.api.rstrip("/")
    plan = [float(x) for x in args.durations.split(",") if x.strip()]
    local = is_local(base)
    try:
        call(base, "/api/session/status", method="GET")
    except Exception as e:
        sys.exit(f"daemon not reachable at {base}: {e}")

    t0_epoch = time.time()
    out_dir = args.out / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_episodes"
    out_dir.mkdir(parents=True)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    run = {"kind": "run_episodes", "label": args.label, "api": base, "group": args.group,
           "plan_s": plan, "pause_s": args.pause,
           "start_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "episodes": []}
    if local:
        run["clock"] = {"start": clock_status()}
        c = run["clock"]["start"]
        if c["ntp_synchronized"] is False or abs(c["offset_ms"] or 0) > 5:
            log(f"WARNING clock not synchronised ({c}): capture_started_utc will be off. "
                "Check `timedatectl timesync-status` before going on.")
    log(f"-> {out_dir}  ({'group' if args.group else 'solo'}, plan {plan} s, "
        f"{f'{args.total:.0f} s' if args.total else f'{args.count} episodes'})")

    i = 0
    while not stop.is_set():
        if args.total is None and i >= args.count:
            break
        if args.total is not None and time.time() - t0_epoch >= args.total:
            break
        if local:
            gb = free_gb(args.data_dir)
            if gb is not None and gb < args.min_free_gb:
                log(f"only {gb:.1f} GB free in {args.data_dir}: stopping before the card fills up")
                run["stopped_for_disk_space"] = True
                break
        seconds = plan[i % len(plan)]
        entry = record_one(base, seconds, args.group)
        run["episodes"].append(entry)
        log(f"episode {entry.get('episode_id', '?')} ({seconds:.0f} s)"
            + (f"  ERROR {entry['error']}" if entry.get("error") else ""))
        i += 1
        stop.wait(args.pause)

    run["end_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if local:
        run["clock"]["end"] = clock_status()
        daemon = journal(t0_epoch, "-u", "grabette")
        (out_dir / "daemon.log").write_text(
            "\n".join(l for l in daemon.splitlines() if "HTTP/1.1" not in l) + "\n")
        (out_dir / "kernel.log").write_text(journal(t0_epoch, "-k"))
    else:
        run["logs"] = "remote daemon: copy its journal over the run window by hand"
    (out_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    ok = sum(1 for e in run["episodes"] if not e.get("error"))
    log(f"{ok}/{len(run['episodes'])} episodes recorded. Next: "
        f"python3 benchmarks/episode_report.py {args.data_dir / 'episodes'} --run {out_dir} "
        f"--label '{args.label}' --out {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
