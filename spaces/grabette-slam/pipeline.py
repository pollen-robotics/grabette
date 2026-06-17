"""Episode → SLAM → LeRobot → Hub pipeline, run in-process inside the Space.

Thin glue over grabette_postprocess. The SLAM binary is bundled in the image
(no Docker), so run_oak_slam is called with binary=OAK_VSLAM_BINARY.
"""

import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from grabette_postprocess.convert import convert_episode
from grabette_postprocess.oak_slam import run_oak_slam
from grabette_postprocess.dataset import build_dataset
from grabette_postprocess.trajectory import analyze_trajectory
from grabette_postprocess.episode_check import check_episode

BINARY = os.environ.get("OAK_VSLAM_BINARY", "/usr/local/bin/offline_vslam")


def find_episode_dirs(root: Path) -> list[Path]:
    """Every directory containing a raw OAK-D recording, anywhere under root.

    Recursive so it works whether the dataset wraps episodes in a folder, puts
    them at the top level, or is a single episode at the root.
    """
    return sorted({p.parent for p in Path(root).rglob("oakd_left.mp4")})


def _process_episode(ep: Path):
    """Convert + SLAM + quality-check one episode (runs in a worker thread).

    Returns (episode_dir | None, log_lines, report | None). The dir is None when
    SLAM produced no trajectory (the episode is dropped outright) — report is then
    None too. Log lines are collected rather than emitted directly so concurrent
    episodes don't interleave in the output. The trajectory quality check is
    advisory at this stage: a BAD/WARN verdict is logged and the report returned,
    so the caller can offer the user a chance to drop flagged episodes before the
    dataset is built (see build_lerobot's `review` hook).

    Pause is enforced by run_slam at dispatch time (it gates before starting each
    episode), so a paused run never reaches this function for not-yet-started
    episodes; one already running its SLAM subprocess finishes on its own.
    """
    # Advisory input check — validates the raw recording (Arducam / OAK RGBD+IMU
    # / gripper angles) before SLAM. Logged, never blocks (a noisy episode can
    # still produce a usable trajectory).
    lines = []
    # require_right=False: the Space skips downloading oakd_right.mp4 (unused by
    # SLAM/dataset), so checking for it would falsely flag every episode.
    chk = check_episode(ep, require_right=False)
    if chk["errors"] or chk["warnings"]:
        verdict = "ERROR" if chk["errors"] else "WARN"
        lines.append(f"  [input/{verdict}] {ep.name}")
        for msg in (*chk["errors"], *chk["warnings"]):
            lines.append(f"      • {msg}")

    lines.append(f"▶ {ep.name}: convert")
    convert_episode(ep)

    lines.append(f"▶ {ep.name}: SLAM")
    r = run_oak_slam(ep, binary=BINARY, show_progress=False)
    if r.trajectory_path is None:
        lines.append(f"  ✗ {ep.name}: SLAM failed (rc={r.returncode})")
        return None, lines, None
    lines.append(f"  ✓ {ep.name}: tracking {r.tracking_pct:.1f}% ({r.tracked_frames}/{r.total_frames})")

    # Trajectory quality check — flags drift/jumps/zigzag. The verdict is returned
    # so the caller can let the user drop flagged episodes (build_lerobot review).
    report = analyze_trajectory(r.trajectory_path, ep / "slam_metadata.json")
    lines.append(
        f"  [{report.verdict}] {ep.name}: dist={report.total_distance_m:.2f}m "
        f"med_step={report.median_step_mm:.1f}mm jumps={report.n_jumps}"
    )
    for msg in (*report.errors, *report.warnings):
        lines.append(f"      • {msg}")
    return ep, lines, report


def run_slam(dataset_dir, log=print, should_stop=None, on_progress=None,
             gate=None) -> list[tuple[Path, "TrajectoryReport"]]:
    """Convert + SLAM every episode in parallel.

    Returns a list of (episode_dir, trajectory_report) for every episode that
    produced a trajectory, sorted by episode name. Episodes whose SLAM failed are
    dropped outright (not returned). The report carries the quality verdict so the
    caller can offer the user a chance to drop flagged episodes before building.

    Episodes are independent (each reads/writes only its own dir), so they run
    concurrently. SLAM is a single-threaded CPU-bound subprocess, so workers are
    capped at the core count; processing whole episodes per worker also overlaps
    one episode's convert with another's SLAM for free.

    should_stop: optional no-arg predicate; when it returns True the loop stops
    collecting results and cancels not-yet-started episodes (a running SLAM
    finishes on its own — a subprocess can't be interrupted mid-call).
    on_progress: optional callback(done, total, "slam") fired per finished episode.
    gate: optional no-arg pause checkpoint. It is honored at dispatch time — before
        each episode is submitted — NOT inside the worker, so a paused run holds
        *between* episodes (it stops starting new ones) while any in-flight SLAM
        subprocess finishes on its own (a subprocess can't be interrupted mid-call).
    """
    episodes = find_episode_dirs(Path(dataset_dir))
    if not episodes:
        raise ValueError(f"No episodes (oakd_left.mp4) found under {dataset_dir}")

    total = len(episodes)
    log(f"Found {total} episode(s) : {', '.join(ep.name for ep in episodes)}")
    if on_progress:
        on_progress(0, total, "slam")
    workers = min(total, os.cpu_count() or 2)
    processed = []
    done = 0
    pending = iter(episodes)

    def _dispatch_next():
        """Pause/stop checkpoint, then return the next episode to start (or None when
        drained / stopped). gate() blocks here while paused, so no NEW episode is
        started until the user resumes — this is what makes Pause work during SLAM."""
        if gate:
            gate()  # blocks while paused; returns at once if a stop is requested
        if should_stop and should_stop():
            return None
        return next(pending, None)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight = set()
        for _ in range(workers):  # prime the pool (gated, so a pause holds at start)
            ep = _dispatch_next()
            if ep is None:
                break
            in_flight.add(pool.submit(_process_episode, ep))

        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in finished:
                ep, lines, report = fut.result()
                for line in lines:
                    log(line)
                if ep is not None:
                    processed.append((ep, report))
                done += 1
                if on_progress:
                    on_progress(done, total, "slam")
            if should_stop and should_stop():
                log("⛔ Stop requested — abandoning remaining episodes")
                pool.shutdown(wait=False, cancel_futures=True)
                break
            while len(in_flight) < workers:  # refill (gated → Pause holds here)
                ep = _dispatch_next()
                if ep is None:
                    break
                in_flight.add(pool.submit(_process_episode, ep))
    return sorted(processed, key=lambda t: t[0].name)


def _branch_name(task: str) -> str:
    """Deterministic branch name from the task — same input gives the same branch,
    so a retry pushes to the branch the first attempt was aiming for."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:40] or "update"
    return f"grabette-{slug}"


def _push_to_branch(ds, target_repo, branch, log, token, attempts=3):
    """Create the branch from main, then push_to_hub to it — retrying transient HF
    5xx errors, with an error message keyed on the real HTTP status (a 5xx is a
    server hiccup, not a token problem — only 401/403 mean the token can't write).

    We create the branch explicitly (from the repo's default branch) and pin
    ds.revision to it, because LeRobot's push_to_hub otherwise branches from its
    codebase version (e.g. "v3.0") — a ref that doesn't exist on a plain repo, so
    HF returns a *deterministic* 500.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for i in range(attempts):
        try:
            api.create_branch(target_repo, branch=branch, repo_type="dataset", exist_ok=True)
            ds.revision = branch  # makes push_to_hub's own create_branch a no-op
            ds.push_to_hub(branch=branch, tags=["lerobot", "grabette"], tag_version=False)
            return
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status and status >= 500 and i < attempts - 1:
                wait = 3 * (i + 1)
                log(f"  ⚠️ HF server error (HTTP {status}); retrying in {wait}s "
                    f"({i + 1}/{attempts - 1})…")
                time.sleep(wait)
                continue
            if status in (401, 403):
                raise RuntimeError(
                    f"Could not write to {target_repo} (HTTP {status}): your sign-in "
                    f"token appears read-only. Re-login to grant the Space the "
                    f"write-repos scope, then re-run."
                ) from e
            if status and status >= 500:
                raise RuntimeError(
                    f"Hugging Face had a server error (HTTP {status}) while pushing to "
                    f"branch '{branch}' on {target_repo}. The built dataset is cached — "
                    f"click “Retry push” to try again without re-running SLAM."
                ) from e
            raise RuntimeError(
                f"Could not push to branch '{branch}' on {target_repo}: {e}"
            ) from e


def build_lerobot(dataset_dir, target_repo, task, root, log=print,
                  should_stop=None, to_branch=False, on_progress=None,
                  token=None, gate=None, review=None) -> list:
    """Convert + SLAM + build the LeRobot dataset on disk under `root`.

    Returns the list of processed episode dirs. Does NOT push — call push_lerobot()
    afterwards. Splitting build from push lets a failed push be retried (the built
    dataset stays on disk) without re-running the slow SLAM step.

    should_stop: optional no-arg predicate; checked before the build so a stop
        request never produces a dataset.
    token: HF token (exported as HF_TOKEN; also used to resolve the username for
        the per-episode traceability sidecar when to_branch is set).
    gate: optional no-arg pause checkpoint; honored per episode during SLAM and
        once more before the build step.
    review: optional callback(results) -> list[Path], called between SLAM and the
        build with the list of (episode_dir, trajectory_report) tuples. It returns
        the episode dirs to keep — letting the caller drop episodes whose
        trajectory check came back flagged. When None, every episode is kept.
    """
    import os

    if token:
        os.environ["HF_TOKEN"] = token

    results = run_slam(dataset_dir, log=log, should_stop=should_stop,
                       on_progress=on_progress, gate=gate)
    if should_stop and should_stop():
        raise RuntimeError("Stopped by user — nothing pushed.")
    if not results:
        raise RuntimeError("No episode produced a trajectory; nothing to push.")

    # Episode review: let the caller drop flagged episodes before building. The
    # callback may block (e.g. waiting on user input) — keep it after the stop
    # check above so a cancel during SLAM never reaches it.
    if review is not None:
        processed = review(results)
    else:
        processed = [ep for ep, _ in results]
    if should_stop and should_stop():
        raise RuntimeError("Stopped by user — nothing pushed.")
    if not processed:
        raise RuntimeError("All episodes were dropped in review — nothing to push.")
    if gate:
        gate()  # pause checkpoint before the (CPU-heavy) build

    # When pushing to a branch of an existing repo, tag each episode with the
    # source recording + user (meta/episode_sources.json) so episodes from
    # different users sharing one repo stay distinguishable.
    source_user = None
    if to_branch:
        try:
            from huggingface_hub import HfApi
            source_user = HfApi(token=token).whoami().get("name")
        except Exception as e:
            log(f"  ⚠️ couldn't resolve username for episode traceability: {e}")

    if on_progress:
        on_progress(0, 0, "build")
    log(f"Building LeRobot dataset from {len(processed)} episode(s)…")
    build_dataset(repo_id=target_repo, episode_dirs=processed, task=task,
                  root=Path(root), source_user=source_user)
    return processed


def push_lerobot(target_repo, task, root, n_episodes, to_branch=False,
                 token=None, log=print, on_progress=None,
                 gate=None) -> tuple[int, str | None, str]:
    """Push an already-built LeRobot dataset (on disk under `root`) to the Hub.

    Separated from the build so a failed push can be retried without re-running
    SLAM. Loads the dataset from disk, so it works on a fresh process too.

    token: HF token used for the upload (also exported as HF_TOKEN, picked up by
        push_to_hub's internal HfApi).
    to_branch: if True, push to a dedicated branch instead of main (used when the
        target repo already exists, to leave main untouched). We push to a branch
        rather than opening a PR because HF "Sign in with HF" OAuth tokens can
        write content but are not allowed to open PRs (that needs the separate
        discussions/PR permission), so create_pr=True always 403s here.

    Returns (n_episodes, link_or_None, mode) where mode is "main" (link None) or
    "branch" (link = branch URL).
    """
    import os

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if token:
        os.environ["HF_TOKEN"] = token  # picked up by push_to_hub's internal HfApi

    ds = LeRobotDataset(target_repo, root=Path(root))

    if gate:
        gate()  # pause checkpoint before the upload
    if on_progress:
        on_progress(0, 0, "push")
    if to_branch:
        branch = _branch_name(task)
        log(f"Pushing to branch '{branch}' on {target_repo} …")
        _push_to_branch(ds, target_repo, branch, log, token)
        branch_url = f"https://huggingface.co/datasets/{target_repo}/tree/{branch}"
        log(f"✅ Pushed to branch '{branch}'.")
        return n_episodes, branch_url, "branch"

    log(f"Pushing to https://huggingface.co/datasets/{target_repo} …")
    ds.push_to_hub(tags=["lerobot", "grabette"])
    return n_episodes, None, "main"


def process_dataset(dataset_dir, target_repo, task, root, log=print,
                    should_stop=None, to_branch=False, on_progress=None,
                    token=None) -> tuple[int, str | None, str]:
    """Convert + SLAM + build a LeRobot dataset, then push it to the Hub.

    Thin wrapper over build_lerobot() + push_lerobot() for callers that want the
    whole thing in one call. Returns (episode_count, link_or_None, mode).
    """
    processed = build_lerobot(dataset_dir, target_repo, task, root, log=log,
                              should_stop=should_stop, to_branch=to_branch,
                              on_progress=on_progress, token=token)
    return push_lerobot(target_repo, task, root, len(processed), to_branch=to_branch,
                        token=token, log=log, on_progress=on_progress)
