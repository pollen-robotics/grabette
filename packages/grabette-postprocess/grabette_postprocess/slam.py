"""Docker-based SLAM orchestration.

Runs grabette_slam inside Docker for mapping (pass 1 + pass 2) and
batch localization against an existing map.
"""

import concurrent.futures
import json
import multiprocessing
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from grabette_postprocess.imu import prepare_imu_for_slam
from grabette_postprocess.mask import generate_mask

# Default Docker image name (local build, no hub pull)
DEFAULT_DOCKER_IMAGE = "pollenrobotics/orbslam3-headless"

# Default SLAM settings — shipped with this package
_PACKAGE_DIR = Path(__file__).parent
DEFAULT_SETTINGS = _PACKAGE_DIR.parent / "config" / "rpi_bmi088_slam_settings.yaml"


def _check_docker():
    """Verify Docker daemon is running. Raises RuntimeError if not."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Docker is not running. Start it with: sudo systemctl start docker"
            )
    except FileNotFoundError:
        raise RuntimeError("Docker is not installed.")


@dataclass
class SlamResult:
    """Result from a single SLAM run."""
    returncode: int
    total_frames: int
    tracked_frames: int
    trajectory_path: Path | None
    abort_reason: str | None = None

    @property
    def tracking_pct(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return 100.0 * self.tracked_frames / self.total_frames


def _ensure_imu_resampled(video_dir: Path) -> str:
    """Resample IMU if not already done. Returns IMU filename to use."""
    resampled = video_dir / "imu_data_resampled.json"
    if not resampled.is_file():
        raw = video_dir / "imu_data.json"
        if not raw.is_file():
            raise FileNotFoundError(f"No imu_data.json in {video_dir}")
        print(f"Resampling IMU to uniform 200Hz...")
        prepare_imu_for_slam(raw, resampled)
    return "imu_data_resampled.json"


def _ensure_mask(video_dir: Path, video_path: Path) -> str:
    """Generate SLAM mask if not already present. Returns mask filename."""
    mask_path = video_dir / "slam_mask.png"
    if not mask_path.is_file():
        # Get video resolution for mask
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            w, h = stream.width, stream.height
        mask = generate_mask(w, h)
        cv2.imwrite(str(mask_path), mask)
    return "slam_mask.png"


def _parse_tracking_rate(trajectory_path: Path) -> tuple[int, int]:
    """Parse trajectory CSV and return (total_frames, tracked_frames)."""
    if not trajectory_path.is_file():
        return 0, 0
    df = pd.read_csv(trajectory_path)
    total = len(df)
    tracked = total - int(df['is_lost'].sum())
    return total, tracked


def _build_docker_cmd(
    video_dir: Path,
    *,
    imu_filename: str,
    output_csv: str,
    settings_path: Path,
    save_map: Path | None = None,
    load_map: Path | None = None,
    mask: bool = True,
    output_gravity: str | None = None,
    output_biases: str | None = None,
    max_lost_frames: int = -1,
    deterministic: bool = False,
    max_lost_pct: float = -1,
    warmup_frames: int = 300,
    frame_skip: int = 1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
) -> tuple[list[str], str]:
    """Build the docker run command list. Returns (cmd, container_name)."""
    # Docker mount points
    data_mount = "/data"
    settings_mount = "/settings"

    # Unique container name so we can docker-kill it on timeout
    container_name = f"grabette-slam-{uuid.uuid4().hex[:8]}"

    cmd = [
        "docker", "run", "--rm", "-t",
        "--name", container_name,
        "--volume", f"{video_dir}:{data_mount}",
    ]

    # Mount map directory if saving or loading
    if save_map is not None:
        cmd.extend(["--volume", f"{save_map.parent}:/map"])
    if load_map is not None:
        cmd.extend(["--volume", f"{load_map.parent}:/map"])

    # Mount settings
    cmd.extend(["--volume", f"{settings_path.parent}:{settings_mount}"])

    # Image + binary
    cmd.extend([
        docker_image,
        "/ORB_SLAM3/Examples/Monocular-Inertial/grabette_slam",
        "--vocabulary", "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
        "--setting", f"{settings_mount}/{settings_path.name}",
        "--input_video", f"{data_mount}/raw_video.mp4",
        "--input_imu_json", f"{data_mount}/{imu_filename}",
        "--output_trajectory_csv", f"{data_mount}/{output_csv}",
    ])

    if save_map is not None:
        cmd.extend(["--save_map", f"/map/{save_map.name}"])
    if load_map is not None:
        cmd.extend(["--load_map", f"/map/{load_map.name}"])
    if mask:
        cmd.extend(["--mask_img", f"{data_mount}/slam_mask.png"])
    if output_gravity:
        cmd.extend(["--output_gravity", f"{data_mount}/{output_gravity}"])
    if output_biases:
        cmd.extend(["--output_biases", f"{data_mount}/{output_biases}"])
    if max_lost_frames > 0:
        cmd.extend(["--max_lost_frames", str(max_lost_frames)])
    if deterministic:
        cmd.append("--deterministic")
    if max_lost_pct > 0:
        cmd.extend(["--max_lost_pct", str(max_lost_pct)])
        cmd.extend(["--warmup_frames", str(warmup_frames)])
    if frame_skip > 1:
        cmd.extend(["--frame_skip", str(frame_skip)])

    return cmd, container_name


def _read_slam_pipe(pipe, stdout_path: Path, show_progress: bool,
                    abort_event: threading.Event | None = None):
    """Reader thread: drain pipe line-by-line, write to log, optionally show progress."""
    pbar = None
    total_frames = None
    n_lost = 0

    with open(stdout_path, "w") as f_out:
        for raw_line in pipe:
            line = raw_line.rstrip("\r\n")
            f_out.write(line + "\n")
            f_out.flush()

            # Detect map reset — unrecoverable in localization mode
            if abort_event is not None and "Reseting active map" in line:
                abort_event.set()

            if not show_progress:
                continue

            if total_frames is None:
                m = re.search(r"There are (\d+) frames in total", line)
                if m:
                    total_frames = int(m.group(1))
                    pbar = tqdm(total=total_frames, unit="fr",
                                desc="  SLAM", leave=True)

            if pbar is not None and "Video FPS:" in line:
                pbar.update(100 - (pbar.n % 100) if pbar.n % 100 else 100)

            if "n_lost_frames=" in line:
                m = re.search(r"n_lost_frames=(\d+)", line)
                if m:
                    n_lost = int(m.group(1))
                    if pbar is not None:
                        pbar.set_postfix(lost=n_lost)

    # Pipe closed (process exited or was killed)
    if pbar is not None:
        if total_frames:
            pbar.n = total_frames
            pbar.refresh()
        pbar.close()


def run_slam(
    video_dir: Path,
    *,
    output_csv: str = "camera_trajectory.csv",
    save_map: Path | None = None,
    load_map: Path | None = None,
    output_gravity: str | None = None,
    output_biases: str | None = None,
    mask: bool = True,
    max_lost_frames: int = -1,
    deterministic: bool = False,
    max_lost_pct: float = -1,
    warmup_frames: int = 300,
    frame_skip: int = 1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_path: Path = DEFAULT_SETTINGS,
    timeout_s: float | None = None,
    log_prefix: str = "slam",
    show_progress: bool = True,
    abort_on_map_reset: bool = False,
) -> SlamResult:
    """Run grabette_slam in Docker on a single video directory.

    Args:
        video_dir: directory containing raw_video.mp4 and imu_data.json
        output_csv: trajectory output filename (inside video_dir)
        save_map: if set, save map atlas to this path
        load_map: if set, load map atlas from this path
        output_gravity: gravity output filename (inside video_dir), or None
        output_biases: biases output filename (inside video_dir), or None
        mask: generate and apply device mask
        max_lost_frames: terminate after N lost frames (-1 = disabled)
        docker_image: Docker image name
        settings_path: SLAM settings YAML path
        timeout_s: subprocess timeout in seconds
        log_prefix: prefix for stdout/stderr log files
        show_progress: show tqdm progress bar

    Returns:
        SlamResult with tracking statistics
    """
    video_dir = Path(video_dir).absolute()
    settings_path = Path(settings_path).absolute()

    # Ensure prerequisites
    imu_filename = _ensure_imu_resampled(video_dir)
    if mask:
        _ensure_mask(video_dir, video_dir / "raw_video.mp4")

    cmd, container_name = _build_docker_cmd(
        video_dir,
        imu_filename=imu_filename,
        output_csv=output_csv,
        settings_path=settings_path,
        save_map=save_map,
        load_map=load_map,
        mask=mask,
        output_gravity=output_gravity,
        output_biases=output_biases,
        max_lost_frames=max_lost_frames,
        deterministic=deterministic,
        max_lost_pct=max_lost_pct,
        warmup_frames=warmup_frames,
        frame_skip=frame_skip,
        docker_image=docker_image,
    )

    stdout_path = video_dir / f"{log_prefix}_stdout.txt"
    stderr_path = video_dir / f"{log_prefix}_stderr.txt"

    # Pipe stdout through a reader thread for real-time progress.
    # Docker -t flag gives line-buffered output inside the container,
    # pipe gives low-latency delivery to the reader thread.
    # Stderr goes to a file (no need for real-time monitoring).
    abort_event = threading.Event() if abort_on_map_reset else None
    abort_reason = None
    returncode = -1
    reader = None
    try:
        with open(stderr_path, "w") as f_err:
            proc = subprocess.Popen(
                cmd,
                cwd=str(video_dir),
                stdout=subprocess.PIPE,
                stderr=f_err,
                text=True,
            )
            reader = threading.Thread(
                target=_read_slam_pipe,
                args=(proc.stdout, stdout_path, show_progress, abort_event),
                daemon=True,
            )
            reader.start()

            start_time = time.monotonic()
            while True:
                try:
                    proc.wait(timeout=0.5)
                    returncode = proc.returncode
                    break
                except subprocess.TimeoutExpired:
                    pass
                # Check abort event
                if abort_event is not None and abort_event.is_set():
                    abort_reason = "map_reset"
                    break
                # Check timeout
                if timeout_s is not None and time.monotonic() - start_time > timeout_s:
                    abort_reason = "timeout"
                    break

            if abort_reason:
                subprocess.run(
                    ["docker", "kill", container_name],
                    capture_output=True, timeout=10,
                )
                proc.wait(timeout=10)
                if show_progress:
                    if abort_reason == "timeout":
                        print(f"\n  SLAM timed out after {timeout_s:.0f}s")
                    else:
                        print(f"\n  SLAM aborted: {abort_reason}")
                returncode = -1

    except Exception as e:
        print(f"  SLAM error: {e}")
        returncode = -1
    finally:
        if reader and reader.is_alive():
            reader.join(timeout=5)

    traj_path = video_dir / output_csv
    total, tracked = _parse_tracking_rate(traj_path)

    return SlamResult(
        returncode=returncode,
        total_frames=total,
        tracked_frames=tracked,
        trajectory_path=traj_path if traj_path.is_file() else None,
        abort_reason=abort_reason,
    )


def _copy_file(src: Path, dst: Path):
    """Copy file handling root-owned Docker output files."""
    if src.is_file():
        data = src.read_bytes()
        if dst.is_file():
            os.remove(str(dst))
        dst.write_bytes(data)


def _run_attempt(
    attempt: int,
    video_dir: Path,
    map_dir: Path,
    *,
    docker_image: str,
    settings_path: Path,
    deterministic: bool = False,
    max_lost_pct: float = -1,
    warmup_frames: int = 300,
    frame_skip: int = 1,
    show_progress: bool = True,
) -> tuple[int, SlamResult]:
    """Run a single pass-1 mapping attempt. Returns (attempt_number, result)."""
    result = run_slam(
        video_dir,
        output_csv=f"mapping_traj_attempt{attempt}.csv",
        save_map=map_dir / f"map_atlas_attempt{attempt}.osa",
        output_gravity=f"gravity_attempt{attempt}.csv",
        output_biases=f"biases_attempt{attempt}.csv",
        deterministic=deterministic,
        max_lost_pct=max_lost_pct,
        warmup_frames=warmup_frames,
        frame_skip=frame_skip,
        docker_image=docker_image,
        settings_path=settings_path,
        log_prefix=f"slam_attempt{attempt}",
        show_progress=show_progress,
    )
    return attempt, result


def create_map(
    video_dir: Path,
    *,
    retries: int = 3,
    parallel: int = 1,
    deterministic: bool = False,
    max_lost_pct: float = -1,
    warmup_frames: int = 300,
    frame_skip: int = 1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_path: Path = DEFAULT_SETTINGS,
) -> Path:
    """Two-pass mapping: pass 1 (save_map + gravity/biases), pass 2 (load_map).

    Pass 1 runs 1+retries attempts (sequentially or in parallel), keeps best by
    tracking rate. Pass 2 re-localizes against the best map.

    When deterministic=True, forces retries=0 and parallel=1 (single
    deterministic pass — no retries needed since results are reproducible).

    Args:
        video_dir: directory containing raw_video.mp4 and imu_data.json
        retries: number of extra attempts for pass 1 (total = 1 + retries)
        parallel: number of pass-1 attempts to run simultaneously
        deterministic: run in deterministic mode (slower, reproducible)
        max_lost_pct: abort attempt if lost rate exceeds this after warmup (-1=disabled)
        warmup_frames: frames before checking lost rate
        docker_image: Docker image name
        settings_path: SLAM settings YAML path

    Returns:
        Path to map_atlas.osa
    """
    video_dir = Path(video_dir).absolute()
    _check_docker()

    map_dir = video_dir / "map"
    map_dir.mkdir(exist_ok=True)
    map_path = map_dir / "map_atlas.osa"

    # Deterministic mode: single pass, no retries needed
    if deterministic:
        retries = 0
        parallel = 1

    total_attempts = 1 + retries

    # Ensure prerequisites once before launching attempts
    _ensure_imu_resampled(video_dir)
    _ensure_mask(video_dir, video_dir / "raw_video.mp4")

    # --- Pass 1: Mapping ---
    if parallel <= 1:
        # Sequential mode: run one at a time with progress bars, early-stop at 90%
        results = _pass1_sequential(
            video_dir, map_dir, total_attempts,
            docker_image=docker_image, settings_path=settings_path,
            deterministic=deterministic,
            max_lost_pct=max_lost_pct, warmup_frames=warmup_frames,
            frame_skip=frame_skip,
        )
    else:
        # Parallel mode: launch all attempts at once
        results = _pass1_parallel(
            video_dir, map_dir, total_attempts, parallel,
            docker_image=docker_image, settings_path=settings_path,
            max_lost_pct=max_lost_pct, warmup_frames=warmup_frames,
            frame_skip=frame_skip,
        )

    # Pick the best attempt
    best_attempt, best_result = max(
        ((a, r) for a, r in results if r.trajectory_path is not None),
        key=lambda x: x[1].tracking_pct,
        default=(0, None),
    )

    if best_result is None or best_result.tracking_pct <= 0:
        raise RuntimeError(f"All {total_attempts} SLAM attempts failed")

    best_pct = best_result.tracking_pct
    print(f"\nBest result: attempt {best_attempt}/{total_attempts} "
          f"({best_result.tracked_frames}/{best_result.total_frames}, {best_pct:.1f}%)")

    # Copy best attempt's output files to canonical names
    _copy_file(video_dir / f"mapping_traj_attempt{best_attempt}.csv",
               video_dir / "mapping_camera_trajectory.csv")
    _copy_file(map_dir / f"map_atlas_attempt{best_attempt}.osa", map_path)
    _copy_file(video_dir / f"gravity_attempt{best_attempt}.csv",
               video_dir / "gravity.csv")
    _copy_file(video_dir / f"biases_attempt{best_attempt}.csv",
               video_dir / "biases.csv")

    # Save metadata for the mapping video
    save_slam_metadata(
        video_dir, best_result,
        method="mapping",
        deterministic=deterministic,
        docker_image=docker_image,
        settings_file=settings_path.name,
        frame_skip=frame_skip,
    )

    # Deterministic mode: single pass is sufficient, skip pass 2.
    # The result is already reproducible — no need to re-localize.
    if deterministic:
        return map_path

    # --- Pass 2: Re-localization ---
    print("\nRunning pass 2 (re-localization to recover init frames)...")

    # Timeout: pass 2 should be faster than pass 1 (no mapping, just localization).
    with av.open(str(video_dir / "raw_video.mp4")) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
    pass2_timeout = max(duration * 5, 180)

    result2 = run_slam(
        video_dir,
        output_csv="mapping_camera_trajectory_pass2.csv",
        load_map=map_path,
        frame_skip=frame_skip,
        docker_image=docker_image,
        settings_path=settings_path,
        timeout_s=pass2_timeout,
        log_prefix="slam_pass2",
        abort_on_map_reset=True,
    )

    # Use pass 2 result if trajectory was written — even if the process timed out
    # (the binary may hang during shutdown after successfully writing the trajectory)
    if result2.trajectory_path is not None and result2.tracked_frames > 0:
        pct2 = result2.tracking_pct
        if result2.returncode != 0:
            print(f"  Pass 2: process exited with rc={result2.returncode} "
                  f"but trajectory exists: {result2.tracked_frames}/{result2.total_frames} ({pct2:.1f}%)")
        else:
            print(f"  Pass 2: {result2.tracked_frames}/{result2.total_frames} ({pct2:.1f}%)")

        if pct2 > best_pct:
            _copy_file(
                video_dir / "mapping_camera_trajectory_pass2.csv",
                video_dir / "mapping_camera_trajectory.csv",
            )
            # Update metadata with pass 2 result
            save_slam_metadata(
                video_dir, result2,
                method="mapping",
                deterministic=deterministic,
                docker_image=docker_image,
                settings_file=settings_path.name,
                frame_skip=frame_skip,
            )
            print(f"  Pass 2 improved tracking: {best_pct:.1f}% -> {pct2:.1f}%")
        else:
            print(f"  Pass 2 did not improve ({pct2:.1f}% vs {best_pct:.1f}%), keeping pass 1")
    else:
        print(f"  Pass 2 failed, keeping pass 1")

    return map_path


def _pass1_sequential(
    video_dir: Path,
    map_dir: Path,
    total_attempts: int,
    *,
    docker_image: str,
    settings_path: Path,
    deterministic: bool = False,
    max_lost_pct: float = -1,
    warmup_frames: int = 300,
    frame_skip: int = 1,
) -> list[tuple[int, SlamResult]]:
    """Run pass-1 attempts sequentially with progress bars. Early-stops at >=90%."""
    results = []
    for attempt in range(1, total_attempts + 1):
        if total_attempts > 1:
            print(f"\n--- Pass 1, attempt {attempt}/{total_attempts} ---")
        else:
            print("Running SLAM mapping (pass 1)...")

        attempt_num, result = _run_attempt(
            attempt, video_dir, map_dir,
            docker_image=docker_image, settings_path=settings_path,
            deterministic=deterministic,
            max_lost_pct=max_lost_pct, warmup_frames=warmup_frames,
            frame_skip=frame_skip,
            show_progress=True,
        )
        results.append((attempt_num, result))

        if result.trajectory_path is None or result.tracked_frames == 0:
            print(f"  SLAM failed (return code {result.returncode})")
            continue

        pct = result.tracking_pct
        print(f"  Tracking: {result.tracked_frames}/{result.total_frames} ({pct:.1f}%)")

        if pct >= 90:
            if total_attempts > 1:
                print(f"  >= 90% tracking, stopping early")
            break

    return results


def _pass1_parallel(
    video_dir: Path,
    map_dir: Path,
    total_attempts: int,
    parallel: int,
    *,
    docker_image: str,
    settings_path: Path,
    max_lost_pct: float = -1,
    warmup_frames: int = 300,
    frame_skip: int = 1,
) -> list[tuple[int, SlamResult]]:
    """Run pass-1 attempts in parallel (no per-attempt progress bars)."""
    n_workers = min(parallel, total_attempts)
    print(f"\nRunning {total_attempts} pass-1 attempts ({n_workers} in parallel)...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for attempt in range(1, total_attempts + 1):
            fut = executor.submit(
                _run_attempt,
                attempt, video_dir, map_dir,
                docker_image=docker_image, settings_path=settings_path,
                max_lost_pct=max_lost_pct, warmup_frames=warmup_frames,
                frame_skip=frame_skip,
                show_progress=False,
            )
            futures[fut] = attempt

        for fut in concurrent.futures.as_completed(futures):
            attempt_num, result = fut.result()
            results.append((attempt_num, result))

            if result.trajectory_path is None or result.tracked_frames == 0:
                print(f"  Attempt {attempt_num}: FAILED (rc={result.returncode})")
            else:
                pct = result.tracking_pct
                print(f"  Attempt {attempt_num}: "
                      f"{result.tracked_frames}/{result.total_frames} ({pct:.1f}%)")

    return results


def save_slam_metadata(
    video_dir: Path,
    result: SlamResult,
    *,
    method: str,
    map_file: Path | None = None,
    deterministic: bool = False,
    frame_skip: int = 1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_file: str = "",
    localization_result: SlamResult | None = None,
):
    """Save SLAM run metadata to slam_metadata.json in the episode directory."""
    metadata = {
        "method": method,
        "map_file": str(map_file) if map_file else None,
        "tracking_pct": round(result.tracking_pct, 2),
        "tracked_frames": result.tracked_frames,
        "total_frames": result.total_frames,
        "returncode": result.returncode,
        "abort_reason": result.abort_reason,
        "deterministic": deterministic,
        "frame_skip": frame_skip,
        "docker_image": docker_image,
        "settings_file": settings_file,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if localization_result is not None:
        metadata["localization_tracking_pct"] = round(localization_result.tracking_pct, 2)
        metadata["localization_abort_reason"] = localization_result.abort_reason
    with open(video_dir / "slam_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def _get_video_duration(video_dir: Path) -> float:
    """Get video duration in seconds."""
    with av.open(str(video_dir / "raw_video.mp4")) as container:
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base)


def batch_slam(
    video_dirs: list[Path],
    map_path: Path | None,
    *,
    num_workers: int | None = None,
    max_lost_frames: int = 60,
    timeout_multiple: float = 16,
    deterministic: bool = False,
    min_tracking_pct: float = 50.0,
    retry_mapping: bool = True,
    force: bool = False,
    frame_skip: int = 1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_path: Path = DEFAULT_SETTINGS,
):
    """Run SLAM on multiple videos. With map_path: localize then retry failures.
    Without map_path (--mapping-only): run independent mapping on each episode.

    Phase 1: Localize each video against the shared map. Aborts early if the
    SLAM system resets its map (unrecoverable in localization-only mode).

    Phase 2 (if retry_mapping=True): Episodes that failed localization or
    tracked below min_tracking_pct are retried in full mapping mode (SLAM
    builds its own map from scratch).

    Saves slam_metadata.json in each episode directory.

    Args:
        video_dirs: list of directories, each with raw_video.mp4 + imu_data.json
        map_path: path to map_atlas.osa, or None for mapping-only mode
        num_workers: parallel Docker containers (default: cpu_count // 2)
        max_lost_frames: terminate individual runs after N lost frames
        timeout_multiple: timeout = video_duration * this
        deterministic: run in deterministic mode (slower, reproducible)
        min_tracking_pct: retry episodes below this tracking percentage
        retry_mapping: retry failed episodes in full mapping mode
        force: reprocess episodes that already have camera_trajectory.csv
        docker_image: Docker image name
        settings_path: SLAM settings YAML path
    """
    _check_docker()

    mapping_only = map_path is None
    if not mapping_only:
        map_path = Path(map_path).absolute()
        if not map_path.is_file():
            raise FileNotFoundError(f"Map not found: {map_path}")
    settings_path = Path(settings_path).absolute()

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    # Prepare all directories (IMU resample + mask) and compute timeouts
    to_process = []
    for vdir in video_dirs:
        vdir = Path(vdir).absolute()
        if not force and (vdir / "camera_trajectory.csv").is_file():
            print(f"  Skipping {vdir.name} (camera_trajectory.csv exists)")
            continue
        if force and (vdir / "camera_trajectory.csv").is_file():
            os.remove(str(vdir / "camera_trajectory.csv"))
        _ensure_imu_resampled(vdir)
        _ensure_mask(vdir, vdir / "raw_video.mp4")
        timeout = _get_video_duration(vdir) * timeout_multiple
        to_process.append((vdir, timeout))

    if not to_process:
        print("Nothing to process.")
        return

    # ---- Phase 1: Localization ----
    loc_results: dict[Path, SlamResult] = {}
    failures = []

    # ---- Phase 1: Localization (skip if mapping-only) ----
    if not mapping_only:
        print(f"\nPhase 1: Localizing {len(to_process)} episodes against map...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for vdir, timeout in to_process:
                fut = executor.submit(
                    run_slam, vdir,
                    output_csv="camera_trajectory.csv",
                    load_map=map_path,
                    max_lost_frames=max_lost_frames,
                    deterministic=deterministic,
                    frame_skip=frame_skip,
                    docker_image=docker_image,
                    settings_path=settings_path,
                    timeout_s=timeout,
                    log_prefix="slam",
                    show_progress=False,
                    abort_on_map_reset=True,
                )
                futures[fut] = vdir

            for fut in concurrent.futures.as_completed(futures):
                vdir = futures[fut]
                result = fut.result()
                loc_results[vdir] = result

                if result.abort_reason:
                    print(f"  {vdir.name}: ABORTED ({result.abort_reason})")
                elif result.tracking_pct == 0:
                    print(f"  {vdir.name}: FAILED (rc={result.returncode})")
                else:
                    pct = result.tracking_pct
                    status = "OK" if pct >= min_tracking_pct else "LOW"
                    print(f"  {vdir.name}: {result.tracked_frames}/{result.total_frames} ({pct:.1f}%) [{status}]")

        # Identify failures
        for vdir, timeout in to_process:
            result = loc_results[vdir]
            if (result.abort_reason
                    or result.tracking_pct < min_tracking_pct
                    or result.tracked_frames == 0):
                failures.append((vdir, timeout))
    else:
        # Mapping-only: all episodes go to mapping phase
        failures = list(to_process)

    # ---- Phase 2: Mapping ----
    map_results: dict[Path, SlamResult] = {}

    if (retry_mapping or mapping_only) and failures:
        if mapping_only:
            print(f"\nMapping {len(failures)} episodes...")
        else:
            print(f"\nPhase 2: Retrying {len(failures)} episodes in mapping mode...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for vdir, timeout in failures:
                # Remove failed localization trajectory
                traj = vdir / "camera_trajectory.csv"
                if traj.is_file():
                    os.remove(str(traj))

                fut = executor.submit(
                    run_slam, vdir,
                    output_csv="camera_trajectory.csv",
                    # No load_map — full SLAM
                    output_gravity="gravity.csv",
                    output_biases="biases.csv",
                    deterministic=deterministic,
                    frame_skip=frame_skip,
                    docker_image=docker_image,
                    settings_path=settings_path,
                    timeout_s=timeout,
                    log_prefix="slam_retry",
                    show_progress=False,
                    abort_on_map_reset=False,
                )
                futures[fut] = vdir

            for fut in concurrent.futures.as_completed(futures):
                vdir = futures[fut]
                result = fut.result()
                map_results[vdir] = result

                if result.tracking_pct == 0:
                    print(f"  {vdir.name}: FAILED (rc={result.returncode})")
                else:
                    pct = result.tracking_pct
                    print(f"  {vdir.name}: {result.tracked_frames}/{result.total_frames} ({pct:.1f}%)")

    # ---- Save metadata ----
    for vdir, _ in to_process:
        loc_result = loc_results.get(vdir)
        map_result = map_results.get(vdir)

        if map_result is not None:
            final_result = map_result
            method = "mapping"
        elif loc_result is not None:
            final_result = loc_result
            method = "localization"
        else:
            continue

        save_slam_metadata(
            vdir, final_result,
            method=method,
            map_file=map_path if method == "localization" else None,
            deterministic=deterministic,
            frame_skip=frame_skip,
            docker_image=docker_image,
            settings_file=settings_path.name,
            localization_result=loc_result if map_result is not None else None,
        )

    # ---- Summary ----
    n_loc_ok = sum(
        1 for vdir, _ in to_process
        if vdir not in map_results
        and vdir in loc_results
        and loc_results[vdir].tracking_pct >= min_tracking_pct
    )
    n_mapping_ok = sum(1 for r in map_results.values() if r.tracking_pct > 0)
    n_failed = len(to_process) - n_loc_ok - n_mapping_ok

    print(f"\nBatch SLAM complete: {len(to_process)} episodes")
    if not mapping_only:
        print(f"  Localized:      {n_loc_ok}")
    if map_results:
        label = "Mapped" if mapping_only else "Mapping retry"
        print(f"  {label}:  {n_mapping_ok}")
    if n_failed > 0:
        print(f"  Failed:         {n_failed}")
