"""OAK-D SR offline VSLAM (pollenrobotics/oak-vslam).

By default the C++ offline_vslam binary runs inside Docker. When the binary is
already present on the host (e.g. inside a HuggingFace Space image that bundles
it), pass `binary=` to run it directly — Docker-in-Docker is not available
there.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DEFAULT_DOCKER_IMAGE = "pollenrobotics/oak-vslam"


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


def _estimate_gravity_imu(imu_acc: pd.DataFrame, g_nominal: float = 9.81,
                          tol: float = 0.5) -> np.ndarray:
    """Robust gravity estimate from a full episode's accel samples.

    Filters samples whose magnitude is within `tol` m/s² of `g_nominal`
    (these are the "stationary-ish" frames where accel is dominated by
    gravity), then returns the median direction. Falls back to the median
    of all samples if too few pass the filter.

    Much more robust than averaging the first N samples — those often
    contain motion-induced accel from grabbing/lifting the device into
    start position.
    """
    accel = imu_acc[["ax", "ay", "az"]].to_numpy()
    mags = np.linalg.norm(accel, axis=1)
    near_g = np.abs(mags - g_nominal) < tol
    if near_g.sum() >= 100:
        filtered = accel[near_g]
        kept_pct = 100.0 * near_g.sum() / len(accel)
        print(f"  gravity estimate: {near_g.sum()}/{len(accel)} samples "
              f"({kept_pct:.1f}%) within {tol} m/s² of {g_nominal}")
    else:
        filtered = accel
        print(f"  gravity estimate: only {near_g.sum()} stationary samples; "
              f"using full-episode median ({len(accel)} samples)")
    return np.median(filtered, axis=0)


def _gravity_align_trajectory(traj_df: pd.DataFrame, oak_dir: Path) -> pd.DataFrame:
    """Rotate trajectory so gravity (in WORLD frame) maps to world -Z.

    rtabmap's IMU init updates the initial pose orientation but does NOT
    redefine the world axes to be gravity-aligned. We post-process the
    integrated trajectory so the saved CSV uses a Z-up world. The rotation
    is a fixed rigid rotation of the world frame — no information is lost
    and camera-local deltas (which the policy consumes) are preserved
    exactly.

    Pipeline:
      1. Robust gravity estimate from accel: median over samples whose
         magnitude is close to 9.81 m/s² (excludes motion-induced accel).
      2. Transform to optical camera frame via calib's imu_to_cam.
      3. Rotate through the first pose's orientation to get gravity in
         the trajectory's world frame.
      4. Find shortest-arc rotation R_align that maps world-gravity to (0,0,-1).
      5. Apply R_align (left-multiply) to all positions and orientations.
    """
    valid = traj_df[~traj_df["is_lost"].astype(bool)]
    if len(valid) == 0:
        return traj_df

    try:
        imu_acc = pd.read_csv(oak_dir / "imu_acc.csv")
        g_imu = _estimate_gravity_imu(imu_acc)
        calib = json.loads((oak_dir / "calib_offline.json").read_text())
        R_imu_to_cam = np.array(calib["imu_to_cam"])[:3, :3]
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  gravity-align skipped: {e}")
        return traj_df

    g_cam = R_imu_to_cam @ g_imu
    g_cam_unit = g_cam / np.linalg.norm(g_cam)

    first_quat = valid[["q_x", "q_y", "q_z", "q_w"]].iloc[0].to_numpy(copy=True)
    R_world_from_cam0 = Rotation.from_quat(first_quat)
    g_world_unit = R_world_from_cam0.apply(g_cam_unit)

    R_align, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [g_world_unit])

    positions = traj_df[["x", "y", "z"]].to_numpy(copy=True)
    quaternions = traj_df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(copy=True)
    new_positions = R_align.apply(positions)
    new_quats = (R_align * Rotation.from_quat(quaternions)).as_quat()

    out = traj_df.copy()
    out[["x", "y", "z"]] = new_positions
    out[["q_x", "q_y", "q_z", "q_w"]] = new_quats
    return out


def _integrate_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Integrate frame-to-frame delta poses into absolute poses.

    Input columns: timestamp_s, dx, dy, dz, dqx, dqy, dqz, dqw, lost, ...
    Output columns match the standard trajectory CSV format so the rest of
    the pipeline (trajectory.py, generate_dataset.py) can consume it unchanged.

    Lost frames hold the last known absolute pose (no motion accumulated
    when tracking fails).
    """
    n = len(delta_df)
    positions = np.zeros((n, 3))
    quaternions = np.zeros((n, 4))  # scipy convention: (x, y, z, w)

    abs_pos = np.zeros(3)
    abs_rot = Rotation.identity()

    for i, row in enumerate(delta_df.itertuples(index=False)):
        if not row.lost:
            d_t = np.array([row.dx, row.dy, row.dz])
            d_r = Rotation.from_quat([row.dqx, row.dqy, row.dqz, row.dqw])
            # SE3 composition: T_i = T_{i-1} * delta_i
            abs_pos = abs_pos + abs_rot.apply(d_t)
            abs_rot = abs_rot * d_r

        positions[i] = abs_pos
        quaternions[i] = abs_rot.as_quat()

    return pd.DataFrame({
        "frame_idx": np.arange(n),
        "timestamp": delta_df["timestamp_s"].values,
        "state": 2,
        "is_lost": delta_df["lost"].values,
        "is_keyframe": 0,
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "q_x": quaternions[:, 0],
        "q_y": quaternions[:, 1],
        "q_z": quaternions[:, 2],
        "q_w": quaternions[:, 3],
    })


def _slam_command(oak_dir: Path, *, docker_image: str, binary: str | None) -> list[str]:
    """Build the offline_vslam invocation.

    With `binary`, call it directly on the host (e.g. inside a Space image that
    bundles it). Otherwise run it in Docker, mounting oak_dir at /data.
    """
    poses_path = oak_dir / "poses.csv"
    if binary:
        return [binary, str(oak_dir), str(poses_path)]
    return [
        "docker", "run", "--rm",
        "--volume", f"{oak_dir}:/data",
        docker_image,
        "/data", "/data/poses.csv",
    ]


def run_oak_slam(
    episode_dir: Path,
    *,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    binary: str | None = None,
    output_csv: str = "camera_trajectory.csv",
    show_progress: bool = True,
) -> SlamResult:
    """Run offline_vslam on the oak/ subdir of an episode directory.

    Produces <episode_dir>/<output_csv> with absolute poses in the standard
    trajectory format (frame_idx, timestamp, state, is_lost, is_keyframe,
    x, y, z, q_x, q_y, q_z, q_w).

    By default the binary runs in Docker; the image must be built once with:
        docker build -t pollenrobotics/oak-vslam docker/oak_vslam/

    Args:
        episode_dir: directory containing oak/ with frames, depth, IMU CSVs, calib
        docker_image: Docker image name (ignored when `binary` is set)
        binary: path to a host offline_vslam binary; if set, run it directly
            instead of via Docker (Docker-in-Docker is unavailable in Spaces)
        output_csv: trajectory output filename (inside episode_dir)
        show_progress: print SLAM stdout in real time

    Returns:
        SlamResult with tracking statistics
    """
    episode_dir = Path(episode_dir).absolute()
    oak_dir = episode_dir / "oak"

    if not oak_dir.is_dir():
        raise FileNotFoundError(f"No oak/ subdir in {episode_dir}")

    poses_path = oak_dir / "poses.csv"
    cmd = _slam_command(oak_dir, docker_image=docker_image, binary=binary)

    if show_progress:
        print(f"Running OAK SLAM on {episode_dir.name}...")

    log_path = episode_dir / "oak_slam_stdout.txt"
    returncode = -1
    try:
        with open(log_path, "w") as f_log:
            proc = subprocess.run(
                cmd,
                stdout=None if show_progress else f_log,
                stderr=subprocess.STDOUT if not show_progress else None,
                text=True,
                check=False,
            )
        returncode = proc.returncode
    except Exception as e:
        print(f"  OAK SLAM error: {e}")
        return SlamResult(
            returncode=-1, total_frames=0, tracked_frames=0, trajectory_path=None
        )

    if not poses_path.is_file():
        print(f"  OAK SLAM produced no poses.csv (rc={returncode})")
        return SlamResult(
            returncode=returncode, total_frames=0, tracked_frames=0, trajectory_path=None
        )

    delta_df = pd.read_csv(poses_path)
    abs_df = _integrate_deltas(delta_df)
    abs_df = _gravity_align_trajectory(abs_df, oak_dir)

    traj_path = episode_dir / output_csv
    abs_df.to_csv(traj_path, index=False)

    total = len(abs_df)
    tracked = int((~abs_df["is_lost"].astype(bool)).sum())

    if show_progress:
        pct = 100.0 * tracked / total if total else 0.0
        print(f"  Tracking: {tracked}/{total} ({pct:.1f}%)")

    return SlamResult(
        returncode=returncode,
        total_frames=total,
        tracked_frames=tracked,
        trajectory_path=traj_path,
    )
