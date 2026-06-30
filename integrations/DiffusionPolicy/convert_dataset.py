"""Convert Grabette dataset for diffusion policy training.

Transforms the dataset:
  1. Converts rotation from axis-angle (3D) to 6D continuous representation.
  2. Computes delta actions: action[t] = pose[t+1] - pose[t] for position/rotation,
     gripper stays absolute.
  3. Builds observation.state depending on --proprioception mode:

     --proprioception none (default):
       observation.state = [proximal, distal]  (2D)
       Model sees camera + gripper only. Simplest approach.

     --proprioception relative:
       observation.state = [dx_start, dy_start, dz_start, r6d_rel_0..5, proximal, distal]  (11D)
       Pose relative to episode start, expressed in the START camera frame
       (R_start^T @ (pos - pos_start), R_start^T @ R) — same gripper-egocentric
       convention as the action deltas, so it's frame-independent (the arbitrary
       per-session world orientation cancels). The model knows how far it
       moved/rotated from the episode start, independent of the world frame.

Usage:
  # Gripper-only state (2D):
  uv run python convert_dataset.py \\
      --repo_id SteveNguyen/Grabette_redcube_quest

  # With relative proprioception (11D, UMI-style):
  uv run python convert_dataset.py \\
      --repo_id SteveNguyen/Grabette_redcube_quest --proprioception relative
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets import LeRobotDataset
from rotation import (
    rotation_6d_to_rotation_matrix_numpy,
    rotation_matrix_to_rotation_6d_numpy,
    rotvec_to_rotation_6d,
)

logger = logging.getLogger(__name__)

# Feature names after conversion
ACTION_NAMES = [
    "dx",
    "dy",
    "dz",
    "dr6d_0",
    "dr6d_1",
    "dr6d_2",
    "dr6d_3",
    "dr6d_4",
    "dr6d_5",
    "proximal",
    "distal",
]
STATE_NAMES_NONE = ["proximal", "distal"]
STATE_NAMES_RELATIVE = [
    "dx_start",
    "dy_start",
    "dz_start",
    "r6d_rel_0",
    "r6d_rel_1",
    "r6d_rel_2",
    "r6d_rel_3",
    "r6d_rel_4",
    "r6d_rel_5",
    "proximal",
    "distal",
]


def pose_8d_to_11d(data_8d: np.ndarray) -> np.ndarray:
    """Convert 8D (pos + axis-angle + gripper) to 11D (pos + rot6d + gripper).

    Args:
        data_8d: (N, 8) — [x, y, z, ax, ay, az, proximal, distal]

    Returns:
        (N, 11) — [x, y, z, r6d_0..5, proximal, distal]
    """
    pos = data_8d[:, :3]
    rotvec = data_8d[:, 3:6]
    gripper = data_8d[:, 6:]

    rot6d = rotvec_to_rotation_6d(rotvec).astype(np.float32)

    return np.concatenate([pos, rot6d, gripper], axis=1)


def compute_delta_actions(poses_11d: np.ndarray, episode_indices: np.ndarray) -> np.ndarray:
    """Compute per-frame delta actions in the CAMERA-LOCAL frame.

    The recorded poses are camera-site SE(3) in a (per-session arbitrary)
    world frame: gravity-aligned Z, but the X/Y horizontal axes depend on
    the SLAM yaw at session start. World-frame deltas are therefore NOT
    portable across sessions — a delta of (+5 mm, 0, 0) recorded in one
    session points in a different physical direction in another.

    To make actions session-invariant, we express each delta in the
    camera's local frame at time t:

        delta_pos_local[t]  = R(cam_t)^T @ (pos[t+1] - pos[t])
        R_delta_local[t]    = R(cam_t)^T @ R(cam_{t+1})           # proper composition
        delta_r6d_local[t]  = rotation_matrix_to_6d(R_delta_local)

    At deployment, the arm-side controller reads its current camera pose
    via FK and applies:

        target_pos = current_pos + R(current_cam) @ delta_pos_local
        target_rot = R(current_cam) @ R_delta_local

    which is fully invariant to any rotation of the world frame around Z
    (the arbitrary part of the SLAM origin).

    Gripper dims (9, 10) stay absolute — no frame to worry about.

    At episode boundaries the delta is zeroed (no next frame).
    """
    n = len(poses_11d)
    actions = np.zeros((n, 11), dtype=np.float32)

    # Precompute rotation matrices for every frame.
    R_all = rotation_6d_to_rotation_matrix_numpy(poses_11d[:, 3:9])  # (N, 3, 3)
    pos_all = poses_11d[:, :3]                                       # (N, 3)

    # Per-frame local-frame deltas. Vectorised wouldn't help much here
    # since each row depends on the previous row's R.
    for i in range(n - 1):
        R_t = R_all[i]
        R_t1 = R_all[i + 1]
        delta_pos_world = pos_all[i + 1] - pos_all[i]
        actions[i, :3] = R_t.T @ delta_pos_world
        R_delta = R_t.T @ R_t1
        actions[i, 3:9] = rotation_matrix_to_rotation_6d_numpy(
            R_delta.reshape(1, 3, 3)
        )[0]

    # Zero out deltas at episode boundaries (action[last_frame_of_ep] = 0).
    ep_change = np.where(episode_indices[1:] != episode_indices[:-1])[0]
    actions[ep_change, :9] = 0.0
    actions[-1, :9] = 0.0

    # Gripper: absolute values (dims 9-10).
    actions[:, 9:] = poses_11d[:, 9:]

    return actions


def compute_relative_to_start_state(poses_11d: np.ndarray, episode_indices: np.ndarray) -> np.ndarray:
    """Compute pose relative to episode start, in the START camera frame.

    This is the proprioception counterpart of `compute_delta_actions`, and it
    MUST follow the exact same gripper-egocentric convention, otherwise the
    arbitrary per-session world frame leaks into the state.

    The recorded poses are camera-site SE(3) in a world frame whose horizontal
    orientation is arbitrary per session (gravity-aligned Z, but X/Y depend on
    the SLAM origin). A bare displacement `pos[t] - pos_start` is expressed in
    those world axes, so it rotates with the session — NOT frame-independent.
    Expressing it in the start camera frame makes the world rotation cancel:

        rel_pos[t] = R_start^T @ (pos[t] - pos_start)
        R_rel[t]   = R_start^T @ R[t]          # current rotation seen from start
        rel_r6d[t] = rotation_matrix_to_6d(R_rel[t])

    (Proof of invariance: under any world rotation R_w, pos->R_w·pos and
    R->R_w·R, so R_start^T·(pos[t]-pos_start) -> (R_w·R_start)^T·R_w·(...) =
    R_start^T·(...), and likewise for R_rel. The world frame drops out.)

    Args:
        poses_11d: (N, 11) absolute poses [x, y, z, r6d_0..5, proximal, distal]
        episode_indices: (N,) episode index per frame

    Returns:
        (N, 11) relative state [dx_start, dy_start, dz_start, r6d_rel_0..5, proximal, distal]
    """
    n = len(poses_11d)
    relative_state = np.zeros((n, 11), dtype=np.float32)

    # Find the start index of each episode
    unique_eps = np.unique(episode_indices)
    ep_start_idx = {}
    for ep in unique_eps:
        ep_start_idx[ep] = np.where(episode_indices == ep)[0][0]

    for i in range(n):
        ep = episode_indices[i]
        start_i = ep_start_idx[ep]

        r_current = rotation_6d_to_rotation_matrix_numpy(poses_11d[i, 3:9].reshape(1, 6))[0]
        r_start = rotation_6d_to_rotation_matrix_numpy(poses_11d[start_i, 3:9].reshape(1, 6))[0]

        # Pose relative to start, expressed in the START camera frame
        # (gripper-egocentric / frame-independent — see docstring).
        relative_state[i, :3] = r_start.T @ (poses_11d[i, :3] - poses_11d[start_i, :3])
        r_relative = r_start.T @ r_current
        relative_state[i, 3:9] = rotation_matrix_to_rotation_6d_numpy(r_relative.reshape(1, 3, 3))[0]

    # Gripper: absolute values
    relative_state[:, 9:] = poses_11d[:, 9:]

    return relative_state


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Grabette dataset for training")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="SteveNguyen/Grabette_redcube_quest",
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--proprioception",
        type=str,
        choices=["none", "relative"],
        default="none",
        help="State mode: 'none' = gripper only (2D), 'relative' = pose relative to episode start (11D)",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default=None,
        help="If set, copy the source dataset to a new local directory "
             "and convert THAT copy, leaving --repo_id untouched. The "
             "destination is a sibling of the source cache dir, keyed by "
             "this repo id (slashes replaced with '--'). Use this when "
             "you want to preserve the raw dataset (default conversion is "
             "in-place on the local HF cache).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Override the destination path for --output_repo_id. By "
             "default the copy lands next to the source under "
             "~/.cache/huggingface/lerobot/local-converted/<repo_id>/. "
             "Ignored if --output_repo_id is not set.",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="If the destination already exists, delete it before copying. "
             "Off by default to avoid accidental data loss.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="If set, push the converted dataset to this Hub repo id "
             "(e.g. 'SteveNguyen/sim_grasp_train_v2'). If the local repo "
             "id does not match, the push retargets to this id.",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make the Hub repo private (default: public).",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    use_relative = args.proprioception == "relative"
    state_names = STATE_NAMES_RELATIVE if use_relative else STATE_NAMES_NONE
    state_dim = len(state_names)

    logger.info(f"Proprioception mode: {args.proprioception} ({state_dim}D state)")

    # If --output_repo_id is set, copy the source dataset to a new local
    # directory and operate on the copy. Otherwise the conversion is in-place
    # on the HF cache (original behavior).
    if args.output_repo_id:
        src_ds = LeRobotDataset(args.repo_id)
        src_root = Path(src_ds.root)
        if args.output_root:
            dst_root = Path(args.output_root).expanduser().resolve()
        else:
            default_parent = Path.home() / ".cache/huggingface/lerobot/local-converted"
            dst_root = default_parent / args.output_repo_id.replace("/", "--")
        if dst_root.exists():
            if args.overwrite_output:
                logger.warning(f"Overwriting existing destination: {dst_root}")
                shutil.rmtree(dst_root)
            else:
                raise FileExistsError(
                    f"Destination already exists: {dst_root}. "
                    f"Pass --overwrite_output to delete it, or pick a different "
                    f"--output_repo_id / --output_root."
                )
        logger.info(f"Copying source dataset {src_root} → {dst_root} ...")
        dst_root.parent.mkdir(parents=True, exist_ok=True)
        # symlinks=False (default): dereference the HF cache symlinks so the
        # copy is a self-contained snapshot. With symlinks=True the destination
        # would inherit broken `../../blobs/<hash>` relative links pointing
        # back at the source cache layout — meta/info.json would appear to
        # exist but read as missing, and LeRobotDataset would then fall back
        # to a Hub lookup against a repo id that doesn't exist yet.
        shutil.copytree(src_root, dst_root)
        logger.info("Copy complete.")
        # Sanity-check that the copy is self-contained (catches the broken-
        # symlink failure mode early, with a clear error rather than a
        # confusing Hub-lookup 404 from LeRobotDataset's metadata fallback).
        for required in ("meta/info.json", "meta/stats.json"):
            p = dst_root / required
            if not p.exists():
                raise FileNotFoundError(
                    f"Copy is incomplete: {p} not found. "
                    f"This usually means the source cache uses HF symlinks "
                    f"and the copy was made with symlinks=True. Delete the "
                    f"destination ({dst_root}) and retry."
                )
        work_repo_id = args.output_repo_id
        work_root: Path | None = dst_root
        ds = LeRobotDataset(work_repo_id, root=work_root)
    else:
        ds = LeRobotDataset(args.repo_id)
        work_repo_id = args.repo_id
        work_root = None  # let LeRobotDataset resolve from HF cache by repo_id
    root = Path(ds.root)
    logger.info(f"Dataset root: {root}")
    logger.info(f"Frames: {len(ds)}, Episodes: {ds.meta.total_episodes}")

    # --- 1. Convert parquet data ---
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    logger.info(f"Processing {len(parquet_files)} parquet file(s)...")

    for pf in parquet_files:
        table = pq.read_table(pf)

        # Read the original action column (= absolute pose at each frame)
        action_col = table.column("action")
        poses_raw = np.array(action_col.to_pylist(), dtype=np.float32)

        # Convert to 11D if still 8D
        if poses_raw.shape[1] == 8:
            poses_11d = pose_8d_to_11d(poses_raw)
            logger.info(f"  {pf.name}: converted rotation 8D -> 11D")
        elif poses_raw.shape[1] == 11:
            # Check if this is already deltas or still absolute
            pos_mean = np.abs(poses_raw[:, :3].mean(axis=0))
            if np.all(pos_mean < 0.01):
                logger.warning(f"  {pf.name}: appears to already be deltas, re-run on original data")
                continue
            poses_11d = poses_raw
        else:
            raise ValueError(f"Unexpected action dim: {poses_raw.shape[1]}")

        # Episode indices
        episode_indices = np.array(table.column("episode_index").to_pylist())

        # Compute delta actions
        delta_actions = compute_delta_actions(poses_11d, episode_indices)
        logger.info(
            f"  {pf.name}: delta actions "
            f"(mean pos delta: {np.linalg.norm(delta_actions[:, :3], axis=1).mean() * 1000:.2f} mm)"
        )

        # Compute observation state
        if use_relative:
            obs_state = compute_relative_to_start_state(poses_11d, episode_indices)
            logger.info(f"  {pf.name}: computed relative-to-start state (11D)")
        else:
            obs_state = poses_11d[:, 9:]  # gripper only (2D)

        # Rebuild table
        df_dict = {}
        for col in table.column_names:
            if col == "action":
                df_dict[col] = pa.array(delta_actions.tolist(), type=pa.list_(pa.float32()))
            elif col == "observation.state":
                df_dict[col] = pa.array(obs_state.tolist(), type=pa.list_(pa.float32()))
            else:
                df_dict[col] = table.column(col)

        if "observation.state" not in table.column_names:
            df_dict["observation.state"] = pa.array(obs_state.tolist(), type=pa.list_(pa.float32()))

        new_table = pa.table(df_dict)
        pq.write_table(new_table, pf)
        logger.info(f"  {pf.name}: written ({new_table.num_rows} rows)")

    # --- 2. Update info.json ---
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["features"]["action"] = {"dtype": "float32", "shape": [11], "names": ACTION_NAMES}
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [state_dim],
        "names": state_names,
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(
        f"Updated info.json: action=11D (deltas), observation.state={state_dim}D ({args.proprioception})"
    )

    # --- 3. Recompute stats ---
    logger.info("Recomputing stats...")
    from lerobot.datasets.dataset_tools import recompute_stats

    ds_updated = LeRobotDataset(work_repo_id, root=work_root)
    recompute_stats(ds_updated, skip_image_video=True)

    # --- 4. Verify ---
    logger.info("\n=== Verification ===")
    ds_final = LeRobotDataset(work_repo_id, root=work_root, episodes=[0])

    logger.info(
        f"observation.state: shape={ds_final.meta.features['observation.state']['shape']}, "
        f"names={ds_final.meta.features['observation.state']['names']}"
    )
    logger.info(
        f"action: shape={ds_final.meta.features['action']['shape']}, "
        f"names={ds_final.meta.features['action']['names']}"
    )

    sample = ds_final[50]
    state = sample["observation.state"].tolist()
    logger.info("\nSample frame 50:")
    logger.info(f"  observation.state ({state_dim}D):")
    for n, v in zip(state_names, state, strict=True):
        logger.info(f"    {n:12s}: {v:+.6f}")

    action = sample["action"].tolist()
    logger.info("  action (11D):")
    for n, v in zip(ACTION_NAMES, action, strict=True):
        logger.info(f"    {n:8s}: {v:+.6f}")

    if use_relative:
        # Frame 0 of episode should have zero relative pose
        sample0 = ds_final[0]
        state0 = sample0["observation.state"].tolist()
        logger.info("\n  Frame 0 state (should be ~0 for pose dims, nonzero for gripper):")
        for n, v in zip(state_names, state0, strict=True):
            logger.info(f"    {n:12s}: {v:+.6f}")

    pos_delta = np.array(action[:3])
    logger.info(f"\n  Position delta magnitude: {np.linalg.norm(pos_delta) * 1000:.2f} mm")

    logger.info("\nConversion complete!")

    if args.push_to_hub:
        target = args.push_to_hub
        logger.info(f"\nPushing to Hub repo: {target} (private={args.hub_private})")
        ds_push = LeRobotDataset(work_repo_id, root=work_root)
        if target != work_repo_id:
            # Retarget — same approach as the standalone push helper.
            ds_push.repo_id = target
            ds_push.meta.repo_id = target
        ds_push.push_to_hub(private=args.hub_private, push_videos=True)
        logger.info(f"Pushed: {target}")

    if args.output_repo_id:
        logger.info(
            f"\nConverted copy lives at {root}. "
            f"Load it with: LeRobotDataset('{work_repo_id}', root='{root}')."
        )


if __name__ == "__main__":
    main()
