"""LeRobot v3 dataset builder for GRABETTE.

Converts trajectory + capture data into LeRobot v3 format (Parquet + MP4).
Supports two camera sources:
  - RPi fisheye camera (raw_video.mp4)
  - Quest POV camera (grpc_camera_frames/*.jpg), optional
"""

import json
from pathlib import Path

import av
import cv2
import numpy as np

from grabette_postprocess.trajectory import (
    load_trajectory_csv,
    trajectory_to_poses,
    interpolate_angles,
)

# Feature schema for the GRABETTE dataset
# Action is 8D: [x, y, z, ax, ay, az, proximal, distal]
# action[t] = state[t+1] (next-step target)
FEATURES_BASE = {
    "observation.images.cam0": {
        "dtype": "video",
        "shape": (3, 720, 960),  # C, H, W — LeRobot convention
        "names": ["channels", "height", "width"],
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["x", "y", "z", "ax", "ay", "az", "proximal", "distal"],
    },
}


def _load_video_frames_indexed(video_path: Path, size: tuple[int, int],
                               needed_indices: set[int]) -> dict[int, np.ndarray]:
    """Load only the video frames at the given indices.

    Returns dict mapping frame_index -> (H, W, 3) uint8 BGR array.
    """
    h, w = size
    cache = {}
    max_idx = max(needed_indices)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in needed_indices:
                img = frame.to_ndarray(format="bgr24")
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h))
                cache[i] = img
            if i >= max_idx:
                break
    return cache


def _load_video_timestamps(episode_dir: Path, video_path: Path) -> np.ndarray:
    """Return per-frame timestamps in seconds for the RPi video.

    Uses frame_timestamps.json if present (ms, relative to recording start).
    Falls back to uniform timestamps derived from the video's declared fps.
    """
    ft_path = episode_dir / "frame_timestamps.json"
    if ft_path.is_file():
        with open(ft_path) as f:
            ts_ms = json.load(f)
        return np.array(ts_ms, dtype=np.float64) / 1000.0

    # Fallback: uniform timestamps
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return np.arange(n_frames, dtype=np.float64) / video_fps


def _load_quest_frames(episode_dir: Path) -> list[tuple[float, Path]] | None:
    """Load Quest camera frame paths with relative timestamps in seconds.

    Timestamps are parsed from filenames (absolute ms), then zero-based so
    the first frame is at t=0s, matching the trajectory clock.

    Returns list of (timestamp_s, path) sorted by timestamp, or None.
    """
    quest_dir = episode_dir / "grpc_camera_frames"
    if not quest_dir.is_dir():
        return None

    frames = []
    for f in sorted(quest_dir.iterdir()):
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        # Filename format: frame_{timestamp_ms}_{seqnum}.ext
        parts = f.stem.split('_')
        if len(parts) >= 2:
            ts_ms = int(parts[1])
            frames.append((ts_ms, f))

    if not frames:
        return None

    # Convert absolute ms timestamps to relative seconds
    t0 = frames[0][0]
    return [(( ts - t0) / 1000.0, path) for ts, path in frames]


def build_dataset(
    repo_id: str,
    episode_dirs: list[Path],
    task: str,
    fps: float | None = None,
    image_size: tuple[int, int] = (720, 960),
    quest_image_size: tuple[int, int] | None = None,
    root: Path | None = None,
    include_quest_camera: bool = False,
):
    """Build LeRobot v3 dataset from processed episode directories.

    Each episode directory must contain:
        - raw_video.mp4
        - imu_data.json (raw, with ANGL stream)
        - camera_trajectory.csv (or mapping_camera_trajectory.csv)

    Optionally:
        - grpc_camera_frames/ (Quest POV images, added as observation.images.cam1)

    Args:
        repo_id: dataset identifier (e.g. "steve/grabette-demo")
        episode_dirs: list of episode directory paths
        task: task description string
        fps: dataset frame rate. If None, auto-detected from trajectory source:
            50fps for SLAM (native RPi camera rate),
            30fps for Quest.
        image_size: (height, width) for RPi camera output frames
        quest_image_size: (height, width) for Quest camera output frames
            (default: same as image_size)
        root: local storage path (default: HF cache)
        include_quest_camera: include Quest POV camera as observation.images.cam1
    """
    # Lazy import — lerobot is a heavy dependency
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if quest_image_size is None:
        quest_image_size = image_size

    # Auto-detect fps from first episode's trajectory source if not specified
    if fps is None:
        first_meta = episode_dirs[0] / "slam_metadata.json"
        if first_meta.is_file():
            with open(first_meta) as f:
                m = json.load(f)
            method = m.get("method", "slam")
            fps = 30.0 if method == "quest" else 50.0
        else:
            fps = 50.0
        print(f"Auto-detected fps: {int(fps)}")

    # Build feature schema
    features = FEATURES_BASE.copy()
    h, w = image_size
    features["observation.images.cam0"] = {
        **features["observation.images.cam0"],
        "shape": (3, h, w),
    }
    if include_quest_camera:
        qh, qw = quest_image_size
        features["observation.images.cam1"] = {
            "dtype": "video",
            "shape": (3, qh, qw),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        root=root,
        robot_type="grabette",
        use_videos=True,
        vcodec="h264",
    )

    for ep_dir in episode_dirs:
        ep_dir = Path(ep_dir).absolute()
        print(f"\nProcessing {ep_dir.name}...")

        # Find trajectory file
        traj_path = ep_dir / "camera_trajectory.csv"
        if not traj_path.is_file():
            traj_path = ep_dir / "mapping_camera_trajectory.csv"
        if not traj_path.is_file():
            print(f"  Skipping: no trajectory CSV found")
            continue

        # Load trajectory and convert to 6D poses
        df = load_trajectory_csv(traj_path)
        poses = trajectory_to_poses(df)
        # Trajectory timestamps in seconds, relative to recording start (t=0)
        traj_ts = df['timestamp'].values.astype(np.float64)
        n_frames = len(df)

        # Load joint angles from raw IMU (ANGL stream)
        imu_path = ep_dir / "imu_data.json"
        if imu_path.is_file():
            joints = interpolate_angles(imu_path, traj_ts)
        else:
            print(f"  Warning: no imu_data.json, joints will be zeros")
            joints = np.zeros((n_frames, 2), dtype=np.float32)

        # Build state: [x, y, z, ax, ay, az, proximal, distal]
        state = np.concatenate([poses, joints], axis=1).astype(np.float32)

        # Action[t] = absolute state at frame t.
        # The pi0 training pipeline converts to relative actions via
        # use_relative_actions=true and derives proprioception state from
        # the action column via derive_state_from_action=true.
        actions = state

        # --- cam0: RPi video, timestamp-based frame selection ---
        video_path = ep_dir / "raw_video.mp4"
        # Per-frame timestamps in seconds, relative to recording start
        video_ts = _load_video_timestamps(ep_dir, video_path)
        print(f"  RPi video: {len(video_ts)} frames, {video_ts[-1]:.2f}s")

        # For each trajectory step, find the nearest video frame index
        cam0_indices = np.array([
            int(np.argmin(np.abs(video_ts - t))) for t in traj_ts
        ])
        cam0_cache = _load_video_frames_indexed(video_path, image_size,
                                                set(cam0_indices.tolist()))

        # --- cam1: Quest camera, timestamp-based frame selection ---
        qh, qw = quest_image_size
        quest_frames = None
        cam1_paths = None
        if include_quest_camera:
            quest_frames = _load_quest_frames(ep_dir)
            if quest_frames:
                # Quest frames already have relative timestamps in seconds
                quest_ts = np.array([f[0] for f in quest_frames])
                print(f"  Quest camera: {len(quest_frames)} frames, {quest_ts[-1]:.2f}s")
                # For each trajectory step, find nearest Quest frame
                cam1_paths = [
                    quest_frames[int(np.argmin(np.abs(quest_ts - t)))][1]
                    for t in traj_ts
                ]
            else:
                print(f"  Warning: no Quest camera frames found")

        for i in range(n_frames):
            img = cam0_cache.get(cam0_indices[i])
            if img is None:
                print(f"  Warning: missing RPi frame at step {i}, t={traj_ts[i]:.3f}s")
                break
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frame_data = {
                "task": task,
                "observation.images.cam0": img_rgb,
                "action": actions[i],
            }

            if include_quest_camera:
                if cam1_paths is not None:
                    quest_img_bgr = cv2.imread(str(cam1_paths[i]))
                    if quest_img_bgr is not None:
                        if quest_img_bgr.shape[0] != qh or quest_img_bgr.shape[1] != qw:
                            quest_img_bgr = cv2.resize(quest_img_bgr, (qw, qh))
                        quest_img = cv2.cvtColor(quest_img_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        quest_img = np.zeros((qh, qw, 3), dtype=np.uint8)
                else:
                    quest_img = np.zeros((qh, qw, 3), dtype=np.uint8)
                frame_data["observation.images.cam1"] = quest_img

            dataset.add_frame(frame_data)

        dataset.save_episode()
        print(f"  Saved episode: {n_frames} frames")

    dataset.finalize()
    print(f"\nDataset complete: {repo_id}")
    if root:
        print(f"  Location: {root}")

    return dataset
