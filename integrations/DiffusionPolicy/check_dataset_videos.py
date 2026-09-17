"""Scan every episode's video segments for decode errors.

Decodes each episode's frames through the SAME path the training dataloader
uses (lerobot decode_video_frames → torchcodec by default), so any episode
that would crash training with e.g. "Could not push packet to decoder:
Invalid data found when processing input" is caught here, with its episode
index and file path, instead of at a random training step.

Typical cause: a video file truncated during the pipeline's re-encode (watch
out for /tmp being a size-capped tmpfs) or a corrupted upload. Fix by
re-running the pipeline step that produced the file, or drop the affected
episodes (--exclude_episodes in train.py takes the printed list).

Usage:
  uv run python check_dataset_videos.py \
      --repo_id local/test_pick_can_200_cartesian \
      --dataset_root /tmp/grabette_pipeline/test_pick_can_200/cartesian
"""

import argparse

from lerobot.datasets import LeRobotDatasetMetadata, load_episodes
from lerobot.datasets.video_utils import decode_video_frames
from tqdm import tqdm


def scan_segment(video_path, from_ts, to_ts, fps, tolerance_s, backend,
                 chunk: int = 64) -> str | None:
    """Decode every frame of one episode's segment. Returns the error string,
    or None if the whole segment decodes.

    Decodes in chunks of `chunk` frames: a full episode in one call peaks at
    ~2.5 GB of frame tensors (300+ frames x 3x720x960 float32); chunking bounds
    the peak to a few hundred MB without changing what is exercised."""
    n_frames = round((to_ts - from_ts) * fps)
    if n_frames <= 0:
        return f"empty segment ({from_ts:.3f}..{to_ts:.3f}s)"
    # File-relative timestamps, same shift the dataloader applies.
    timestamps = [from_ts + i / fps for i in range(n_frames)]
    for i in range(0, n_frames, chunk):
        try:
            decode_video_frames(video_path, timestamps[i:i + chunk], tolerance_s, backend)
        except Exception as e:  # noqa: BLE001 — report every decode failure, whatever its type
            t0 = timestamps[i]
            return f"{type(e).__name__} near {t0:.2f}s (frames {i}..{min(i + chunk, n_frames) - 1}): {e}"
    return None


def main():
    p = argparse.ArgumentParser(description="Decode-check all video segments of a LeRobot dataset")
    p.add_argument("--repo_id", required=True)
    p.add_argument("--dataset_root", default=None, help="Local dataset root (same as train.py)")
    p.add_argument("--backend", default="torchcodec", choices=["torchcodec", "pyav"],
                   help="Video backend; torchcodec matches training's default")
    args = p.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    if meta.episodes is None:
        meta.episodes = load_episodes(meta.root)
    # Same tolerance the dataset reader uses (frame period minus epsilon).
    tolerance_s = 1.0 / meta.fps - 1e-4

    bad: list[tuple[int, str, str]] = []
    for ep_idx in tqdm(range(meta.total_episodes), desc="episodes"):
        ep = meta.episodes[ep_idx]
        for vk in meta.video_keys:
            video_path = meta.root / meta.video_path.format(
                video_key=vk,
                chunk_index=ep[f"videos/{vk}/chunk_index"],
                file_index=ep[f"videos/{vk}/file_index"],
            )
            if not video_path.exists():
                bad.append((ep_idx, vk, f"missing file {video_path}"))
                continue
            err = scan_segment(
                video_path,
                ep[f"videos/{vk}/from_timestamp"],
                ep[f"videos/{vk}/to_timestamp"],
                meta.fps,
                tolerance_s,
                args.backend,
            )
            if err is not None:
                bad.append((ep_idx, vk, f"{video_path.name}: {err}"))

    print(f"\nchecked {meta.total_episodes} episodes x {len(meta.video_keys)} camera(s)")
    if not bad:
        print("all video segments decode cleanly — the dataset is safe to train on")
        return
    bad_eps = sorted({e for e, _, _ in bad})
    print(f"{len(bad)} FAILING segment(s) in {len(bad_eps)} episode(s):")
    for ep_idx, vk, err in bad:
        print(f"  ep {ep_idx:4d}  {vk}: {err}")
    print(f"\nepisodes to drop / re-encode: {bad_eps}")
    print("(train.py --exclude_episodes accepts this list directly)")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
