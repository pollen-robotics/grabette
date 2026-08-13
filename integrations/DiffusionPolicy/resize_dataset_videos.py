"""Re-encode a LeRobot dataset's videos at a lower resolution (training copy).

Why: the policy's encoder resizes every frame to 236x236 internally, yet the
datasets store 960x720 — every training sample decodes ~12x more pixels than
the network uses. Decode CPU, worker RAM, shm traffic and download size all
scale with those wasted pixels (this is what made trainings dataloader-bound).
Storing at ~2x the network input (480x360) keeps a comfortable margin for the
crop while cutting decode work ~4x.

This produces a NEW dataset copy (non-destructive); the raw data always stays
on the Hub, so any resolution decision is reversible by rebuilding. Actions,
states and episode metadata are untouched — only the video files and their
declared shapes change. Image normalization stats are per-channel and
resolution-independent, so they are kept as-is.

Deployment note: the robot feeds full-resolution live frames; training frames
now come from a downscaled store. Both meet at the encoder's internal
236x236 resize, where the difference reduces to negligible resampling
character. State it, don't fear it.

Usage:
  uv run python resize_dataset_videos.py \\
      --repo_id <user>/<dataset>_cartesian \\
      --output_root <dir> [--width 480 --height 360] \\
      [--push_to_hub <user>/<dataset>_cartesian_480]

Then decode-check the result (check_dataset_videos.py) before training on it.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from lerobot.datasets import LeRobotDataset


def resize_videos(root: Path, width: int, height: int, crf: int) -> int:
    n = 0
    for mp4 in sorted((root / "videos").rglob("*.mp4")):
        tmp = mp4.with_suffix(".resized.mp4")
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(mp4),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            # ALL-INTRA (-g 1), matching lerobot's own encoding: training does
            # RANDOM access, and with normal GOPs (x264 default: 250) every
            # 2-frame sample decodes the whole GOP from the last keyframe —
            # measured to make training SLOWER despite 4x smaller frames.
            "-g", "1",
            "-pix_fmt", "yuv420p", "-an", str(tmp),
        ]
        subprocess.run(cmd, check=True)
        tmp.replace(mp4)
        n += 1
        print(f"  resized {mp4.relative_to(root)}")
    return n


def patch_info(root: Path, width: int, height: int) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key, ft in info["features"].items():
        if ft.get("dtype") == "video":
            ft["shape"] = [3, height, width]
            vi = ft.get("info", {})
            vi["video.height"] = height
            vi["video.width"] = width
            vi["video.codec"] = "h264"
            vi["video.pix_fmt"] = "yuv420p"
    info_path.write_text(json.dumps(info, indent=4))
    print(f"  patched info.json video shapes -> (3, {height}, {width})")


def main():
    p = argparse.ArgumentParser(description="Downscale a LeRobot dataset's videos (new copy)")
    p.add_argument("--repo_id", required=True)
    p.add_argument("--root", default=None, help="Local dataset root (else HF cache/Hub)")
    p.add_argument("--output_root", required=True, help="Where the resized copy is written")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better/larger)")
    p.add_argument("--push_to_hub", default=None, help="Push the resized copy to this Hub repo id")
    p.add_argument("--hub_private", action="store_true")
    args = p.parse_args()

    src = LeRobotDataset(args.repo_id, root=args.root)  # ensures full local snapshot
    dst = Path(args.output_root).expanduser().resolve()
    if dst.exists():
        raise SystemExit(f"{dst} already exists — pick a fresh --output_root.")
    print(f"Copying {src.root} -> {dst}")
    shutil.copytree(src.root, dst)

    n = resize_videos(dst, args.width, args.height, args.crf)
    patch_info(dst, args.width, args.height)
    print(f"Done: {n} video file(s) at {args.width}x{args.height}. "
          f"Decode-check with check_dataset_videos.py before training.")

    if args.push_to_hub:
        ds = LeRobotDataset(args.push_to_hub, root=dst)
        ds.push_to_hub(private=args.hub_private, push_videos=True)
        print(f"Pushed: {args.push_to_hub}")


if __name__ == "__main__":
    main()
