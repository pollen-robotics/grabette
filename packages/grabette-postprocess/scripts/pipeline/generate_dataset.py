#!/usr/bin/env python3
"""Generate a LeRobot v3 dataset from processed episode directories."""

import click
from pathlib import Path

from grabette_postprocess.dataset import build_dataset
from grabette_postprocess.episode_manager import find_processed_episodes


@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True),
              help="Parent directory containing episode subdirectories")
@click.option("--repo_id", required=True,
              help="Dataset identifier (e.g. '<user>/<dataset>')")
@click.option("--task", required=True,
              help="Task description (e.g. 'cup manipulation')")
@click.option("--fps", type=float, default=None,
              help="Video frame rate (default: 50fps, native RPi camera rate)")
@click.option("--image_height", type=int, default=720)
@click.option("--image_width", type=int, default=960)
@click.option("--root", type=click.Path(), default=None,
              help="Local storage path (default: HF cache)")
def main(input_dir, repo_id, task, fps, image_height, image_width, root):
    input_dir = Path(input_dir).expanduser().absolute()

    # Episode dirs that already carry a trajectory CSV (SLAM has run) and the
    # Arducam video build_dataset needs.
    episode_dirs = [ep for ep in find_processed_episodes(input_dir)
                    if (ep / "raw_video.mp4").is_file()]
    print(f"Found {len(episode_dirs)} episodes with trajectories")

    if not episode_dirs:
        raise click.ClickException(f"No processed episodes found under {input_dir}")

    build_dataset(
        repo_id=repo_id,
        episode_dirs=episode_dirs,
        task=task,
        fps=fps,
        image_size=(image_height, image_width),
        root=Path(root) if root else None,
    )


if __name__ == "__main__":
    main()
