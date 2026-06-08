#!/usr/bin/env python3
"""Generate a LeRobot v3 dataset from processed episode directories."""

import click
from pathlib import Path

from grabette_postprocess.dataset import build_dataset


@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True),
              help="Parent directory containing episode subdirectories")
@click.option("--repo_id", required=True,
              help="Dataset identifier (e.g. 'steve/grabette-demo')")
@click.option("--task", required=True,
              help="Task description (e.g. 'cup manipulation')")
@click.option("--fps", type=float, default=None,
              help="Video frame rate (default: auto-detect from trajectory source)")
@click.option("--image_height", type=int, default=720)
@click.option("--image_width", type=int, default=960)
@click.option("--quest-camera", is_flag=True, default=False,
              help="Include Quest POV camera as observation.images.cam1")
@click.option("--quest_image_height", type=int, default=480)
@click.option("--quest_image_width", type=int, default=640)
@click.option("--root", type=click.Path(), default=None,
              help="Local storage path (default: HF cache)")
def main(input_dir, repo_id, task, fps, image_height, image_width,
         quest_camera, quest_image_height, quest_image_width, root):
    input_dir = Path(input_dir).expanduser().absolute()

    # Find all episode directories that have a trajectory CSV
    episode_dirs = sorted([
        p.parent for p in input_dir.glob("*/raw_video.mp4")
        if (p.parent / "camera_trajectory.csv").is_file()
        or (p.parent / "mapping_camera_trajectory.csv").is_file()
    ])
    print(f"Found {len(episode_dirs)} episodes with trajectories")

    if not episode_dirs:
        raise click.ClickException(f"No processed episodes found under {input_dir}")

    build_dataset(
        repo_id=repo_id,
        episode_dirs=episode_dirs,
        task=task,
        fps=fps,
        image_size=(image_height, image_width),
        quest_image_size=(quest_image_height, quest_image_width),
        root=Path(root) if root else None,
        include_quest_camera=quest_camera,
    )


if __name__ == "__main__":
    main()
