#!/usr/bin/env python3
"""Push a local LeRobot dataset to Hugging Face Hub.

Requires: huggingface-cli login (or HF_TOKEN env var).
"""

import click
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@click.command()
@click.option("--repo_id", required=True,
              help="HF dataset repo (e.g. 'pollenrobotics/grabette-demo')")
@click.option("--root", required=True, type=click.Path(exists=True),
              help="Local dataset root (same --root used in generate_dataset.py)")
@click.option("--private", is_flag=True, default=False,
              help="Create a private dataset on the Hub")
@click.option("--tags", multiple=True, default=["lerobot", "grabette"],
              help="Dataset tags (repeatable)")
def main(repo_id, root, private, tags):
    root = Path(root).expanduser().absolute()

    print(f"Loading dataset {repo_id} from {root}...")
    ds = LeRobotDataset(repo_id, root=root)
    print(f"  Episodes: {ds.num_episodes}")
    print(f"  Frames:   {ds.num_frames}")

    print(f"\nPushing to https://huggingface.co/datasets/{repo_id} ...")
    ds.push_to_hub(
        tags=list(tags),
        private=private,
    )
    print(f"\nDone: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
