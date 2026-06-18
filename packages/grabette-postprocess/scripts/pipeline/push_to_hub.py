#!/usr/bin/env python3
"""Push a local LeRobot dataset to Hugging Face Hub.

Requires: huggingface-cli login (or HF_TOKEN env var).
"""

import click
from pathlib import Path

from grabette_postprocess.dataset import push_dataset


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
    push_dataset(repo_id, Path(root).expanduser().absolute(),
                 private=private, tags=tuple(tags))


if __name__ == "__main__":
    main()
