#!/usr/bin/env python3
"""CLI wrapper around grabette_postprocess.convert.convert_episode.

Converts grabette episode directories into the oak/ layout consumed by
run_oak_slam.py / docker/oak_vslam. See grabette_postprocess/convert.py for
the conversion details.

Usage:
    python scripts/pipeline/convert_episode_to_oak.py -i /path/to/episode_dir
    # then:
    python scripts/pipeline/run_oak_slam.py -i /path/to/episode_dir
"""

import sys
from pathlib import Path

import click

from grabette_postprocess.convert import convert_episode


@click.command()
@click.option("-i", "--input_dir", required=True, multiple=True,
              type=click.Path(exists=True),
              help="Episode directory (repeatable for batch)")
@click.option("--force", is_flag=True, help="Overwrite existing oak/ subdir")
def main(input_dir, force):
    """Convert episode directories to the oak/ layout for SLAM."""
    for d in input_dir:
        ep_dir = Path(d).expanduser().absolute()
        print(f"Converting {ep_dir.name}...")
        try:
            convert_episode(ep_dir, force=force)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
