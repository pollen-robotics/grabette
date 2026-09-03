"""Bring a raw capture into the shape our pipeline and eval loop expect.

Newer, bimanual-capable captures differ from what the pipeline assumes in three
ways, none of them handled downstream:

  - channel names carry the arm side (`right_x`, `right_proximal`);
  - cameras are named `right_camN` rather than `cam0`, and `evaluate.py` sends
    `observation.images.cam0`, so a checkpoint trained on the prefixed key
    cannot be deployed;
  - there is no `observation.state`. `convert_dataset.py` normally synthesises
    it, but a chunk-relative dataset skips that script entirely — the delta
    conversion is exactly what chunk-relative replaces — so the state has to be
    added here instead.

Written because the pick3 chunk-relative dataset was assembled with throwaway
scripts (recorded in docs/relative_actions_lerobot_native.md but not committed),
which made the build unreproducible. Everything here is non-destructive: the
source is never modified.

The camera rename is deliberately a surgical metadata edit rather than
lerobot's add+remove of a video feature, which would re-encode every frame to
change a name.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import click
import numpy as np
import pandas as pd

from grabette_postprocess.grasp_projection_convert import _gripper_columns

logger = logging.getLogger(__name__)

STATE_NAMES = ["proximal", "distal"]


def rename_video_feature(root: Path, old_key: str, new_key: str) -> None:
    """Rename a video feature everywhere it is referenced.

    Four places, and missing any one of them leaves a dataset that either fails
    to load or silently loses its video: info.json features, the videos/
    directory, the per-episode columns in meta/episodes/*.parquet
    (`videos/<key>/...` and `stats/<key>/...`), and meta/stats.json.
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if old_key not in info["features"]:
        raise SystemExit(
            f"{old_key!r} is not a feature of this dataset.\n"
            f"features: {sorted(info['features'])}"
        )
    if new_key in info["features"]:
        raise SystemExit(f"{new_key!r} already exists; refusing to overwrite it.")

    # info.json — preserve key order so the diff stays readable
    info["features"] = {
        (new_key if k == old_key else k): v for k, v in info["features"].items()
    }
    info_path.write_text(json.dumps(info, indent=4))

    old_dir, new_dir = root / "videos" / old_key, root / "videos" / new_key
    if old_dir.is_dir():
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        old_dir.rename(new_dir)
        logger.info("  videos/%s -> videos/%s", old_key, new_key)
    else:
        logger.warning("  no videos/%s directory to rename", old_key)

    for pf in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(pf)
        mapping = {c: c.replace(old_key, new_key) for c in df.columns if old_key in c}
        if mapping:
            df = df.rename(columns=mapping)
            df.to_parquet(pf, index=False)
            logger.info("  %s: renamed %d column(s)", pf.name, len(mapping))

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        if old_key in stats:
            stats = {(new_key if k == old_key else k): v for k, v in stats.items()}
            stats_path.write_text(json.dumps(stats, indent=4))
            logger.info("  stats.json: renamed %s", old_key)


def strip_name_prefix(root: Path, prefix: str, features: tuple[str, ...]) -> None:
    """Drop `prefix` from the channel names of the given features (info.json only).

    Channel names live nowhere else — `action` is one list column — so this is a
    pure metadata edit. It matters because our tooling locates channels BY NAME
    (`_gripper_columns`, the publish checks, the training guards).
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for feat in features:
        names = (info["features"].get(feat) or {}).get("names")
        if not names:
            continue
        stripped = [n[len(prefix):] if str(n).startswith(prefix) else n for n in names]
        if stripped != names:
            info["features"][feat]["names"] = stripped
            logger.info("  %s: %s -> %s", feat, names, stripped)
    info_path.write_text(json.dumps(info, indent=4))


def add_gripper_state(root: Path, repo_id: str, output_root: Path) -> Path:
    """Add `observation.state` = the action's (proximal, distal) pair.

    Follows convert_dataset.py's convention for the 2D gripper-only
    proprioception mode: the policy sees where the fingers ARE, and no absolute
    SLAM pose enters the observation.
    """
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import modify_features

    root, output_root = Path(root), Path(output_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    if "observation.state" in info["features"]:
        raise SystemExit("observation.state already exists; nothing to add.")

    names = info["features"]["action"].get("names")
    dim = int(info["features"]["action"]["shape"][0])
    ip, idl = _gripper_columns(names, dim)
    logger.info("  gripper columns in action: %s -> observation.state", (ip, idl))

    # Derive the state from each ROW's own action rather than from a
    # pre-assembled array. modify_features passes the row dict to a callable, so
    # the two cannot be misaligned by construction — with an array it aligns by
    # position across files, which a differently-ordered read would silently
    # shift. (The array path also rejects multi-dimensional features outright:
    # it assigns straight into a DataFrame column.)
    def state_of_row(row, _ep_idx, _frame_in_ep):
        a = np.asarray(row["action"], dtype=np.float32)
        return np.array([a[ip], a[idl]], dtype=np.float32)

    sample = pd.read_parquet(
        sorted((root / "data").rglob("*.parquet"))[0], columns=["action"])
    arr = np.stack(sample["action"].to_numpy()).astype(np.float32)
    logger.info("  state range prox [%.3f, %.3f] dist [%.3f, %.3f] (first file)",
                arr[:, ip].min(), arr[:, ip].max(),
                arr[:, idl].min(), arr[:, idl].max())

    ds = LeRobotDataset(repo_id, root=root)
    out = modify_features(
        ds,
        add_features={"observation.state": (
            state_of_row, {"dtype": "float32", "shape": [2], "names": STATE_NAMES})},
        output_dir=output_root,
        repo_id=repo_id,
    )
    return Path(out.root)


@click.command()
@click.option("--src_root", required=True, type=click.Path(exists=True),
              help="Source dataset root (cleaned + resized). Never modified.")
@click.option("--dst_root", required=True, type=click.Path(),
              help="Destination root.")
@click.option("--repo_id", required=True,
              help="Local id for the result, e.g. local/sugar_cup_raw8d.")
@click.option("--camera", default=None,
              help="Video feature to rename to observation.images.cam0, short "
                   "('right_cam0') or full. Skipped if already cam0.")
@click.option("--strip_prefix", default=None,
              help="Prefix to drop from channel names, e.g. 'right_'.")
@click.option("--task", default=None,
              help="Replace the task string for every episode. A VLA is "
                   "conditioned on this verbatim, so it must be an instruction.")
@click.option("--overwrite", is_flag=True)
def main(src_root, dst_root, repo_id, camera, strip_prefix, task, overwrite):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src_root, dst_root = Path(src_root), Path(dst_root)
    if dst_root.exists():
        if not overwrite:
            raise SystemExit(f"{dst_root} exists; pass --overwrite.")
        shutil.rmtree(dst_root)

    # observation.state first: modify_features writes the new dataset, so the
    # surgical edits below then apply to the copy and the source stays clean.
    logger.info("Adding observation.state")
    out = add_gripper_state(src_root, repo_id, dst_root)

    if camera:
        info = json.loads((out / "meta" / "info.json").read_text())
        vids = [k for k in info["features"] if k.startswith("observation.images.")]
        match = [k for k in vids if k == camera or k.endswith("." + camera)]
        if not match:
            raise SystemExit(f"--camera {camera!r} matches none of {vids}")
        if match[0] != "observation.images.cam0":
            logger.info("Renaming %s -> observation.images.cam0", match[0])
            rename_video_feature(out, match[0], "observation.images.cam0")

    if strip_prefix:
        logger.info("Stripping channel-name prefix %r", strip_prefix)
        strip_name_prefix(out, strip_prefix, ("action", "observation.state"))

    if task:
        from lerobot.datasets import LeRobotDataset
        from lerobot.datasets.dataset_tools import modify_tasks

        logger.info("Relabelling task -> %r", task)
        modify_tasks(LeRobotDataset(repo_id, root=out), new_task=task)

    from grabette_postprocess.checks.publish import check_publish

    audit = check_publish(out)
    for kind in ("errors", "warnings"):
        for m in audit[kind]:
            logger.warning("publish check (%s): %s", kind[:-1], m)
    logger.info("Normalized dataset ready: %s", out)


if __name__ == "__main__":
    main()
