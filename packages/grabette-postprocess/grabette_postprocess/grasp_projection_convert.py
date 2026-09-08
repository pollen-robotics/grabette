"""Re-express a dataset's gripper channels as (strategy, commanded closure).

WHAT CHANGES
------------
The two gripper channels are replaced, in place of the raw angles:

    action[-2:]            -> (s, c_target)
    observation.state[-2:] -> (s_obs, c_obs)

`c_target` is what the policy should COMMAND:

    close      -> 1.0        drive fully closed along `s`; the OBJECT stops the
                             fingers, at the servo's torque cap
    otherwise  -> passthrough of the closure the human actually used

So the ONLY thing this conversion changes is that the grasp becomes a full close.
Everything else is the demonstration unaltered, which keeps the distribution shift
as small as it can be.

An earlier version commanded `c = 0` (fully open) on frames labelled "open". That
was wrong: humans open only about as wide as the grasp needs, and most detected
opens in the existing data are start-of-episode pre-shape adjustments, not
releases. Commanding a full open there changed the approach posture by up to 76
degrees and would have put the observed gripper out of distribution. Detecting
opens still matters — it is what stops a full close being commanded during the
approach or a release — it just does not dictate the value.

`c_obs` stays CONTINUOUS on purpose. The command is the same for rest and for a
grasp, so "am I actually holding something" lives in the observation — the jaws
reach the mechanical stop when empty and are blocked short when not. That is the
right place for it: something the policy can condition on rather than infer.

WHY
---
Replayed angles under-close. The demonstrated angle is where the human's fingers
sat *while pressing* the object, and a position servo reproducing it stops just
short. Measured over these datasets, demonstrations use only 38-60% of the
proximal range. Commanding `c = 1` removes the need to predict an
object-dependent angle at all.

Non-destructive: the source dataset is copied, never modified.

PUSHING THE RESULT TO THE HUB
----------------------------
`hf upload` is not sufficient on its own. LeRobot resolves a dataset by a git TAG
matching its `codebase_version` (e.g. `v3.0`), and `hf upload` only writes `main`.
Without the tag, training dies in `get_safe_version` with a
`RevisionNotFoundError` that lerobot then re-raises incorrectly, surfacing as an
unrelated `TypeError: HfHubHTTPError.__init__() missing ... 'response'`.

So after uploading:

    from huggingface_hub import create_tag
    create_tag(repo_id, tag="v3.0", repo_type="dataset", revision="main")

using whatever `codebase_version` meta/info.json reports.

The other thing `hf upload` skips is the dataset CARD. LeRobot's own
`push_to_hub` renders one from a template; uploading files directly does not, so
the repo lands with no README, no `LeRobot` tag, and — the part that is easy to
miss — no link to the online viewer. Include this block in the card, since the
Space takes the repo id as `path` (NOT `dataset`, which silently shows nothing):

    <a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=<repo_id>">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
    </a>

and keep `tags: [LeRobot]` in the front matter, which is what makes the dataset
discoverable as a LeRobot one.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gripette.grasp_projection import (
    PROJECTION_SIDECAR,
    GraspProjection,
    normalize_closing_sign,
    segment_grip,
)

# Card front matter for a converted dataset. `grasp-projection` is queryable
# through the Hub API without downloading anything, which is what makes a
# projected dataset identifiable from a listing; the channel names in info.json
# remain the authoritative signal.
PROJECTION_HUB_TAG = "grasp-projection"
_CARD_TAGS = ["LeRobot", "grabette", "gripette", PROJECTION_HUB_TAG]

logger = logging.getLogger(__name__)

C_CLOSE = 1.0   # every other frame passes the demonstrated closure through


def _gripper_columns(names: list[str] | None, dim: int) -> tuple[int, int]:
    """Locate (proximal, distal) BY NAME. Layouts differ between dataset eras —
    8-D [x,y,z,ax,ay,az,proximal,distal] and 11-D [dx,dy,dz,dr6d_0..5,
    proximal,distal] — so a positional guess silently reads rotation channels
    as gripper angles.
    """
    if names:
        low = [str(n).lower() for n in names]
        ip = next((i for i, n in enumerate(low) if "proximal" in n), None)
        idl = next((i for i, n in enumerate(low) if "distal" in n and "dr6d" not in n), None)
        if ip is not None and idl is not None:
            return ip, idl
    logger.warning("Gripper channels not found by name; falling back to the last two")
    return dim - 2, dim - 1


def convert_episode(
    prox: list[float],
    dist: list[float],
    gp: GraspProjection,
    rest_is_closed: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One episode -> (action gripper pairs, state gripper pairs, report)."""
    prox, flipped_p = normalize_closing_sign(prox)
    dist, flipped_d = normalize_closing_sign(dist)

    # Clamp negatives: the recorded open pose sits slightly BELOW zero (the
    # device's zero is a calibration, good to ~1 deg), and a negative angle has
    # no meaning in the projection.
    s_obs, c_obs = [], []
    for p_, d_ in zip(prox, dist):
        s, c = gp.encode(max(p_, 0.0), max(d_, 0.0))
        s_obs.append(s)
        c_obs.append(c)

    labels, events = segment_grip(c_obs, rest_is_closed=rest_is_closed)

    # Only the grasp is rewritten; approach, open and rest pass through, so the
    # decoded command matches the demonstration everywhere except where the
    # under-close happens.
    c_target = [C_CLOSE if lab == "close" else c_obs[i]
                for i, lab in enumerate(labels)]

    # LATCH the strategy across each close. Left per-frame, `s` drifts while the
    # fingers settle onto the object, and since the commanded pose is
    # decode(s, 1.0), the target then wanders — observed swinging between 57 and
    # 93 degrees of proximal mid-grasp, which would make a servo already stalled
    # on the object shuffle. A grasp has ONE shape, so it is taken from the pose
    # the hand ends in (the plateau) and held from the onset.
    s_cmd = list(s_obs)
    for ev in (e for e in events if e.kind == "close"):
        tail = s_obs[max(ev.end - 5, ev.onset):ev.end + 1]
        if not tail:
            continue
        grasp_s = sorted(tail)[len(tail) // 2]
        end = next((i for i in range(ev.onset, len(labels)) if labels[i] != "close"),
                   len(labels))
        for i in range(ev.onset, end):
            s_cmd[i] = grasp_s
    return (
        np.asarray(list(zip(s_cmd, c_target)), dtype=np.float32),
        np.asarray(list(zip(s_obs, c_obs)), dtype=np.float32),
        {
            "n_close": sum(1 for e in events if e.kind == "close"),
            "n_open": sum(1 for e in events if e.kind == "open"),
            "sign_flipped": bool(flipped_p or flipped_d),
            "frac_close": labels.count("close") / max(len(labels), 1),
        },
    )


def _drop_stale_action_stats(dst_root: Path) -> None:
    """Remove action-stats provenance copied from the source dataset.

    The source tree is copied wholesale, so if it carried chunk-relative action
    stats its `action_relative_meta` marker survives — while this conversion has
    just recomputed `action` for the NEW gripper channels, i.e. as absolute
    stats. A dataset then claims relative stats while holding absolute ones,
    which is exactly the mismatch the training guard exists to catch, and the
    stale marker would make that guard PASS.

    `action_absolute` goes for the same reason: it archives the pre-relative
    stats of a different representation.
    """
    path = dst_root / "meta" / "stats.json"
    if not path.is_file():
        return
    stats = json.loads(path.read_text())
    dropped = [k for k in ("action_relative_meta", "action_absolute") if k in stats]
    if not dropped:
        return
    for k in dropped:
        del stats[k]
    path.write_text(json.dumps(stats, indent=2) + "\n")
    logger.info("Dropped stale action-stats provenance: %s", ", ".join(dropped))


def _write_sidecar(dst_root: Path, gp: GraspProjection, src_root: Path) -> None:
    """Record which projection built this dataset, and from what.

    Written every conversion, unconditionally: a projected dataset with no record
    of its calibration is the ambiguity this file exists to remove.
    """
    path = dst_root / PROJECTION_SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        gp.to_metadata(source_root=str(src_root)), indent=2) + "\n")
    logger.info("Wrote %s", path)


def _write_card(dst_root: Path, repo_id: str | None = None) -> None:
    """Write a minimal dataset card if there is none.

    `hf upload` does not render LeRobot's card template, so a hand-uploaded
    dataset otherwise lands with no README, no tags and no viewer link. Never
    overwrites: a hand-written card carries more than this can.

    The viewer badge needs the repo id, which only exists at upload time — hence
    `repo_id`. Without it the card is still written (tags are the valuable part)
    and check_publish reports the missing viewer link.

    CAVEAT: `LeRobotDataset.push_to_hub` REPLACES README.md with LeRobot's own
    template, so this card does not survive that route — the template keeps a
    correct viewer badge (set `ds.repo_id` before pushing) but carries only the
    `LeRobot` tag, and check_publish then reports the missing `grasp-projection`
    tag. After such a push the tags have to be merged back into the uploaded
    card, preserving the template's `configs:` block, which drives the Hub
    dataset viewer. This card survives an `hf upload`, which renders no template
    of its own.
    """
    path = dst_root / "README.md"
    if path.exists():
        logger.info("%s exists; leaving it alone", path)
        return
    tags = "\n".join(f"- {t}" for t in _CARD_TAGS)
    badge = ""
    if repo_id:
        badge = (
            f'\n<a class="flex" href="https://huggingface.co/spaces/lerobot/'
            f'visualize_dataset?path={repo_id}">\n'
            '<img class="block dark:hidden" src="https://huggingface.co/datasets/'
            'huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>\n'
            '<img class="hidden dark:block" src="https://huggingface.co/datasets/'
            'huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>\n'
            "</a>\n"
        )
    path.write_text(
        f"---\ntask_categories:\n- robotics\ntags:\n{tags}\n---\n{badge}\n"
        f"# {dst_root.name}\n\n"
        "Gripper channels are **(strategy, closure)**, not joint angles: the last "
        "two channels of `action` and `observation.state` are normalised to the "
        "gripper's reachable travel. In `action`, `closure = 1.0` means \"drive "
        "fully closed\" and the object stops the fingers; in `observation.state` "
        "closure stays continuous, so \"am I holding something\" remains "
        "observable.\n\n"
        f"The exact calibration is recorded in `{PROJECTION_SIDECAR}`. Decode with "
        "`gripette.grasp_projection.GraspProjection` — see "
        "[docs/grasp_projection.md](https://github.com/pollen-robotics/grabette/"
        "blob/develop/docs/grasp_projection.md).\n"
    )
    logger.info("Wrote %s", path)


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    rest_is_closed: bool = False,
    overwrite: bool = False,
    stats_repo_id: str | None = None,
) -> dict:
    """Copy the dataset and rewrite its gripper channels. Returns a report."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst_root}. Pass --overwrite.")
        logger.warning("Overwriting %s", dst_root)
        shutil.rmtree(dst_root)

    logger.info("Copying %s -> %s", src_root, dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    # Dereferences symlinks, so the result is a self-contained snapshot and the
    # source (a shared HF cache) cannot be touched by anything downstream.
    shutil.copytree(src_root, dst_root, symlinks=False)

    # Open the dataset BEFORE rewriting anything.
    #
    # LeRobotDataset syncs from the Hub on construction even when `root` is
    # local, and it must be given a repo_id that exists (an invented one 404s,
    # and HF_HUB_OFFLINE makes the check raise rather than skip). Constructing it
    # here means that download lands on the freshly-copied files, which are still
    # identical to the source. Constructing it AFTER the rewrite re-downloads the
    # source parquet over the converted data and silently undoes everything —
    # observed, and only caught by re-running the round-trip check.
    ds = None
    if stats_repo_id:
        from lerobot.datasets import LeRobotDataset

        ds = LeRobotDataset(repo_id=stats_repo_id, root=dst_root)
        logger.info("Opened %s at %s for the stats pass", stats_repo_id, dst_root)

    info_path = dst_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    a_names = info["features"]["action"].get("names")
    a_dim = int(info["features"]["action"]["shape"][0])
    s_names = info["features"]["observation.state"].get("names")
    s_dim = int(info["features"]["observation.state"]["shape"][0])
    a_ip, a_id = _gripper_columns(a_names, a_dim)
    s_ip, s_id = _gripper_columns(s_names, s_dim)
    logger.info("action gripper cols %s, state gripper cols %s", (a_ip, a_id), (s_ip, s_id))

    gp = GraspProjection()
    reports = []
    for pf in sorted((dst_root / "data").rglob("*.parquet")):
        table = pq.read_table(pf)
        actions = np.array(table.column("action").to_pylist(), dtype=np.float32)
        states = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
        eps = np.array(table.column("episode_index").to_pylist())

        for ep in np.unique(eps):
            m = eps == ep
            a_pair, s_pair, rep = convert_episode(
                actions[m, a_ip].tolist(), actions[m, a_id].tolist(), gp, rest_is_closed
            )
            actions[m, a_ip], actions[m, a_id] = a_pair[:, 0], a_pair[:, 1]
            states[m, s_ip], states[m, s_id] = s_pair[:, 0], s_pair[:, 1]
            rep["episode"] = int(ep)
            reports.append(rep)

        cols = {}
        for c in table.column_names:
            if c == "action":
                cols[c] = pa.array(actions.tolist(), type=pa.list_(pa.float32()))
            elif c == "observation.state":
                cols[c] = pa.array(states.tolist(), type=pa.list_(pa.float32()))
            else:
                cols[c] = table.column(c)
        pq.write_table(pa.table(cols), pf)
        logger.info("  rewrote %s (%d rows)", pf.name, len(actions))

    # Rename the two channels so the new meaning is visible in the metadata
    # rather than implied. Shapes are unchanged.
    def _renamed(names, ip, idl):
        out = list(names)
        out[ip], out[idl] = "strategy", "closure"
        return out

    if a_names:
        info["features"]["action"]["names"] = _renamed(a_names, a_ip, a_id)
    if s_names:
        info["features"]["observation.state"]["names"] = _renamed(s_names, s_ip, s_id)
    info_path.write_text(json.dumps(info, indent=4))
    logger.info("Updated %s", info_path)

    if ds is not None:
        # Mandatory: `closure` is now 0..1 where the raw angle was 0..1.6 rad, so
        # stale stats would mis-scale the channel. Uses the object opened above —
        # constructing a new one here is what caused the clobber.
        from lerobot.datasets.dataset_tools import recompute_stats

        logger.info("Recomputing stats...")
        recompute_stats(ds, skip_image_video=True)
        logger.info("Stats recomputed")

    _drop_stale_action_stats(dst_root)
    _write_sidecar(dst_root, gp, src_root)
    _write_card(dst_root, repo_id=stats_repo_id)

    # WARN, don't refuse. A converted dataset that only ever lives locally for an
    # experiment is legitimate, so publication rules must not block conversion —
    # but the operator should hear about them here, while the context is fresh,
    # rather than from a training run that dies in `get_safe_version` an hour in.
    # Run `checks.publish.check_publish` as the hard gate before uploading.
    from grabette_postprocess.checks.publish import check_publish

    audit = check_publish(dst_root)
    for msg in audit["errors"]:
        logger.warning("publish check: %s", msg)
    for msg in audit["warnings"]:
        logger.warning("publish check: %s", msg)
    if audit["errors"] or audit["warnings"]:
        logger.warning(
            "%d publish issue(s) above — fix them before uploading; "
            "the git tag and card checks only run against a pushed repo",
            len(audit["errors"]) + len(audit["warnings"]),
        )

    return {
        "episodes": len(reports),
        "publish_errors": audit["errors"],
        "publish_warnings": audit["warnings"],
        "single_close": sum(1 for r in reports if r["n_close"] == 1),
        "no_close": sum(1 for r in reports if r["n_close"] == 0),
        "multi_close": sum(1 for r in reports if r["n_close"] > 1),
        "with_open": sum(1 for r in reports if r["n_open"] > 0),
        "sign_flipped": sum(1 for r in reports if r["sign_flipped"]),
        "frac_close_median": float(np.median([r["frac_close"] for r in reports])),
        "per_episode": reports,
    }


@click.command()
@click.option("--src_root", required=True, type=click.Path(exists=True),
              help="Source dataset root (a LeRobot dataset directory).")
@click.option("--dst_root", required=True, type=click.Path(),
              help="Destination root. The source is never modified.")
@click.option("--rest_is_closed", is_flag=True,
              help="Recording convention where the operator holds the gripper CLOSED "
                   "when idle. Then rest and grasp are the same command and only "
                   "opens are detected. Use for datasets recorded that way.")
@click.option("--overwrite", is_flag=True, help="Replace the destination if it exists.")
@click.option("--recompute_stats/--no_recompute_stats", default=True,
              help="Recompute normalisation stats. The gripper channels change "
                   "meaning and RANGE, so stale stats mis-normalise training.")
@click.option("--repo_id", default=None,
              help="repo_id to open the converted dataset under, for the stats "
                   "pass only; the data always comes from --dst_root. A LOCAL id "
                   "(local/anything) is the right choice and needs no Hub access: "
                   "LeRobotDataset only queries the Hub when the root is "
                   "incomplete, and by this point it is fully written. Do NOT pass "
                   "the SOURCE dataset's Hub id — opening the converted dataset "
                   "under it has re-downloaded source parquet over the converted "
                   "files, silently reverting the conversion.")
def main(src_root, dst_root, rest_is_closed, overwrite, recompute_stats, repo_id):
    """Re-express a dataset's gripper channels as (strategy, commanded closure)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = convert_dataset(Path(src_root), Path(dst_root),
                            rest_is_closed=rest_is_closed, overwrite=overwrite,
                            stats_repo_id=repo_id if recompute_stats else None)

    n = report["episodes"]
    click.echo("")
    click.echo(f"episodes:            {n}")
    click.echo(f"  exactly one close: {report['single_close']} "
               f"({100 * report['single_close'] / max(n, 1):.1f}%)")
    click.echo(f"  no close found:    {report['no_close']}")
    click.echo(f"  multiple closes:   {report['multi_close']}")
    click.echo(f"  with open events:  {report['with_open']}")
    click.echo(f"  sign-flipped:      {report['sign_flipped']}")
    click.echo(f"  median frames labelled close: "
               f"{100 * report['frac_close_median']:.0f}%")

    click.echo("")
    click.echo(f"Done: {dst_root}")


if __name__ == "__main__":
    main()
