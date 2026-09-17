"""Publish a two-camera copy of a sugar LeRobot dataset.

Every sugar checkpoint so far was trained on ONE view. The recording has two:
`chouziel/grabette-sugar-cup-2008` carries `right_cam0` (colour, fisheye) and
`right_cam1` (monochrome, narrower field of view, and the one where the sugar
jar is actually resolved). The conversion pipeline kept only cam0, so the
second stream has never reached a policy.

This script puts it back, WITHOUT re-running the conversion. That is safe
because of four facts, each verified before anything is written (see
`verify`):

  1. The two source streams are frame-synchronised. Cross-correlating
     per-frame motion energy over a 6 s window peaks at lag 0 (r = 0.56) and
     falls off sharply either side -- both cameras ride the same rig, so
     global image motion is a shared signal.
  2. Episode segmentation is identical. The derived datasets' per-episode
     `videos/.../cam0/{from,to}_timestamp` match the raw's `right_cam0`
     values exactly, and both sides total 23046 frames over 150 episodes.
  3. In the raw dataset cam1's offsets are IDENTICAL to cam0's -- one
     concatenated file per camera, cut the same way. So the new camera's
     offsets are a copy of the target's own cam0 columns; the raw's episode
     table is never joined against, only checked.
  4. lerobot's pi0.5 has no camera-count limit. `_preprocess_images` loops
     over `config.image_features`, which `make_policy` fills from dataset
     metadata, so a second video feature is picked up with no code change.

A copy rather than an in-place edit, for the same reason retask_dataset.py
makes copies: the existing single-camera checkpoints were trained against
these repos, and adding a feature to them would make their provenance a lie.
The single-camera runs stay reproducible, and the two-camera run differs from
them in exactly one variable.

    python add_camera.py --dry-run     # verify and report, write nothing
    python add_camera.py               # build, verify, push both datasets

Check the dry run first -- it is the only place the alignment guarantees are
tested, and everything downstream assumes them.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Source of the second view. Same 150 episodes as every derived sugar dataset.
RAW_REPO = "chouziel/grabette-sugar-cup-2008"
RAW_CAMERA = "right_cam1"

# What the new feature is called in the published copy. `cam1` and not
# `right_cam1` to match the `cam0` convention the eval loop sends
# (normalize_raw.py renames on the way in for exactly this reason).
NEW_CAMERA = "cam1"

# Both action representations, so the two-camera comparison can be read against
# the single-camera one on both sides. Same episodes, same prompt, same seed --
# only the camera count differs.
DATASETS = {
    # chunk-relative, 8-dim -- pairs with GRABETTE_CHUNK_RELATIVE=1
    "chunkrel": ("SteveNguyen/sugarcube_in_mug_chunkrel",
                 "SteveNguyen/sugarcube_in_mug_chunkrel_2cam"),
    # plain delta, 11-dim graspproj
    "delta": ("SteveNguyen/sugarcube_in_mug_graspproj",
              "SteveNguyen/sugarcube_in_mug_graspproj_2cam"),
}

IMG = "observation.images"
OFFSET_FIELDS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
STAT_FIELDS = ("count", "min", "max", "mean", "std",
               "q01", "q10", "q50", "q90", "q99")


def probe(video: Path, *fields: str) -> dict[str, str]:
    """ffprobe one video stream. Used to copy cam0's geometry rather than
    trust info.json, which is metadata and can disagree with the file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=" + ",".join(fields),
         "-of", "default=noprint_wrappers=1:nokey=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines())


def sole_video(root: Path, camera: str) -> Path:
    """The one mp4 for a camera.

    These datasets hold all 150 episodes in a single concatenated file per
    camera. Refusing anything else on purpose: a multi-file layout would need
    the raw's chunk/file indices mapped onto the target's, and silently
    getting that wrong produces a dataset that loads and trains on
    mismatched frames.
    """
    files = sorted((root / "videos" / f"{IMG}.{camera}").rglob("*.mp4"))
    if len(files) != 1:
        raise SystemExit(
            f"expected exactly one mp4 for {camera} under {root}, found "
            f"{len(files)}. This script assumes the single-file layout these "
            "datasets use; a chunked one needs the index mapping written."
        )
    return files[0]


def episode_table(root: Path) -> tuple[pa.Table, Path]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if len(files) != 1:
        raise SystemExit(f"expected one episode parquet under {root}, got {len(files)}")
    return pq.read_table(files[0]), files[0]


def verify(src: Path, raw: Path) -> dict:
    """Every assumption this script rests on, checked against the real files.

    Returns the facts the build needs. Raises rather than warns: each of these
    failing silently yields a dataset that loads, trains, and is wrong.
    """
    src_info = json.loads((src / "meta" / "info.json").read_text())
    raw_info = json.loads((raw / "meta" / "info.json").read_text())

    if f"{IMG}.{NEW_CAMERA}" in src_info["features"]:
        raise SystemExit(f"{src} already has {IMG}.{NEW_CAMERA}; nothing to add")
    if f"{IMG}.{RAW_CAMERA}" not in raw_info["features"]:
        raise SystemExit(f"{RAW_REPO} has no {IMG}.{RAW_CAMERA}")

    for field in ("total_episodes", "total_frames", "fps"):
        if src_info[field] != raw_info[field]:
            raise SystemExit(
                f"{field} differs: target {src_info[field]} vs raw "
                f"{raw_info[field]} — these are not the same recording"
            )

    src_ep, _ = episode_table(src)
    raw_ep, _ = episode_table(raw)
    if src_ep.num_rows != raw_ep.num_rows:
        raise SystemExit(f"episode counts differ: {src_ep.num_rows} vs {raw_ep.num_rows}")

    s_idx = np.asarray(src_ep.column("episode_index"))
    r_idx = np.asarray(raw_ep.column("episode_index"))
    if not np.array_equal(s_idx, r_idx):
        raise SystemExit(
            "episode_index order differs between target and raw; the "
            "per-episode stats copy below aligns by row and would scramble"
        )

    # Fact 2: the target's cam0 cuts are the raw's cam0 cuts.
    for field in ("from_timestamp", "to_timestamp"):
        a = np.asarray(src_ep.column(f"videos/{IMG}.cam0/{field}"))
        b = np.asarray(raw_ep.column(f"videos/{IMG}.right_cam0/{field}"))
        if not np.allclose(a, b):
            raise SystemExit(
                f"cam0 {field} does not match the raw's right_cam0 "
                f"(max diff {np.abs(a - b).max()}); the target was not derived "
                "from this recording without re-cutting"
            )

    # Fact 3: cam1 is cut identically to cam0, which is what lets the new
    # offsets be a copy of the target's own cam0 columns.
    for field in ("from_timestamp", "to_timestamp"):
        a = np.asarray(raw_ep.column(f"videos/{IMG}.right_cam0/{field}"))
        b = np.asarray(raw_ep.column(f"videos/{IMG}.{RAW_CAMERA}/{field}"))
        if not np.allclose(a, b):
            raise SystemExit(
                f"raw {RAW_CAMERA} {field} differs from right_cam0's; the two "
                "streams are cut differently and the offsets must be mapped, "
                "not copied"
            )

    cam0 = probe(sole_video(src, "cam0"),
                 "width", "height", "nb_frames", "codec_name", "pix_fmt")
    cam1_raw = probe(sole_video(raw, RAW_CAMERA), "width", "height", "nb_frames")
    if int(cam1_raw["nb_frames"]) != int(cam0["nb_frames"]):
        raise SystemExit(
            f"frame counts differ: cam0 {cam0['nb_frames']} vs raw "
            f"{RAW_CAMERA} {cam1_raw['nb_frames']}"
        )
    if int(cam0["nb_frames"]) != src_info["total_frames"]:
        raise SystemExit(
            f"cam0 holds {cam0['nb_frames']} frames but info.json says "
            f"total_frames={src_info['total_frames']}"
        )

    return {"info": src_info, "cam0": cam0, "cam1_raw": cam1_raw,
            "episodes": src_ep.num_rows, "frames": src_info["total_frames"]}


def encode(raw_video: Path, dst_video: Path, width: int, height: int) -> None:
    """Re-encode the raw second view to match cam0's geometry and GOP.

    ALL-INTRA (`-g 1`), because that is what cam0 measurably is: every frame a
    keyframe. lerobot's pipeline asks for `g=2` via pyav, which through its
    h264 encoder lands on all-keyframe output; reproducing that with the
    ffmpeg CLI needs `-g 1` (`-g 2` gives an alternating I/P stream, 50%
    keyframes). The GOP is not cosmetic: the dataloader seeks by timestamp, so
    a long GOP on one camera and none on the other makes the two streams cost
    wildly different amounts to sample.

    crf 30 / yuv420p / h264 are lerobot's defaults, with h264 chosen over the
    libsvtav1 default by the grabette pipeline (dataset.py: RGBEncoderConfig)
    so the Hub visualiser can play the result.
    """
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-i", str(raw_video),
         "-vf", f"scale={width}:{height}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-g", "1",
         "-an", str(dst_video)],
        check=True,
    )


def patch_info(staged: Path, width: int, height: int) -> None:
    """Add the new video feature, positioned right after cam0.

    Key order is preserved so the published info.json diffs cleanly against
    the single-camera original, and so `config.image_features` — which the
    policy iterates to build the prefix — puts cam0 first. Token order is the
    layout the attention tool reads back; keeping it stable keeps cam0's map
    in the same place it was in every earlier run.
    """
    path = staged / "meta" / "info.json"
    info = json.loads(path.read_text())
    cam0 = info["features"][f"{IMG}.cam0"]

    feature = {
        "dtype": "video",
        "shape": [3, height, width],
        "names": ["channels", "height", "width"],
        # Copied from cam0 and then corrected for size. Same codec, same
        # pix_fmt, same fps by construction (encode() above). `video.channels`
        # stays 3: the stream is monochrome in content but yuv420p on disk,
        # and lerobot decodes it to three equal channels.
        "info": dict(cam0["info"], **{"video.height": height, "video.width": width}),
    }

    features = {}
    for key, value in info["features"].items():
        features[key] = value
        if key == f"{IMG}.cam0":
            features[f"{IMG}.{NEW_CAMERA}"] = feature
    info["features"] = features
    path.write_text(json.dumps(info, indent=4))


def patch_episodes(staged: Path, raw: Path) -> int:
    """Give the new camera its per-episode offsets and image stats.

    Offsets are copies of the target's OWN cam0 columns — justified by
    `verify`, which checks that the raw cuts both cameras identically.

    Per-episode image stats come from the raw, which is the only place they
    exist for this camera. They were computed on the 960x720 original rather
    than the 480x360 copy; an area resize barely moves a per-channel mean, and
    pi0.5 normalises VISUAL features with IDENTITY (configuration_pi05.py), so
    nothing in training consumes them. They are written because the loader
    expects every video feature to have them.

    Any residual `right_cam1` columns are dropped last. The chunk-relative
    datasets carry them — leftovers from the conversion that dropped the
    camera — and leaving both would publish a dataset naming a feature that
    does not exist.
    """
    table, path = episode_table(staged)
    raw_table, _ = episode_table(raw)

    for field in OFFSET_FIELDS:
        table = table.append_column(
            f"videos/{IMG}.{NEW_CAMERA}/{field}",
            table.column(f"videos/{IMG}.cam0/{field}"),
        )
    for field in STAT_FIELDS:
        table = table.append_column(
            f"stats/{IMG}.{NEW_CAMERA}/{field}",
            raw_table.column(f"stats/{IMG}.{RAW_CAMERA}/{field}"),
        )

    stale = [c for c in table.column_names if RAW_CAMERA in c]
    if stale:
        table = table.drop_columns(stale)
    pq.write_table(table, path)
    return len(stale)


def patch_stats(staged: Path, raw: Path) -> None:
    """Add the new camera to meta/stats.json, from the raw's entry."""
    path = staged / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    raw_stats = json.loads((raw / "meta" / "stats.json").read_text())
    stats[f"{IMG}.{NEW_CAMERA}"] = raw_stats[f"{IMG}.{RAW_CAMERA}"]
    path.write_text(json.dumps(stats, indent=4))


def check_result(staged: Path, repo_id: str, frames: int) -> None:
    """Load the built dataset the way training will, and read one frame.

    The point is the decode: metadata can be self-consistent and still point
    at a video the loader cannot open, or at the wrong stream. Asserting cam1
    comes back monochrome is the cheap proof that the second view is the
    second view, and not a second copy of cam0.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=str(staged))
    keys = sorted(k for k in ds.meta.features if k.startswith(IMG))
    if keys != [f"{IMG}.cam0", f"{IMG}.{NEW_CAMERA}"]:
        raise SystemExit(f"unexpected camera set after build: {keys}")
    if ds.meta.total_frames != frames:
        raise SystemExit(f"total_frames moved: {ds.meta.total_frames} != {frames}")

    item = ds[len(ds) // 2]
    for key in keys:
        shape = tuple(item[key].shape)
        if shape != (3, 360, 480):
            raise SystemExit(f"{key} decoded to {shape}, expected (3, 360, 480)")

    cam0 = item[f"{IMG}.cam0"].float()
    cam1 = item[f"{IMG}.{NEW_CAMERA}"].float()
    # Tolerances are in the tensor's own units: lerobot hands back float in
    # [0, 1], but a uint8 path would make a fraction-of-1 threshold meaningless.
    scale = 255.0 if max(float(cam0.max()), float(cam1.max())) > 1.5 else 1.0

    spread = float((cam1.max(0).values - cam1.min(0).values).abs().max()) / scale
    if spread > 0.02:
        raise SystemExit(
            f"{NEW_CAMERA} decoded with colour (max channel spread {spread:.3f}); "
            "the raw second view is monochrome, so this is the wrong stream"
        )
    difference = float((cam0 - cam1).abs().mean()) / scale
    if difference < 0.01:
        raise SystemExit(
            f"cam0 and cam1 decoded to near-identical images (mean |diff| "
            f"{difference:.4f}); the new feature is probably pointing at cam0"
        )
    print(f"  load check: both cameras decode at {tuple(cam1.shape)}, "
          f"cam1 monochrome (channel spread {spread:.4f}), "
          f"mean |cam0-cam1| = {difference:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=sorted(DATASETS) + ["all"], default="all")
    ap.add_argument("--dry-run", action="store_true",
                    help="download, verify and report; write and push nothing")
    ap.add_argument("--no-push", action="store_true",
                    help="build and check locally, skip create_repo/upload")
    ap.add_argument("--stage-dir", default="/tmp",
                    help="where to assemble the copy (needs ~250 MB per dataset)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    print(f"second view: {RAW_REPO}:{IMG}.{RAW_CAMERA} -> {IMG}.{NEW_CAMERA}")
    raw = Path(snapshot_download(
        RAW_REPO, repo_type="dataset",
        # Only the second view's video is needed; the raw's cam0 stays on the
        # Hub. Its episode table is still checked, but that lives in meta/.
        allow_patterns=["meta/*", f"videos/{IMG}.{RAW_CAMERA}/*"],
    ))
    print(f"  raw snapshot: {raw}")

    for key in (sorted(DATASETS) if args.which == "all" else [args.which]):
        src_repo, dst_repo = DATASETS[key]
        print(f"\n=== {key}: {src_repo}  ->  {dst_repo}")
        src = Path(snapshot_download(src_repo, repo_type="dataset"))
        print(f"  source snapshot: {src}")

        facts = verify(src, raw)
        cam0 = facts["cam0"]
        width, height = int(cam0["width"]), int(cam0["height"])
        print(f"  verified: {facts['episodes']} episodes, {facts['frames']} frames, "
              f"cam0 {width}x{height} {cam0['codec_name']}/{cam0['pix_fmt']}")
        print(f"  raw {RAW_CAMERA}: {facts['cam1_raw']['width']}x"
              f"{facts['cam1_raw']['height']} -> will rescale to {width}x{height}")

        if args.dry_run:
            print("  dry run: verified only, nothing written")
            continue

        staged = Path(args.stage_dir) / f"add_camera_{key}"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(src, staged, symlinks=False)

        # Mirror cam0's chunk/file path exactly — the episode offsets copied
        # from cam0 name that same chunk_index/file_index pair.
        cam0_video = sole_video(staged, "cam0")
        dst_video = staged / "videos" / f"{IMG}.{NEW_CAMERA}" / \
            cam0_video.relative_to(staged / "videos" / f"{IMG}.cam0")
        print(f"  encoding {dst_video.relative_to(staged)} (all-intra, crf 30) …")
        encode(sole_video(raw, RAW_CAMERA), dst_video, width, height)

        made = probe(dst_video, "width", "height", "nb_frames", "codec_name", "pix_fmt")
        if int(made["nb_frames"]) != facts["frames"]:
            raise SystemExit(
                f"encode produced {made['nb_frames']} frames, expected "
                f"{facts['frames']} — refusing to publish a stream that cannot "
                "align with the action table"
            )
        print(f"  encoded: {made['width']}x{made['height']} "
              f"{made['codec_name']}/{made['pix_fmt']}, {made['nb_frames']} frames, "
              f"{dst_video.stat().st_size / 1e6:.0f} MB")

        patch_info(staged, width, height)
        dropped = patch_episodes(staged, raw)
        patch_stats(staged, raw)
        print(f"  metadata patched (dropped {dropped} stale {RAW_CAMERA} column(s))")

        check_result(staged, dst_repo, facts["frames"])

        if args.no_push:
            print(f"  --no-push: built and checked at {staged}")
            continue

        # Public, like the sources. `exist_ok` does not change the visibility
        # of a repo that already exists — flip that in the Hub settings.
        api.create_repo(dst_repo, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(staged), repo_id=dst_repo, repo_type="dataset",
            commit_message=f"Add second view {IMG}.{NEW_CAMERA} from "
                           f"{RAW_REPO}:{RAW_CAMERA} (copy of {src_repo}; "
                           "actions, prompt and episodes unchanged)",
        )
        print(f"  pushed {dst_repo}")

        # THE CODEBASE-VERSION TAG. upload_folder copies files, not refs, and
        # lerobot resolves a revision through get_safe_version(), which raises
        # RevisionNotFoundError on a repo with no version tag. An untagged copy
        # fails at LeRobotDatasetMetadata.__init__ — and that error cannot
        # construct itself in this huggingface_hub version
        # (HfHubHTTPError.__init__ missing 'response'), so the traceback ends
        # in an unrelated TypeError and never mentions tags. A copy is not a
        # usable dataset until this runs.
        version = json.loads((staged / "meta" / "info.json").read_text())[
            "codebase_version"
        ]
        api.create_tag(dst_repo, tag=version, repo_type="dataset", exist_ok=True)
        refs = api.list_repo_refs(dst_repo, repo_type="dataset")
        print(f"  tagged {version} — tags now {[t.name for t in refs.tags]}")


if __name__ == "__main__":
    main()
