"""Is this dataset fit to publish (and to train on)?

The last stage of the pipeline, and the one that had no checks: everything below
was rediscovered from the failure end at least once, usually hours into a training
run or a robot session.

Runs against either a LOCAL dataset root (before uploading) or a `user/repo` id
(to verify what actually landed). The Hub-only checks are skipped for a local root
rather than reported as failures.

Deliberately metadata-only by default — no video decoding — so it is cheap enough
to run on every dataset every time. `deep=True` adds the one check that needs to
read frames.

WHY EACH CHECK EXISTS
---------------------
- TASK STRINGS. A VLA is conditioned on these. Datasets have shipped with
  `test_pick_mustard_200` — a directory name — where the language prompt belongs,
  which trains the model on a string no human would ever type at it.
- GRIPPER CHANNELS. A projected dataset (strategy, closure) and a raw one
  (proximal, distal radians) are indistinguishable by range alone: a raw dataset's
  q99 measured 0.894, below 1.0. Only the NAMES separate them, so unnamed channels
  are an error — downstream code otherwise falls back to "the last two", and a
  closure of 1.0 sent as 1.0 radian under-closes every grasp silently.
- THE GIT TAG. LeRobot resolves a dataset by a tag matching `codebase_version`,
  not by `main`. Without it, training dies in `get_safe_version` with a
  RevisionNotFoundError that lerobot re-raises incorrectly, surfacing as an
  unrelated `TypeError: HfHubHTTPError.__init__() missing ... 'response'`.
- THE CARD. `hf upload` writes files but does not render LeRobot's card template,
  so a hand-uploaded dataset has no README, no `LeRobot` tag, and no viewer link.
- FPS. Three different numbers were in play at once on this project: recorded at
  46, declared 50, streamed 30 at inference. Since actions are per-frame deltas,
  the control rate is the arm's speed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# The viewer Space takes the repo id as `path`. `?dataset=` renders a blank page,
# which looks like a broken dataset rather than a broken link.
VIEWER_HOST = "huggingface.co/spaces/lerobot/visualize_dataset"
VIEWER_PARAM = "path="

# The tag every LeRobot dataset must carry to be discoverable as one.
LEROBOT_TAG = "LeRobot"

# Gripper channel names this project uses, per representation.
RAW_GRIPPER = ("proximal", "distal")
PROJECTED_GRIPPER = ("strategy", "closure")

# A task string is a natural-language instruction, not an identifier. Identifiers
# are what actually leaked in: `test_pick_mustard_200`, `Cup Grasping`.
_IDENTIFIER_RE = re.compile(r"^[\w.-]+$")

# Normalised channels live in [0, 1]; allow the slack a float32 round-trip adds.
_UNIT_TOL = 1e-3


def check_publish(target: str | Path, deep: bool = False) -> dict:
    """Check a dataset for publication, return a status dict.

    Keys: name, errors, warnings, info (lists of strings) — the same shape as the
    other checks in this package, so a caller can treat them uniformly.

    target: a local dataset root, or a `user/repo` Hub id.
    deep:   also verify the frame rate by decoding frames (downloads a video).
    """
    src = _Source(target)
    status = {"name": src.name, "errors": [], "warnings": [], "info": []}

    info = _check_info(src, status)
    if info is None:
        return status                      # nothing else is meaningful without it

    _check_tasks(src, info, status)
    _check_gripper_channels(src, info, status)
    _check_stats(src, status)
    _check_card(src, status)
    _check_tag(src, info, status)
    _check_fps(src, info, status, deep)
    return status


# ---- the checks ---------------------------------------------------------------


def _check_info(src, status):
    """meta/info.json, and the fields everything downstream reads."""
    try:
        info = src.read_json("meta/info.json")
    except Exception as e:
        status["errors"].append(f"meta/info.json unreadable: {type(e).__name__}")
        return None
    for key in ("fps", "codebase_version", "total_episodes", "total_frames"):
        if info.get(key) in (None, ""):
            status["errors"].append(f"meta/info.json missing '{key}'")
    status["info"].append(
        f"{info.get('total_episodes')}ep {info.get('total_frames')}fr "
        f"{info.get('fps')}fps {info.get('codebase_version')}"
    )
    return info


def _check_tasks(src, info, status):
    """Task strings must be language — a VLA is conditioned on them verbatim."""
    try:
        tasks = src.read_parquet("meta/tasks.parquet")
    except Exception as e:
        status["errors"].append(f"meta/tasks.parquet unreadable: {type(e).__name__}")
        return
    # The strings are the INDEX (its name is 'task' in some writers, unset in others).
    names = [str(t) for t in tasks.index]
    if not names:
        status["errors"].append("meta/tasks.parquet has no tasks")
        return
    for t in names:
        if _IDENTIFIER_RE.match(t) or "_" in t:
            status["errors"].append(
                f"task {t!r} looks like an identifier, not an instruction — a VLA "
                "is conditioned on this string verbatim"
            )
    if info.get("total_tasks") not in (None, len(names)):
        status["warnings"].append(
            f"info.total_tasks={info.get('total_tasks')} but tasks.parquet has "
            f"{len(names)}"
        )
    status["info"].append(f"tasks={names}")

    # Second copy of the same strings, in the per-episode metadata. Training
    # resolves through task_index -> tasks.parquet, so a mismatch is not fatal —
    # but it means two files disagree about what the episodes are.
    ep = _episodes_meta(src, status)
    if ep is None or "tasks" not in ep.columns:
        return
    seen = {_first_task(t) for t in ep["tasks"]}
    unknown = seen - set(names)
    if unknown:
        status["warnings"].append(
            f"meta/episodes 'tasks' column disagrees with tasks.parquet: "
            f"{sorted(unknown)} not in the authoritative mapping"
        )


def _check_gripper_channels(src, info, status):
    """The gripper channels must say which representation they are in."""
    feats = info.get("features", {})
    for key in ("action", "observation.state"):
        feat = feats.get(key)
        if not feat:
            status["errors"].append(f"info.json has no '{key}' feature")
            continue
        names = feat.get("names") or []
        if not names or len(names) < 2:
            status["errors"].append(
                f"'{key}' channels are unnamed — downstream code falls back to "
                "'the last two', which cannot tell radians from a normalised closure"
            )
            continue
        last2 = tuple(str(n).lower() for n in names[-2:])
        if last2 == PROJECTED_GRIPPER:
            _check_unit_range(src, key, status)
        elif last2 == RAW_GRIPPER:
            _warn_if_unit_range(src, key, status)
        else:
            status["errors"].append(
                f"'{key}' last two channels are {last2}, expected "
                f"{RAW_GRIPPER} (raw angles) or {PROJECTED_GRIPPER} (projected)"
            )


def _check_unit_range(src, key, status):
    """Channels named (strategy, closure) are normalised, so they live in [0, 1]."""
    lo, hi = _channel_range(src, key, status)
    if hi is None:
        return
    if hi > 1.0 + _UNIT_TOL or lo < -_UNIT_TOL:
        status["errors"].append(
            f"'{key}' is named {PROJECTED_GRIPPER} but ranges [{lo:.3f}, {hi:.3f}] "
            "— normalised channels cannot leave [0, 1]; are these radians?"
        )


def _warn_if_unit_range(src, key, status):
    """Raw angles CAN happen to fit in [0, 1] rad (measured q99 0.894), so this is
    genuinely ambiguous and only a human can settle it."""
    lo, hi = _channel_range(src, key, status)
    if hi is None:
        return
    if hi <= 1.0 + _UNIT_TOL:
        status["warnings"].append(
            f"'{key}' is named {RAW_GRIPPER} (radians) but never exceeds "
            f"{hi:.3f} — indistinguishable from a normalised channel; confirm the "
            "names are right before training"
        )


def _channel_range(src, key, status):
    """(min, max) over the last two channels, from meta/stats.json."""
    try:
        stats = src.read_json("meta/stats.json")
    except Exception:
        return None, None
    s = stats.get(key)
    if not s or "min" not in s or "max" not in s:
        return None, None
    return float(min(s["min"][-2:])), float(max(s["max"][-2:]))


def _check_stats(src, status):
    """Training normalises from these; without them it cannot start."""
    if not src.exists("meta/stats.json"):
        status["errors"].append("meta/stats.json missing — training cannot normalise")


def _check_card(src, status):
    """The card, its LeRobot tag, and the viewer link."""
    if not src.exists("README.md"):
        status["warnings"].append(
            "no README.md — `hf upload` does not render LeRobot's card template, "
            "so the dataset has no description, no tag and no viewer link"
        )
        return
    card = src.read_text("README.md")
    if LEROBOT_TAG not in card:
        status["warnings"].append(
            f"card has no '{LEROBOT_TAG}' tag — it will not be discoverable as a "
            "LeRobot dataset"
        )
    if VIEWER_HOST not in card:
        status["warnings"].append("card has no link to the online dataset viewer")
    elif VIEWER_PARAM not in card:
        status["warnings"].append(
            f"viewer link is missing '?{VIEWER_PARAM}' — the Space takes the repo "
            "id as `path`, and any other parameter renders a blank page"
        )


def _check_tag(src, info, status):
    """LeRobot resolves a dataset by a git tag matching codebase_version."""
    if not src.is_repo:
        status["info"].append("local root: skipped the git-tag check")
        return
    want = str(info.get("codebase_version", ""))
    from huggingface_hub import list_repo_refs
    try:
        refs = list_repo_refs(src.target, repo_type="dataset")
    except Exception as e:
        status["warnings"].append(f"could not list repo refs: {type(e).__name__}")
        return
    tags = [t.name for t in refs.tags]
    if want not in tags:
        status["errors"].append(
            f"no git tag {want!r} (tags: {tags or 'none'}) — LeRobot resolves by "
            "tag, not main, and fails with a misleading TypeError without it"
        )


def _check_fps(src, info, status, deep):
    """Declared fps versus what the video says, and optionally the frames."""
    declared = info.get("fps")
    vid = next((k for k in info.get("features", {})
                if k.startswith("observation.images.")), None)
    if vid is None:
        return
    vinfo = info["features"][vid].get("info") or {}
    vfps = vinfo.get("video.fps")
    if vfps is not None and declared is not None and abs(float(vfps) - float(declared)) > 0.5:
        status["warnings"].append(
            f"info.fps={declared} but {vid} declares video.fps={vfps}"
        )
    if not deep:
        status["info"].append("fps: metadata only (pass deep=True to decode frames)")
        return
    _check_fps_deep(src, info, vid, status)


def _check_fps_deep(src, info, vid_key, status):
    """Duplicate frames mean the recorder sampled faster than the sensor ran.

    The per-frame `timestamp` column cannot answer this: it is synthesised as
    index/fps (exactly 20.000 ms, no jitter), so it only echoes the declared value.
    """
    import cv2

    path = src.video_path(info, vid_key, status)
    if path is None:
        return
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < 300:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 10:
        status["warnings"].append(f"{vid_key}: could not decode enough frames")
        return
    dups = sum(1 for a, b in zip(frames, frames[1:]) if not (a - b).any())
    frac = dups / (len(frames) - 1)
    status["info"].append(f"{vid_key}: {frac:.0%} duplicate frames over {len(frames)}")
    if frac > 0.05:
        status["warnings"].append(
            f"{vid_key}: {frac:.0%} of frames are exact duplicates — the recorder "
            f"sampled faster than the sensor; the true rate is nearer "
            f"{float(info['fps']) * (1 - frac):.0f} than the declared {info['fps']}"
        )


# ---- local root or Hub repo, behind one interface -----------------------------


class _Source:
    """Reads dataset files from a local root or a Hub repo, identically."""

    def __init__(self, target: str | Path):
        self.target = str(target)
        self.root = Path(target)
        # A repo id is `owner/name` and is not a directory on disk.
        self.is_repo = not self.root.exists() and self.target.count("/") == 1
        if not self.is_repo and not self.root.exists():
            raise FileNotFoundError(f"{target} is neither a directory nor a repo id")
        self.name = self.target if self.is_repo else self.root.name

    def _path(self, rel: str) -> Path:
        if not self.is_repo:
            return self.root / rel
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(self.target, rel, repo_type="dataset"))

    def exists(self, rel: str) -> bool:
        if not self.is_repo:
            return (self.root / rel).exists()
        from huggingface_hub import list_repo_files
        if not hasattr(self, "_files"):
            self._files = set(list_repo_files(self.target, repo_type="dataset"))
        return rel in self._files

    def read_json(self, rel: str):
        return json.loads(self._path(rel).read_text())

    def read_text(self, rel: str) -> str:
        return self._path(rel).read_text()

    def read_parquet(self, rel: str):
        import pandas as pd
        return pd.read_parquet(self._path(rel))

    def episode_meta_files(self) -> list[str]:
        if not self.is_repo:
            base = self.root / "meta" / "episodes"
            return [str(p.relative_to(self.root)) for p in sorted(base.rglob("*.parquet"))]
        from huggingface_hub import list_repo_files
        return sorted(f for f in list_repo_files(self.target, repo_type="dataset")
                      if f.startswith("meta/episodes/") and f.endswith(".parquet"))

    def video_path(self, info, vid_key, status):
        """Path to the first video file, resolved through the episode metadata."""
        try:
            import pandas as pd
            ep = pd.concat([self.read_parquet(f) for f in self.episode_meta_files()],
                           ignore_index=True)
            r = ep.iloc[0]
            rel = info["video_path"].format(
                video_key=vid_key,
                chunk_index=int(r[f"videos/{vid_key}/chunk_index"]),
                file_index=int(r[f"videos/{vid_key}/file_index"]),
            )
            return self._path(rel)
        except Exception as e:
            status["warnings"].append(f"could not locate a video file: {type(e).__name__}")
            return None


def _episodes_meta(src, status):
    try:
        import pandas as pd
        files = src.episode_meta_files()
        if not files:
            return None
        return pd.concat([src.read_parquet(f) for f in files], ignore_index=True)
    except Exception as e:
        status["warnings"].append(f"meta/episodes unreadable: {type(e).__name__}")
        return None


def _first_task(t):
    """The per-episode `tasks` cell is a sequence (list or ndarray by writer)."""
    if isinstance(t, str):
        return t
    try:
        return str(t[0])
    except (IndexError, TypeError):
        return str(t)
