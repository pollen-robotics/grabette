"""Eval-set selection and best-checkpoint keeping for the pi05 launcher.

Both exist because a 20000-step run on a 23046-frame dataset (sugar cup,
2026-09-04) overfitted from its FIRST eval point — eval_loss 0.0679 at step 1000
rising monotonically to 0.2053 at step 20000 — and `--save_freq 10000` had saved
only the two worst checkpoints. Two separate defects:

1. lerobot holds out "the last ceil(n * eval_split) episodes per task", i.e. the
   contiguous TAIL of the recording session. Those episodes are maximally
   correlated with each other, so the eval loss measures roughly one situation
   repeated, and is both noisy and a weak generalisation signal. 8 of 150
   episodes, all recorded back to back.

2. Checkpoints are saved on a fixed step cadence with no regard for eval loss,
   and `push_to_hub` ships the LAST policy. There is no case where "keep the
   worst checkpoint" is wanted.

Neither is fixed by a fork: both hooks are module attributes on
`lerobot.scripts.lerobot_train`, patched here the same way the launcher already
patches `make_pre_post_processors`. Verified against the pinned revision
(e40b58a8) rather than the workspace's lerobot, which is a different version.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# lerobot emits exactly this immediately before the save in the same loop
# iteration, so the value is current when our save hook runs.
_EVAL_LINE = re.compile(r"step (\d+): eval_loss=([0-9.]+)")


class EvalLossCapture(logging.Handler):
    """Read eval_loss off lerobot's own log line.

    It is a local variable in `train()` with no accessor, and the log line is the
    only stable seam. Pinned revision, pinned format.
    """

    def __init__(self, state: dict):
        super().__init__()
        self.state = state

    def emit(self, record):
        try:
            m = _EVAL_LINE.search(record.getMessage())
        except Exception:
            return
        if m:
            self.state["step"] = int(m.group(1))
            self.state["loss"] = float(m.group(2))


def select_diverse(descriptors: dict[int, np.ndarray], k: int) -> list[int]:
    """Farthest-point sampling: k episodes that span the descriptor space.

    Greedy — start from the episode farthest from the centroid, then repeatedly
    take whichever is farthest from everything already chosen. That maximises
    coverage of where the object actually was, which is the point: an eval set of
    near-duplicates cannot tell you whether the policy generalises.
    """
    eps = sorted(descriptors)
    if k >= len(eps):
        return eps
    shapes = {np.asarray(descriptors[e]).shape for e in eps}
    if len(shapes) != 1:
        raise ValueError(
            f"descriptors have mixed shapes {sorted(shapes)}; every episode must "
            "be described the same way or they cannot be compared")
    X = np.stack([descriptors[e] for e in eps]).astype(np.float64)
    # Scale each dimension so one wide-ranging channel cannot dominate.
    sd = X.std(axis=0)
    X = X / np.where(sd > 1e-9, sd, 1.0)

    chosen = [int(np.argmax(np.linalg.norm(X - X.mean(axis=0), axis=1)))]
    d = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
    return sorted(eps[i] for i in chosen)


def select_stride(episodes: list[int], k: int) -> list[int]:
    """Evenly spaced through the recording order — the fallback when no
    descriptor is available. Already the default in ood_check.py."""
    if k >= len(episodes):
        return list(episodes)
    idx = np.linspace(0, len(episodes) - 1, k).round().astype(int)
    return sorted({episodes[i] for i in idx})


def episode_descriptors(actions_by_episode: dict[int, np.ndarray],
                        closure_col: int | None) -> dict[int, np.ndarray]:
    """One vector per episode describing the situation it represents.

    Where a closure channel exists the grasp POINT is the most meaningful
    descriptor — it is literally where the object was. Otherwise fall back to the
    trajectory's start, end and extent, which still separates episodes that go to
    different places.
    """
    out = {}
    for ep, a in actions_by_episode.items():
        if len(a) == 0:
            continue
        if closure_col is not None:
            # The frame of PEAK closure, not merely of a detected close: 12% of
            # sugar-cup episodes never reach 0.9 (the fingers stop on the cube),
            # and mixing 3-vectors for those that do with 9-vectors for those
            # that do not gives a ragged set that cannot be stacked. Peak closure
            # is the right answer for both — it is where the grasp was attempted.
            cl = a[:, closure_col]
            hit = np.where(cl >= 0.9)[0]
            i = int(hit[0]) if len(hit) else int(np.argmax(cl))
            out[ep] = np.asarray(a[i, :3], dtype=np.float64)
            continue
        out[ep] = np.concatenate([a[0, :3], a[-1, :3],
                                  a[:, :3].max(axis=0) - a[:, :3].min(axis=0)])
    return out


def plan_eval_split(episodes_by_task: dict[str, list[int]],
                    descriptors: dict[int, np.ndarray],
                    eval_split: float, mode: str) -> list[int]:
    """Which episodes to hold out, per task, matching lerobot's own count."""
    chosen: list[int] = []
    for task, eps in sorted(episodes_by_task.items()):
        k = math.ceil(len(eps) * eval_split)
        if k <= 0:
            continue
        have = {e: descriptors[e] for e in eps if e in descriptors}
        if mode == "diverse" and len(have) >= k:
            pick = select_diverse(have, k)
        elif mode == "tail":
            pick = sorted(eps)[len(eps) - k:]
        else:
            pick = select_stride(sorted(eps), k)
        chosen.extend(pick)
        logger.info("[eval-split] %s: %d/%d held out (%s) -> %s",
                    task or "<no task>", len(pick), len(eps), mode, pick)
    return sorted(set(chosen))


def spread(descriptors: dict[int, np.ndarray], subset: list[int]) -> float:
    """Mean distance from the centroid — how much ground a set covers."""
    pts = [descriptors[e] for e in subset if e in descriptors]
    if len(pts) < 2:
        return float("nan")
    X = np.stack(pts)
    return float(np.linalg.norm(X - X.mean(axis=0), axis=1).mean())


class BestCheckpointKeeper:
    """Track the best-on-eval checkpoint and bound disk while doing it.

    A pi05 checkpoint is 9.35 GB, so keeping every eval-cadence save is 100+ GB
    of bucket. Invariant: at most the best plus the one just written are on disk,
    and the final step is never pruned — eval loss is a proxy measured on a
    handful of episodes, and pick3's step-20000 checkpoint was 1.71x past its own
    eval minimum yet grasped 2/2 on the robot, so the last is worth keeping.

    Pruning is deferred by one save: lerobot calls `update_last_checkpoint` on
    the directory right after `save_checkpoint` returns, so deleting the
    just-written directory would leave that pointer dangling.
    """

    def __init__(self):
        self.best_loss = float("inf")
        self.best_step: int | None = None
        self.best_dir: Path | None = None
        self._prunable: Path | None = None

    def offer(self, checkpoint_dir: Path, step: int, loss: float | None,
              final_step: int | None) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        prev, self._prunable = self._prunable, None
        if prev is not None and prev != checkpoint_dir and prev != self.best_dir:
            shutil.rmtree(prev, ignore_errors=True)
            logger.info("[best] pruned %s", prev.name)

        if loss is None:
            return  # not an eval step: leave the cadence save alone
        if loss < self.best_loss:
            stale = self.best_dir
            self.best_loss, self.best_step, self.best_dir = loss, step, checkpoint_dir
            (checkpoint_dir / "grabette_best.json").write_text(json.dumps(
                {"step": step, "eval_loss": loss}, indent=2))
            logger.info("[best] step %d is the best so far (eval_loss=%.4f)", step, loss)
            if stale is not None and stale != checkpoint_dir:
                shutil.rmtree(stale, ignore_errors=True)
                logger.info("[best] pruned superseded best %s", stale.name)
        elif step != final_step:
            self._prunable = checkpoint_dir
