# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "lerobot[pi,dataset] @ git+https://github.com/huggingface/lerobot@e40b58a8dfa9e7b86918c374791599d070518d11",
#   "scipy", "sentencepiece", "num2words", "accelerate", "protobuf", "wandb",
#   "av",  # pyav video backend — HF-jobs images ship no FFmpeg shared libs,
#          # so torchcodec cannot load there; pass --dataset.video_backend=pyav
# ]
# ///
"""VLA fine-tuning launcher (pi05 / pi0_fast): stock `lerobot-train` with ONE
surgical fix.

Why this exists: when fine-tuning FROM a base checkpoint
(`--policy.pretrained_path=lerobot/pi05_base` or `lerobot/pi0fast-base`),
lerobot-train deserializes the processor pipeline SAVED ALONGSIDE that base
checkpoint instead of building one for YOUR run — so the pipeline carries the
base checkpoint's serialized settings rather than your policy-config
overrides and your dataset's normalization stats. (For pi0fast specifically
it also pins `action_tokenizer_name='physical-intelligence/fast'`, which is
unloadable on transformers v5 and the wrong tokenizer anyway — see
pi0fast/README.md.)

The fix: build the pipeline FRESH from the policy config (which carries your
CLI overrides) and the training dataset's stats, instead of deserializing the
base checkpoint's. Model weights still load from the base checkpoint; only
the data-processing pipeline is rebuilt. Everything else is stock
lerobot-train — all lerobot-train flags work unchanged.

Usage (full recipe in README.md):
  uv run python train.py --policy.type=pi05 \\
      --policy.pretrained_path=lerobot/pi05_base ...
"""

import dataclasses
import json
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path

import lerobot.scripts.lerobot_train as lerobot_train
import numpy as np
from lerobot.policies import factory as policy_factory

logger = logging.getLogger(__name__)

_original = policy_factory.make_pre_post_processors

# Opt-in chunk-relative actions (docs/relative_actions_lerobot_native.md).
# An env var, not a CLI flag: lerobot-train parses argv itself and rejects
# unknown flags. Pass it with `hf jobs uv run --env GRABETTE_CHUNK_RELATIVE=1`.
_CHUNK_RELATIVE = os.environ.get("GRABETTE_CHUNK_RELATIVE", "").lower() in ("1", "true", "yes")


# Marker key that `write_relative_action_stats` leaves in meta/stats.json. It
# survives lerobot's stats loading (cast_stats_to_numpy flattens/unflattens the
# whole dict), so training can see whether the dataset it was handed expects the
# chunk-relative step.
_RELATIVE_MARKER = "action_relative_meta"


def _as_int(value):
    """Unwrap a stats leaf to a plain int.

    Every leaf in meta/stats.json arrives as a numpy array: lerobot's
    `cast_stats_to_numpy` runs `np.atleast_1d` over the flattened dict. Calling
    `int()` on an ndim>0 array is deprecated and raises on newer numpy, so a
    1-element array is unwrapped explicitly.
    """
    flat = getattr(value, "flat", None)
    return int(next(iter(flat))) if flat is not None else int(value)


def _check_stats_match_representation(policy_cfg, dataset_stats):
    """Refuse to train when the dataset's stats and the action representation
    disagree. Both directions are silent failures that a healthy-looking loss
    curve will not reveal:

      - relative stats + absolute actions: what a 100-step smoke actually did on
        2026-09-02 because `hf jobs` uploaded the train.py from the wrong
        checkout. It ran to completion and checkpointed.
      - absolute stats + relative actions: rotation channels normalise OUTSIDE
        [-1, 1] (measured), i.e. values the model never sees in training.
    """
    meta = (dataset_stats or {}).get(_RELATIVE_MARKER)
    if meta is not None and not _CHUNK_RELATIVE:
        raise RuntimeError(
            f"this dataset's action stats are CHUNK-RELATIVE (meta/stats.json "
            f"carries '{_RELATIVE_MARKER}') but GRABETTE_CHUNK_RELATIVE is not "
            "set, so the actions would stay absolute and be normalised with the "
            "wrong scale. Set GRABETTE_CHUNK_RELATIVE=1 — and make sure the "
            "train.py being uploaded is the one containing the hook."
        )
    if meta is None and _CHUNK_RELATIVE:
        raise RuntimeError(
            "GRABETTE_CHUNK_RELATIVE is set but this dataset's action stats are "
            "ABSOLUTE. Run write_relative_action_stats(root, chunk_size=<policy "
            "chunk_size>) first, or the rotation channels normalise outside "
            "[-1, 1]."
        )
    if meta is None:
        return

    # The stats are horizon-dependent: a 50-frame offset spans further than a
    # 15-frame one, so stats computed for another chunk length are the wrong
    # scale even though nothing looks wrong.
    stats_chunk = _as_int(meta["chunk_size"])
    policy_chunk = _as_int(getattr(policy_cfg, "chunk_size", stats_chunk))
    if stats_chunk != policy_chunk:
        raise RuntimeError(
            f"action stats were computed for chunk_size={stats_chunk} but the "
            f"policy uses chunk_size={policy_chunk}. Relative-action stats scale "
            "with the horizon; recompute them for this chunk size."
        )
    print(f"[chunk-relative] dataset stats match: chunk_size={stats_chunk}")


def _install_chunk_relative(pre, post):
    """Swap LeRobot's elementwise relative steps for the composed ones.

    LeRobot's built-in subtracts `observation.state` componentwise, which is not
    rotation composition and yields world-frame offsets — unusable for a
    Cartesian action space recorded against an arbitrary SLAM origin. Ours
    composes into the chunk's reference frame instead.

    Raises if the expected steps are absent. Silently training on absolute
    actions while believing otherwise is the expensive failure here: the loss
    would look healthy and the policy would command garbage.
    """
    from grabette_chunkrel.chunk_relative_processor import (
        AbsoluteFromChunkRelativeStep,
        ChunkRelativeActionsStep,
    )

    pre_names = [type(s).__name__ for s in pre.steps]
    post_names = [type(s).__name__ for s in post.steps]
    if "RelativeActionsProcessorStep" not in pre_names:
        raise RuntimeError(
            f"no RelativeActionsProcessorStep to replace in {pre_names}; this "
            "policy's pipeline has no relative-action slot, so chunk-relative "
            "actions cannot be installed"
        )
    if "AbsoluteActionsProcessorStep" not in post_names:
        raise RuntimeError(
            f"no AbsoluteActionsProcessorStep to replace in {post_names}; "
            "without the inverse the policy's output stays relative"
        )
    if "NormalizerProcessorStep" in pre_names and (
        pre_names.index("RelativeActionsProcessorStep")
        > pre_names.index("NormalizerProcessorStep")
    ):
        raise RuntimeError(
            "the relative slot sits AFTER the normaliser; the dataset stats "
            "describe the relative distribution, so this ordering would "
            "normalise with the wrong scale"
        )

    forward = ChunkRelativeActionsStep()
    pre_steps = list(pre.steps)
    pre_steps[pre_names.index("RelativeActionsProcessorStep")] = forward
    post_steps = list(post.steps)
    post_steps[post_names.index("AbsoluteActionsProcessorStep")] = (
        AbsoluteFromChunkRelativeStep(relative_step=forward)
    )
    print(
        "[chunk-relative] installed: actions are offsets from each chunk's "
        "reference pose, composed into its frame. Dataset action stats MUST "
        "describe the relative distribution (write_relative_action_stats)."
    )
    return (dataclasses.replace(pre, steps=pre_steps),
            dataclasses.replace(post, steps=post_steps))


def _fresh_pipeline(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
    # Drop the pretrained pipeline source and the overrides that only apply
    # to the deserialization path; the fresh build takes everything it needs
    # (normalization stats, tokenizer names, device) from policy_cfg +
    # dataset_stats.
    kwargs.pop("preprocessor_overrides", None)
    kwargs.pop("postprocessor_overrides", None)
    _check_stats_match_representation(policy_cfg, kwargs.get("dataset_stats"))
    pre, post = _original(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs)
    if _CHUNK_RELATIVE:
        pre, post = _install_chunk_relative(pre, post)
    return pre, post



# ── eval-set selection and best-checkpoint keeping ──────────────────────
#
# INLINED DELIBERATELY. `hf jobs uv run` uploads ONE script — there is no way to
# ship a sibling module — so a separate checkpointing.py would ImportError in the
# container. The tests load these helpers out of this file rather than keeping a
# second copy.
#
# Eval-set selection and best-checkpoint keeping for the pi05 launcher.
#
# Both exist because a 20000-step run on a 23046-frame dataset (sugar cup,
# 2026-09-04) overfitted from its FIRST eval point — eval_loss 0.0679 at step 1000
# rising monotonically to 0.2053 at step 20000 — and `--save_freq 10000` had saved
# only the two worst checkpoints. Two separate defects:
#
# 1. lerobot holds out "the last ceil(n * eval_split) episodes per task", i.e. the
#    contiguous TAIL of the recording session. Those episodes are maximally
#    correlated with each other, so the eval loss measures roughly one situation
#    repeated, and is both noisy and a weak generalisation signal. 8 of 150
#    episodes, all recorded back to back.
#
# 2. Checkpoints are saved on a fixed step cadence with no regard for eval loss,
#    and `push_to_hub` ships the LAST policy. There is no case where "keep the
#    worst checkpoint" is wanted.
#
# Neither is fixed by a fork: both hooks are module attributes on
# `lerobot.scripts.lerobot_train`, patched here the same way the launcher already
# patches `make_pre_post_processors`. Verified against the pinned revision
# (e40b58a8) rather than the workspace's lerobot, which is a different version.

_EVAL_LINE = re.compile(r"step (\d+): eval_loss=([0-9.]+)")

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
        self._warned = False

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


# ── launcher wiring for the two features above ──────────────────────────
#
# Env vars rather than CLI flags: Env vars rather than CLI flags:
# lerobot-train parses argv itself and rejects unknown ones.
_EVAL_SELECT = os.environ.get("GRABETTE_EVAL_SELECT", "diverse").lower()
_SAVE_BEST = os.environ.get("GRABETTE_SAVE_BEST", "1").lower() not in ("0", "false", "no")

_EVAL_STATE: dict = {"step": None, "loss": None}
_KEEPER = BestCheckpointKeeper()
_orig_make_datasets = lerobot_train.make_train_eval_datasets
_orig_save_checkpoint = lerobot_train.save_checkpoint


def _actions_by_episode(repo_id, root):
    """Per-episode action arrays, read straight from the parquet files.

    Only meta/ has been downloaded at this point: `make_dataset` -- which fetches
    the data and video files -- runs inside the function we are wrapping, AFTER
    this. So globbing the dataset root finds nothing on a fresh machine, which is
    exactly how the first run of this code fell back to the tail split. Fetch the
    data parquet explicitly instead; it is a few MB and no video is touched.

    Avoids building a second LeRobotDataset, which would download everything.
    """
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    # DatasetInfo is a dataclass on newer lerobot and dict-style access is
    # deprecated there; the pinned revision may be either.
    try:
        n_eps = int(meta.info.total_episodes)
    except AttributeError:
        n_eps = int(meta.info["total_episodes"])
    rel = sorted({str(meta.get_data_file_path(e)) for e in range(n_eps)})

    paths = []
    for r in rel:
        local = Path(meta.root) / r
        if local.is_file():
            paths.append(local)
            continue
        from huggingface_hub import hf_hub_download

        paths.append(Path(hf_hub_download(repo_id, r, repo_type="dataset")))

    frames = [pd.read_parquet(p, columns=["episode_index", "action"]) for p in paths]
    if not frames:
        return {}, meta
    df = pd.concat(frames, ignore_index=True)
    arr = np.stack(df["action"].to_numpy()).astype(np.float64)
    eps = df["episode_index"].to_numpy()
    return {int(e): arr[eps == e] for e in np.unique(eps)}, meta


def _make_train_eval_datasets(cfg):
    """Hold out a set of episodes that SPANS the data instead of its tail.

    Implemented by reordering `cfg.dataset.episodes` so that the episodes we want
    held out are last within each task, then delegating to lerobot — which takes
    the tail. Reordering rather than reimplementing keeps this working across
    lerobot revisions: none of lerobot's dataset construction is duplicated here.
    """
    if _EVAL_SELECT == "tail" or getattr(cfg.dataset, "eval_split", 0.0) <= 0.0:
        return _orig_make_datasets(cfg)
    try:
        by_ep, meta = _actions_by_episode(cfg.dataset.repo_id, cfg.dataset.root)
        if not by_ep:
            raise RuntimeError("no action data found")
        names = (meta.features.get("action") or {}).get("names") or []
        closure_col = next(
            (i for i, n in enumerate(names) if str(n).lower() == "closure"), None)
        descriptors = episode_descriptors(by_ep, closure_col)

        base = list(cfg.dataset.episodes) if cfg.dataset.episodes else sorted(by_ep)
        ep_tasks = meta.episodes["tasks"]
        by_task: dict = {}
        for e in base:
            t = ep_tasks[e][0] if ep_tasks[e] else ""
            by_task.setdefault(str(t), []).append(e)

        held = plan_eval_split(
            by_task, descriptors, cfg.dataset.eval_split, _EVAL_SELECT)
        if not held:
            return _orig_make_datasets(cfg)

        held_set = set(held)
        cfg.dataset.episodes = ([e for e in base if e not in held_set]
                                + [e for e in base if e in held_set])
        logging.info(
            "[eval-split] mode=%s, %d held out; spread %.1f mm vs %.1f mm for the "
            "whole set (higher is better: a tail split measures one situation "
            "repeated)",
            _EVAL_SELECT, len(held),
            1000 * spread(descriptors, held),
            1000 * spread(descriptors, base))
    except Exception as e:  # noqa: BLE001 - never lose a run over the eval split
        logging.warning(
            "[eval-split] FELL BACK to lerobot's STOCK TAIL SPLIT (%r). The "
            "held-out episodes are the contiguous tail of the recording session "
            "and the eval loss is a weak generalisation signal. Not fatal, but "
            "this is not the split you asked for.", e)
        return _orig_make_datasets(cfg)
    return _orig_make_datasets(cfg)


def _save_checkpoint(*args, **kwargs):
    """Save as lerobot does, then remember the best-on-eval and prune the rest."""
    _orig_save_checkpoint(*args, **kwargs)
    if not _SAVE_BEST:
        return
    ckpt_dir = kwargs.get("checkpoint_dir", args[0] if args else None)
    step = kwargs.get("step", args[1] if len(args) > 1 else None)
    cfg = kwargs.get("cfg", args[2] if len(args) > 2 else None)
    if ckpt_dir is None or step is None:
        return
    loss = _EVAL_STATE["loss"] if _EVAL_STATE["step"] == step else None
    if loss is None and _EVAL_STATE["step"] is None and not _KEEPER._warned:
        _KEEPER._warned = True
        logging.warning(
            "[best] no eval_loss has been captured, so nothing can be selected "
            "and NOTHING WILL BE PRUNED — every save_freq checkpoint will be "
            "kept (9.35 GB each for pi05). Either --eval_steps is unset or the "
            "log capture is not attached.")
    _KEEPER.offer(Path(ckpt_dir), int(step), loss,
                  getattr(cfg, "steps", None) if cfg is not None else None)


def _rewrite_last_push_target(argv):
    """Send lerobot's end-of-training push (the LAST policy) to <repo_id>_step<N>.

    The best is uploaded separately to <repo_id>_best. Neither lands on the bare
    id, so a run can never silently replace a good checkpoint with a worse one —
    which is what shipped on 2026-09-04.
    """
    repo, steps = None, None
    for a in argv:
        if a.startswith("--policy.repo_id="):
            repo = a.split("=", 1)[1]
        elif a.startswith("--steps="):
            steps = a.split("=", 1)[1]
    if not (repo and steps) or repo.endswith(f"_step{steps}"):
        return argv, repo
    target = f"{repo}_step{steps}"
    # print, not logging: this runs before lerobot's main() configures logging,
    # so a logging call here goes nowhere and the rewrite looks like it never
    # happened. Observed on the first run with these patches.
    print(f"[checkpoints] last -> {target}, best -> {repo}_best", flush=True)
    return [f"--policy.repo_id={target}" if a.startswith("--policy.repo_id=") else a
            for a in argv], repo


_orig_init_logging = lerobot_train.init_logging


def _init_logging(*args, **kwargs):
    """Re-attach the eval-loss capture after lerobot resets logging.

    `init_logging` calls `logger.handlers.clear()` on the ROOT logger, so a
    handler installed before `main()` is silently discarded — which is exactly
    what happened on the first run: no [best] events, no pruning, and 16
    checkpoints of 9.35 GB each heading for the bucket. Re-adding it here is the
    only placement that survives.
    """
    _orig_init_logging(*args, **kwargs)
    root = logging.getLogger()
    if not any(isinstance(h, EvalLossCapture) for h in root.handlers):
        root.addHandler(EvalLossCapture(_EVAL_STATE))
    print("[checkpoints] eval-loss capture attached", flush=True)


def _push_best(repo_id):
    """Upload the best checkpoint's pretrained_model/ as a loadable repo."""
    if not (_SAVE_BEST and repo_id and _KEEPER.best_dir):
        return
    folder = Path(_KEEPER.best_dir) / "pretrained_model"
    if not folder.is_dir():
        logging.warning("[best] %s missing; nothing pushed", folder)
        return
    from huggingface_hub import HfApi

    target = f"{repo_id}_best"
    api = HfApi()
    api.create_repo(target, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=target, repo_type="model",
                      commit_message=f"best on eval: step {_KEEPER.best_step} "
                                     f"(eval_loss={_KEEPER.best_loss:.4f})")
    logging.info("[best] pushed step %d (eval_loss=%.4f) -> %s",
                 _KEEPER.best_step, _KEEPER.best_loss, target)


# Patch both the factory and the reference lerobot_train imported.
policy_factory.make_pre_post_processors = _fresh_pipeline
lerobot_train.make_pre_post_processors = _fresh_pipeline
lerobot_train.make_train_eval_datasets = _make_train_eval_datasets
lerobot_train.save_checkpoint = _save_checkpoint
lerobot_train.init_logging = _init_logging

if __name__ == "__main__":
    _base_repo = None
    if _SAVE_BEST:
        sys.argv, _base_repo = _rewrite_last_push_target(sys.argv)
    try:
        lerobot_train.main()
    finally:
        _push_best(_base_repo)
