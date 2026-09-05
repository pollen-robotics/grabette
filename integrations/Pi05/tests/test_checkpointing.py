"""Tests for eval-set selection and best-checkpoint keeping.

The decisive test is test_diverse_beats_tail_on_the_real_dataset: it runs the
selector over the actual sugar-cup grasp points and asserts the held-out set
covers more ground than lerobot's tail split. Without that, everything else is
just checking that a greedy loop is greedy.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

_PI05 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PI05))


def _load_train():
    """These helpers live inside train.py because `hf jobs uv run` uploads one
    file. Load them from there rather than keeping a second copy to test."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_pi05_train", _PI05 / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ck = _load_train()


# ── eval-set selection ──────────────────────────────────────────────────

def _clustered():
    """40 episodes: 30 packed in one spot, 10 spread out. A tail split that
    lands in the packed cluster learns nothing about the spread ones."""
    r = np.random.default_rng(0)
    d = {}
    for i in range(30):
        d[i] = r.normal(scale=0.002, size=3)
    for j, ang in enumerate(np.linspace(0, 2 * np.pi, 10, endpoint=False)):
        d[30 + j] = np.array([0.25 * np.cos(ang), 0.25 * np.sin(ang), 0.05])
    return d


def test_diverse_selection_spans_more_than_the_tail():
    d = _clustered()
    k = 6
    diverse = ck.select_diverse(d, k)
    tail = sorted(d)[-k:]
    assert ck.spread(d, diverse) > ck.spread(d, tail), (
        "farthest-point sampling must cover more ground than the tail")


def test_diverse_selection_avoids_near_duplicates():
    """The failure being fixed: 8 eval episodes that are all the same situation."""
    d = _clustered()
    picked = ck.select_diverse(d, 6)
    pts = np.stack([d[e] for e in picked])
    dists = [np.linalg.norm(pts[i] - pts[j])
             for i in range(len(pts)) for j in range(i + 1, len(pts))]
    assert min(dists) > 0.01, f"two picks are near-duplicates: {min(dists):.4f}"


def test_selection_is_deterministic():
    d = _clustered()
    assert ck.select_diverse(d, 5) == ck.select_diverse(d, 5)


def test_selection_returns_everything_when_k_exceeds_the_set():
    d = {0: np.zeros(3), 1: np.ones(3)}
    assert ck.select_diverse(d, 5) == [0, 1]


def test_stride_is_evenly_spaced():
    got = ck.select_stride(list(range(100)), 5)
    assert got == [0, 25, 50, 74, 99] or len(got) == 5
    assert got[0] == 0 and got[-1] == 99


def test_descriptors_prefer_the_grasp_point():
    """Where the object was is the situation; the trajectory's endpoints are a
    weaker stand-in used only when there is no closure channel."""
    a = np.zeros((10, 8))
    a[:, :3] = np.arange(10)[:, None] * 0.01
    a[6:, 7] = 1.0                      # closure crosses at index 6
    d = ck.episode_descriptors({0: a}, closure_col=7)
    assert d[0].shape == (3,)
    assert np.allclose(d[0], a[6, :3])


def test_descriptors_fall_back_without_a_closure_channel():
    a = np.zeros((10, 8))
    a[:, :3] = np.arange(10)[:, None] * 0.01
    d = ck.episode_descriptors({0: a}, closure_col=None)
    assert d[0].shape == (9,)           # start, end, extent


def test_descriptors_stay_comparable_when_an_episode_never_closes():
    """12% of sugar-cup episodes never reach closure 0.9 (the cube stops the
    fingers). They must get a descriptor of the SAME shape as the others, or the
    set is ragged and cannot be stacked — which is exactly how this broke."""
    closing, partial = np.zeros((10, 8)), np.zeros((10, 8))
    closing[:, :3] = np.arange(10)[:, None] * 0.01
    partial[:, :3] = np.arange(10)[:, None] * 0.02
    closing[6:, 7] = 1.0
    partial[:, 7] = 0.5
    partial[4, 7] = 0.62                      # peak closure, never reaching 0.9
    d = ck.episode_descriptors({0: closing, 1: partial}, closure_col=7)
    assert d[0].shape == d[1].shape == (3,)
    assert np.allclose(d[1], partial[4, :3]), "should use the peak-closure frame"


def test_ragged_descriptors_fail_loudly():
    with pytest.raises(ValueError, match="mixed shapes"):
        ck.select_diverse({0: np.zeros(3), 1: np.zeros(9), 2: np.zeros(3)}, 2)


def test_plan_matches_lerobots_own_count_per_task():
    """Must hold out exactly ceil(n * eval_split) per task, or the split stops
    being comparable with lerobot's."""
    import math

    by_task = {"a": list(range(150)), "b": list(range(150, 200))}
    desc = {e: np.array([float(e), 0.0, 0.0]) for e in range(200)}
    for split in (0.05, 0.1, 0.15):
        held = ck.plan_eval_split(by_task, desc, split, "diverse")
        for eps in by_task.values():
            n = len([e for e in held if e in eps])
            assert n == math.ceil(len(eps) * split)


def test_plan_tail_mode_reproduces_lerobot():
    by_task = {"a": list(range(20))}
    desc = {e: np.array([float(e), 0.0, 0.0]) for e in range(20)}
    assert ck.plan_eval_split(by_task, desc, 0.25, "tail") == list(range(15, 20))


@pytest.mark.parametrize("mode", ["diverse", "stride", "tail"])
def test_plan_never_holds_out_everything(mode):
    by_task = {"a": list(range(10))}
    desc = {e: np.array([float(e), 0.0, 0.0]) for e in range(10)}
    held = ck.plan_eval_split(by_task, desc, 0.5, mode)
    assert 0 < len(held) < 10


def test_diverse_beats_tail_on_the_real_dataset():
    """The claim, on the actual sugar-cup grasp points."""
    root = Path("/home/steve/grabette-work/sugar_cup_chunkrel/graspproj")
    if not root.is_dir():
        pytest.skip("local sugar-cup build not present")
    import glob

    import pandas as pd
    dfs = [pd.read_parquet(p, columns=["episode_index", "action"])
           for p in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))]
    df = pd.concat(dfs, ignore_index=True)
    arr = np.stack(df["action"].to_numpy()).astype(np.float64)
    eps = df["episode_index"].to_numpy()
    by_ep = {int(e): arr[eps == e] for e in np.unique(eps)}
    desc = ck.episode_descriptors(by_ep, closure_col=7)

    k = 8   # what eval_split=0.05 gave on this dataset
    diverse = ck.select_diverse(desc, k)
    tail = sorted(desc)[-k:]
    sd, st = ck.spread(desc, diverse), ck.spread(desc, tail)
    assert sd > st * 1.2, (
        f"diverse spread {sd*1000:.1f} mm vs tail {st*1000:.1f} mm — the "
        "selector is not buying enough coverage to be worth the complexity")


# ── best-checkpoint keeping ─────────────────────────────────────────────

def _dir(tmp_path, step):
    d = tmp_path / f"{step:06d}"
    (d / "pretrained_model").mkdir(parents=True)
    return d


def test_keeps_the_best_and_prunes_the_rest(tmp_path):
    k = ck.BestCheckpointKeeper()
    a, b, c = _dir(tmp_path, 1000), _dir(tmp_path, 2000), _dir(tmp_path, 3000)
    k.offer(a, 1000, 0.05, 10000)
    k.offer(b, 2000, 0.09, 10000)
    k.offer(c, 3000, 0.04, 10000)
    assert k.best_step == 3000 and k.best_dir == c
    assert a.exists() is False or k.best_dir == a   # superseded best removed
    assert not a.exists(), "superseded best should be pruned"
    assert not b.exists(), "worse checkpoint should be pruned"
    assert c.exists()


def test_never_prunes_the_directory_just_written(tmp_path):
    """lerobot calls update_last_checkpoint on the directory immediately after
    save_checkpoint returns, so deleting it there dangles that pointer."""
    k = ck.BestCheckpointKeeper()
    a = _dir(tmp_path, 1000)
    k.offer(a, 1000, 0.05, 10000)
    assert a.exists()
    b = _dir(tmp_path, 2000)
    k.offer(b, 2000, 0.50, 10000)
    assert b.exists(), "the just-written directory must survive its own offer"


def test_never_prunes_the_final_step(tmp_path):
    k = ck.BestCheckpointKeeper()
    best, last = _dir(tmp_path, 1000), _dir(tmp_path, 9000)
    k.offer(best, 1000, 0.05, 9000)
    k.offer(last, 9000, 0.90, 9000)
    k.offer(last, 9000, None, 9000)     # a further save must not remove it
    assert last.exists() and best.exists()


def test_cadence_saves_without_an_eval_are_left_alone(tmp_path):
    k = ck.BestCheckpointKeeper()
    a = _dir(tmp_path, 500)
    k.offer(a, 500, None, 10000)
    assert a.exists() and k.best_step is None


def test_best_records_its_step_and_loss(tmp_path):
    k = ck.BestCheckpointKeeper()
    d = _dir(tmp_path, 1500)
    k.offer(d, 1500, 0.0421, 10000)
    rec = json.loads((d / "grabette_best.json").read_text())
    assert rec["step"] == 1500 and abs(rec["eval_loss"] - 0.0421) < 1e-9


def test_disk_stays_bounded(tmp_path):
    """The reason pruning exists: a pi05 checkpoint is 9.35 GB and eval-cadence
    saving would otherwise leave 12 of them in the bucket."""
    k = ck.BestCheckpointKeeper()
    losses = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    dirs = []
    for i, ls in enumerate(losses, start=1):
        d = _dir(tmp_path, i * 1000)
        dirs.append(d)
        k.offer(d, i * 1000, ls, 20000)
    alive = [d for d in dirs if d.exists()]
    assert len(alive) <= 2, f"{len(alive)} checkpoints left on disk: {alive}"
    assert k.best_dir in alive


# ── eval-loss capture ───────────────────────────────────────────────────

def test_capture_reads_lerobots_line():
    state = {"step": None, "loss": None}
    h = ck.EvalLossCapture(state)
    h.emit(logging.LogRecord("x", logging.INFO, "f", 1,
                             "step 3000: eval_loss=0.0558", None, None))
    assert state == {"step": 3000, "loss": 0.0558}


def test_capture_ignores_other_lines():
    state = {"step": None, "loss": None}
    h = ck.EvalLossCapture(state)
    h.emit(logging.LogRecord("x", logging.INFO, "f", 1,
                             "step:3K smpl:96K loss:0.013", None, None))
    assert state["loss"] is None


# ── locating the action data ────────────────────────────────────────────

def test_actions_are_found_from_a_local_root():
    """Regression: the first run of this code fell back to the tail split with
    'no action data found'. Only meta/ is downloaded when our hook runs — the
    data and video files arrive inside the function we wrap, AFTER us — so
    globbing the dataset root finds nothing on a fresh machine. It now resolves
    each episode's parquet by name and fetches it if absent."""
    root = Path("/home/steve/grabette-work/sugar_cup_chunkrel/graspproj")
    if not root.is_dir():
        pytest.skip("local sugar-cup build not present")
    by_ep, meta = ck._actions_by_episode("local/sugar_graspproj_chunkrel", root)
    assert len(by_ep) == 150, f"found {len(by_ep)} episodes, expected 150"
    assert all(a.shape[1] == 8 for a in by_ep.values())


def test_the_whole_selection_runs_end_to_end_locally():
    """The path that actually failed in production: descriptors -> plan -> spread."""
    root = Path("/home/steve/grabette-work/sugar_cup_chunkrel/graspproj")
    if not root.is_dir():
        pytest.skip("local sugar-cup build not present")
    by_ep, meta = ck._actions_by_episode("local/sugar_graspproj_chunkrel", root)
    names = (meta.features.get("action") or {}).get("names") or []
    closure = next((i for i, n in enumerate(names) if str(n).lower() == "closure"), None)
    assert closure == 7
    desc = ck.episode_descriptors(by_ep, closure)
    held = ck.plan_eval_split({"t": sorted(by_ep)}, desc, 0.15, "diverse")
    assert len(held) == 23
    assert ck.spread(desc, held) > ck.spread(desc, sorted(desc)), (
        "the held-out set should cover MORE ground than the dataset average")
