"""Quantify a Grabette dataset — RAW (8D absolute pose) or CONVERTED (11D delta).

Handles both action layouts the converter accepts, so it works at README step 1
(inspect the raw dataset, before conversion) AND step 2 (the converted dataset):
  raw 8D:        [x, y, z, ax, ay, az, proximal(6), distal(7)]   — absolute pose
  converted 11D: [dx, dy, dz, r6d_0..5, proximal(9), distal(10)] — per-step deltas
Position-delta / SLAM-spike checks are computed consistently for both (raw poses
are diffed within-episode; converted actions are already per-step deltas).

Prints the QA numbers that gate training: dataset size, per-step action-delta
magnitudes (position + rotation), supervision SNR (motion vs SLAM noise),
glitchy/truncated episode candidates, and video/parquet length mismatches.

Lightweight: reads only the `action`/`episode_index` columns (no video).

Usage:
  uv run python analyze_dataset.py --repo_id <id> [<id2> ...]
"""

import argparse
import numpy as np
from lerobot.datasets import LeRobotDataset, load_episodes
from scipy.spatial.transform import Rotation

from rotation import rotation_6d_to_rotation_matrix_numpy

def per_step_rotation_deg(A: np.ndarray, ep: np.ndarray, eps: np.ndarray, is_delta: bool) -> np.ndarray:
    """Per-step rotation-delta magnitude in degrees, for both action layouts.

    converted 11D → dims 3:9 are the per-step delta as 6D rotation;
    raw 8D       → dims 3:6 are the ABSOLUTE axis-angle pose: diff consecutive
                   frames within each episode (like the position deltas).
    """
    if is_delta:
        r6d = A[:, 3:9]
        Rd = rotation_6d_to_rotation_matrix_numpy(r6d)
        cos = np.clip((np.trace(Rd, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        drot = np.degrees(np.arccos(cos))
        # Episode-boundary deltas are zeroed to [0..0]; that degenerate 6D
        # yields a bogus angle — force those steps to 0.
        drot[np.linalg.norm(r6d, axis=1) < 1e-6] = 0.0
        return drot
    drot = np.zeros(len(A))
    for e in eps:
        idx = np.where(ep == e)[0]
        if len(idx) > 1:
            R = Rotation.from_rotvec(A[idx, 3:6])
            rel = R[1:] * R[:-1].inv()
            drot[idx[1:]] = np.degrees(rel.magnitude())
    return drot


def video_length_mismatches(meta) -> list[tuple[int, str, int, int]]:
    """Episodes whose stored video segment doesn't span `length` frames.

    Camera frames lost at capture/encode without padding make the video
    SHORTER than the parquet rows. Such episodes (a) desync image
    observations from actions after each gap (20 ms per missing frame at
    50 fps), and (b) crash LeRobot's `delete_episodes` with "Episode length
    mismatch" when they are KEPT while another episode in the same video
    file is deleted. Drop them before training/cleaning.

    Returns (episode_index, video_key, length, video_span) tuples; metadata
    only, no video decode.
    """
    if not meta.video_keys:
        return []
    if meta.episodes is None:
        meta.episodes = load_episodes(meta.root)
    bad = []
    for e in range(meta.total_episodes):
        ep_meta = meta.episodes[e]
        for vk in meta.video_keys:
            from_ts = ep_meta.get(f"videos/{vk}/from_timestamp")
            to_ts = ep_meta.get(f"videos/{vk}/to_timestamp")
            if from_ts is None or to_ts is None:
                continue  # pre-v3 metadata without per-episode video spans
            span = round(to_ts * meta.fps) - round(from_ts * meta.fps)
            if span != ep_meta["length"]:
                bad.append((e, vk, int(ep_meta["length"]), int(span)))
    return bad


def analyze(repo_id: str, root: str | None = None):
    ds = LeRobotDataset(repo_id, root=root)
    hf = ds.hf_dataset.select_columns(["action", "episode_index"])
    A = np.stack([np.asarray(a, dtype=np.float64) for a in hf["action"]])  # [T, 11]
    ep = np.asarray(hf["episode_index"])
    eps = np.unique(ep)

    print(f"\n{'='*64}\n  {repo_id}\n{'='*64}")

    # --- Layout detection. The converter accepts a raw 8D or converted 11D
    #     action; the gripper columns and the meaning of the first 3 dims
    #     differ, so branch on the action dim rather than assuming 11D. ---
    D = A.shape[1]
    if D == 8:
        is_delta = False
        layout = "raw 8D (absolute pose + axis-angle + 2D gripper)"
    elif D == 11:
        is_delta = True
        layout = "converted 11D (camera-local delta actions + 2D gripper)"
    else:
        print(f"  ERROR: action dim = {D}; expected 8 (raw) or 11 (converted).")
        print("  This tool inspects Grabette datasets. If this is a raw recording,")
        print("  it should be 8D; if converted, 11D. Nothing to analyze — skipping.")
        return

    # Per-step position-delta magnitude (mm), consistent across layouts:
    #   converted 11D → first 3 dims are already the per-step delta;
    #   raw 8D        → positions are absolute, so diff consecutive frames
    #                   WITHIN each episode (this is where SLAM spikes appear).
    if is_delta:
        dpos_mm = np.linalg.norm(A[:, :3], axis=1) * 1000.0
    else:
        dpos_mm = np.zeros(len(A))
        pos = A[:, :3]
        for e in eps:
            idx = np.where(ep == e)[0]
            if len(idx) > 1:
                dpos_mm[idx[1:]] = np.linalg.norm(np.diff(pos[idx], axis=0), axis=1) * 1000.0

    print(f"  layout: {layout}")
    lens = np.array([(ep == e).sum() for e in eps])
    print(f"  episodes: {len(eps)}   frames: {len(A)}   "
          f"ep length min/mean/max: {lens.min()}/{lens.mean():.0f}/{lens.max()}")

    # --- Action deltas (position) ---
    print(f"\n  POSITION DELTAS (per-step, mm)")
    print(f"    |Δpos| mean {dpos_mm.mean():.2f}  p50 {np.median(dpos_mm):.2f}  "
          f"p95 {np.percentile(dpos_mm,95):.2f}  max {dpos_mm.max():.2f}")

    # --- Supervision SNR (is the delta signal motion or SLAM noise?) ---
    # smooth-motion : residual ratio inside a 5-frame window; ≥3 is healthy.
    # Low SNR means the per-step deltas the policy must imitate are dominated
    # by tracking jitter — convert_dataset.py --smooth_poses exists to fix
    # exactly this.
    snr = []
    for e in eps:
        d = A[ep == e][:, :3] if is_delta else np.diff(A[ep == e][:, :3], axis=0)
        if len(d) < 12:
            continue
        k = 5
        smooth = np.stack([np.convolve(d[:, j], np.ones(k) / k, mode="valid")
                           for j in range(3)], 1)
        resid = d[k // 2: k // 2 + len(smooth)] - smooth
        snr.append(np.linalg.norm(smooth, axis=1).mean()
                   / max(np.linalg.norm(resid, axis=1).mean(), 1e-9))
    print(f"\n  SUPERVISION SNR (smooth-motion : noise, 5-frame window; ≥3 is healthy)")
    print(f"    median over episodes: {np.median(snr):.2f}" if snr else
          "    n/a (episodes too short)")

    # --- Action deltas (rotation) ---
    drot_deg = per_step_rotation_deg(A, ep, eps, is_delta)
    print(f"\n  ROTATION DELTAS (per-step, deg)")
    print(f"    |Δrot| mean {drot_deg.mean():.2f}  p50 {np.median(drot_deg):.2f}  "
          f"p95 {np.percentile(drot_deg,95):.2f}  p99 {np.percentile(drot_deg,99):.2f}  "
          f"max {drot_deg.max():.2f}")

    # --- Data-hygiene anomalies: glitchy (position-delta spike) + truncated ---
    SPIKE_MM = 80.0    # per-step delta this large = SLAM glitch, not motion (matches clean/convert --despike_max_mm; 80mm/step ≈ 4 m/s at 50fps)
    SPIKE_DEG = 5.0    # per-step rotation above this = SLAM orientation glitch (5°/step = 250°/s at 50fps; human wrist peaks ~150°/s). Matches convert's --despike_max_deg default: on a CONVERTED dataset this list should be empty; on a RAW dataset it previews what despike will zero.
    SHORT_FRAMES = 80  # episodes shorter than this are likely truncated/incomplete
    glitchy, rot_glitchy, short = [], [], []
    for e in eps:
        m = ep == e
        if dpos_mm[m].max() > SPIKE_MM:
            glitchy.append((int(e), round(float(dpos_mm[m].max()), 1)))
        if drot_deg[m].max() > SPIKE_DEG:
            rot_glitchy.append((int(e), round(float(drot_deg[m].max()), 1)))
        if m.sum() < SHORT_FRAMES:
            short.append((int(e), int(m.sum())))
    print(f"\n  ANOMALIES (candidates to drop before training)")
    print(f"    glitchy (Δpos spike >{SPIKE_MM}mm): {len(glitchy)} eps  "
          f"{glitchy[:12]}{' ...' if len(glitchy)>12 else ''}")
    print(f"    rot-glitchy (Δrot spike >{SPIKE_DEG}°): {len(rot_glitchy)} eps  "
          f"{rot_glitchy[:12]}{' ...' if len(rot_glitchy)>12 else ''}")
    print(f"    truncated (<{SHORT_FRAMES} frames): {len(short)} eps  "
          f"{short[:12]}{' ...' if len(short)>12 else ''}")

    # --- Video/parquet consistency (metadata only, v3 datasets) ---
    mismatched = video_length_mismatches(ds.meta)
    bad_eps = sorted({e for e, _, _, _ in mismatched})
    print(f"    video shorter than data (lost frames, MUST drop): {len(bad_eps)} eps  {bad_eps}")
    if bad_eps:
        worst = max(mismatched, key=lambda m: m[2] - m[3])
        print(f"      worst: ep {worst[0]} ({worst[1]}) is {worst[2] - worst[3]} frame(s) short. "
              f"These desync image obs from actions and crash delete_episodes if kept.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", nargs="+", required=True, help="One or more dataset repo ids to analyze/compare")
    p.add_argument("--root", default=None, help="Local dataset root (for a local-converted dataset, e.g. ~/.cache/huggingface/lerobot/local-converted/<repo--id>). Applies to all --repo_id.")
    args = p.parse_args()
    for rid in args.repo_id:
        analyze(rid, root=args.root)


if __name__ == "__main__":
    main()
