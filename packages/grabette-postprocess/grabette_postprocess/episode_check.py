"""Raw-episode health checks for the OAK + Arducam rig.

Validates the recordings produced by the current hardware before SLAM:
  - Arducam observation camera : raw_video.mp4 (+ frame_timestamps.json)
  - OAK RGBD                   : oakd_left/right.mp4, oakd_depth/ (+ *_timestamps.json)
  - OAK IMU                    : oakd_imu.json (accel + gyro + rotation)
  - Gripper                    : angle_data.json (joint angles)
  - SLAM outputs (if present)  : camera_trajectory.csv

Counts are cross-checked against metadata.json. `check_episode` returns a status
dict; both scripts/check_dataset.py (CLI) and the HF Space pipeline use it.
"""

import json
from pathlib import Path

import numpy as np
import av

# Nominal sample rates of the rig (used to estimate expected sample counts).
OAK_IMU_HZ = 200.0   # BNO086 accel/gyro/rotation on the OAK
GRAVITY = 9.81       # m/s², expected accel norm at rest


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _video_info(path: Path) -> dict:
    """Return {frames, fps, duration, res} for a video, decoding to count if needed."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        res = f"{stream.width}x{stream.height}"
        n = stream.frames
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if not n:  # some encodings don't store frame count in the header
            n = sum(1 for _ in container.decode(stream))
    return {"frames": n, "fps": fps, "duration": duration, "res": res}


def _samples(path: Path) -> list:
    """Load the flat {"samples": [...]} schema (timestamps / imu / angle files)."""
    return _load_json(path).get("samples", [])


def _drops(host_ms: list[float], label: str) -> str | None:
    """Detect frame drops from a host_ms timestamp list (gap > 1.5x median interval)."""
    if len(host_ms) < 3:
        return None
    intervals = np.diff(np.asarray(host_ms, dtype=float))
    med = float(np.median(intervals))
    if med <= 0:
        return None
    drops = intervals > med * 1.5
    n = int(np.sum(drops))
    if n:
        missed = int(sum(round(intervals[i] / med) - 1 for i in np.where(drops)[0]))
        return f"{label}: {n} gaps ({missed} frames missed, ~{1000/med:.0f}Hz nominal)"
    return None


def _dupes(values: list, n_check: int = 200) -> float:
    """Percentage of consecutive-duplicate values in the first n_check samples."""
    m = min(len(values), n_check)
    if m < 10:
        return 0.0
    dupes = sum(1 for i in range(1, m) if values[i] == values[i - 1])
    return 100.0 * dupes / m


def check_episode(ep_dir: Path, require_right: bool = True) -> dict:
    """Check one raw episode directory, return a status dict.

    Keys: name, errors, warnings, info (lists of strings), and optionally
    trajectory (a "traj:tracked/total (pct%)" string if SLAM already ran).

    require_right: when False, the right OAK camera is not checked at all. The
    pipeline never consumes oakd_right.mp4 (SLAM is RGB-D on left+depth), so a
    caller that intentionally skips downloading it (e.g. the Space) sets this to
    avoid a spurious "missing oakd_right.mp4" error.
    """
    ep_dir = Path(ep_dir)
    status = {"name": ep_dir.name, "errors": [], "warnings": [], "info": []}
    err, warn, info = status["errors"], status["warnings"], status["info"]

    meta = _load_json(ep_dir / "metadata.json") if (ep_dir / "metadata.json").is_file() else {}
    duration = float(meta.get("duration_seconds", 0.0))
    oak_meta = meta.get("oakd", {})

    # --- Arducam observation camera ---------------------------------------
    raw = ep_dir / "raw_video.mp4"
    if not raw.is_file():
        err.append("missing raw_video.mp4 (Arducam)")
    else:
        v = _video_info(raw)
        info.append(f"arducam {v['res']} {v['frames']}f@{v['fps']:.0f}")
        if not duration:
            duration = v["duration"]
        meta_frames = int(meta.get("frame_count", 0))
        if meta_frames and abs(meta_frames - v["frames"]) > 1:
            warn.append(f"arducam: metadata {meta_frames}f but video {v['frames']}f")

        ft = ep_dir / "frame_timestamps.json"
        if ft.is_file():
            ts = _load_json(ft)
            if not ts:
                warn.append("frame_timestamps.json is empty (dataset will use uniform fps)")
            else:
                if abs(len(ts) - v["frames"]) > 1:
                    warn.append(f"arducam: {len(ts)} timestamps != {v['frames']} video frames")
                d = _drops(ts, "arducam")
                if d:
                    warn.append(d)

    # --- OAK RGBD ---------------------------------------------------------
    for side in (("left", "right") if require_right else ("left",)):
        mp4 = ep_dir / f"oakd_{side}.mp4"
        ts_path = ep_dir / f"oakd_{side}_timestamps.json"
        if not mp4.is_file():
            err.append(f"missing oakd_{side}.mp4")
            continue
        v = _video_info(mp4)
        n_ts = len(_samples(ts_path)) if ts_path.is_file() else 0
        info.append(f"oak_{side} {v['frames']}f/{n_ts}ts")
        exp = int(oak_meta.get(f"{side}_frames", 0))
        if exp and abs(exp - n_ts) > 2:
            warn.append(f"oak_{side}: metadata {exp} frames but {n_ts} timestamps")
        if ts_path.is_file():
            d = _drops([s["host_ms"] for s in _samples(ts_path)], f"oak_{side}")
            if d:
                warn.append(d)

    # depth — either a packed lossless video (oakd_depth.mkv) or a legacy PNG dir
    depth_dir = ep_dir / "oakd_depth"
    depth_mkv = ep_dir / "oakd_depth.mkv"
    depth_ts = ep_dir / "oakd_depth_timestamps.json"
    n_depth_ts = len(_samples(depth_ts)) if depth_ts.is_file() else 0
    if depth_mkv.is_file():
        # The video frames aren't cheap to count without decoding; trust the
        # timestamps as the frame count (cross-checked against metadata below).
        info.append(f"depth video/{n_depth_ts}ts")
        exp = int(oak_meta.get("depth_frames", 0))
        if exp and abs(exp - n_depth_ts) > 2:
            warn.append(f"depth: metadata {exp} frames but {n_depth_ts} timestamps")
    elif depth_dir.is_dir():
        n_depth_png = len(list(depth_dir.glob("*.png")))
        info.append(f"depth {n_depth_png}png/{n_depth_ts}ts")
        if abs(n_depth_png - n_depth_ts) > 2:
            warn.append(f"depth: {n_depth_png} PNGs != {n_depth_ts} timestamps")
        exp = int(oak_meta.get("depth_frames", 0))
        if exp and abs(exp - n_depth_png) > 2:
            warn.append(f"depth: metadata {exp} frames but {n_depth_png} PNGs")
    else:
        err.append("missing depth (oakd_depth.mkv or oakd_depth/)")

    # left/depth seq overlap drives the SLAM frame count
    lt = ep_dir / "oakd_left_timestamps.json"
    if lt.is_file() and depth_ts.is_file():
        left_seqs = {int(s["seq"]) for s in _samples(lt)}
        depth_seqs = {int(s["seq"]) for s in _samples(depth_ts)}
        overlap = len(left_seqs & depth_seqs)
        smaller = min(len(left_seqs), len(depth_seqs))
        info.append(f"slam_frames~{overlap}")
        if smaller and overlap < smaller * 0.8:
            warn.append(f"only {overlap} left∩depth seqs (left {len(left_seqs)}, depth {len(depth_seqs)})")

    # --- OAK IMU ----------------------------------------------------------
    imu_path = ep_dir / "oakd_imu.json"
    if not imu_path.is_file():
        err.append("missing oakd_imu.json")
    else:
        samples = _samples(imu_path)
        by_kind = {"accel": [], "gyro": [], "rotation": []}
        for s in samples:
            if s.get("kind") in by_kind:
                by_kind[s["kind"]].append(s)
        info.append("imu " + "/".join(f"{k[:3]}:{len(v)}" for k, v in by_kind.items()))

        expected = int(duration * OAK_IMU_HZ)
        for kind, ss in by_kind.items():
            n = len(ss)
            if not n:
                err.append(f"oak imu: no {kind} samples")
                continue
            if expected and n < expected * 0.5:
                err.append(f"oak {kind}: {n} samples (expected ~{expected} @ {OAK_IMU_HZ:.0f}Hz)")
            elif expected and n < expected * 0.8:
                warn.append(f"oak {kind}: {n} samples (expected ~{expected})")
            dp = _dupes([s["value"] for s in ss])
            if dp > 30:
                warn.append(f"oak {kind}: {dp:.0f}% duplicate values (stale reads)")

        # accel magnitude sanity: should sit near 1g
        acc = by_kind["accel"]
        if acc:
            norms = np.linalg.norm([s["value"] for s in acc[:500]], axis=1)
            med = float(np.median(norms))
            if not (7.0 < med < 12.0):
                warn.append(f"oak accel: median |a|={med:.1f} m/s² (expected ~{GRAVITY}); units/scale?")

    # --- Gripper joint angles --------------------------------------------
    angle_path = ep_dir / "angle_data.json"
    if not angle_path.is_file():
        err.append("missing angle_data.json (gripper)")
    else:
        ang = _samples(angle_path)
        info.append(f"angle:{len(ang)}")
        exp = int(meta.get("angle_sample_count", 0))
        if exp and abs(exp - len(ang)) > 2:
            warn.append(f"angle: metadata {exp} samples but {len(ang)} in file")
        if not ang:
            err.append("angle_data.json has no samples")
        else:
            dim = len(ang[0].get("value", []))
            if dim != 2:
                warn.append(f"angle: value dim {dim} (expected 2 = [distal, proximal])")
            if duration and len(ang) / duration < 5:
                warn.append(f"angle: only {len(ang)/duration:.0f}Hz (sensor stalling?)")
            dp = _dupes([s["value"] for s in ang])
            if dp > 60:
                warn.append(f"angle: {dp:.0f}% duplicate values (sensor stuck?)")

    # --- Calib ------------------------------------------------------------
    calib = ep_dir / "oakd_calib_offline.json"
    if not calib.is_file():
        err.append("missing oakd_calib_offline.json")
    else:
        c = _load_json(calib)
        required = ["width", "height", "fx", "fy", "cx", "cy", "baseline", "imu_to_cam"]
        miss = [k for k in required if k not in c]
        if miss:
            err.append(f"calib missing keys: {miss}")
        elif not (c["fx"] > 0 and c["fy"] > 0 and c["baseline"] > 0):
            err.append(f"calib: bad intrinsics fx={c['fx']} fy={c['fy']} baseline={c['baseline']}")

    # --- SLAM outputs (if already run) -----------------------------------
    for name in ("camera_trajectory.csv", "mapping_camera_trajectory.csv"):
        traj = ep_dir / name
        if traj.is_file():
            import pandas as pd
            df = pd.read_csv(traj)
            tracked = len(df) - int(df["is_lost"].sum())
            pct = 100 * tracked / len(df) if len(df) else 0
            status["trajectory"] = f"traj:{tracked}/{len(df)} ({pct:.0f}%)"
            break

    return status
