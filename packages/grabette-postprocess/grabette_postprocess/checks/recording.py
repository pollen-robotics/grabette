"""Raw-recording content check for the OAK + Arducam rig (before SLAM).

Deliberately minimal and cheap: it answers one question — *is the data the
pipeline needs actually present and non-empty?* — without the expensive
full-video decodes and count/drift heuristics an exhaustive audit would run.

What it verifies:
  - SLAM inputs (all required):
      oakd_left.mp4               — video present, non-empty, decodable
      oakd_left_timestamps.json   — has samples
      oakd_depth.mkv | oakd_depth/ — depth present, non-empty
      oakd_depth_timestamps.json  — has samples
      oakd_imu.json               — accel + gyro present (rotation warned if absent)
      oakd_calib_offline.json     — required keys + positive intrinsics
  - Dataset inputs (required):
      angle_data.json             — has samples; each joint actually moves
      raw_video.mp4 (Arducam)     — video present, non-empty, decodable

`check_recording` runs one `_check_*` helper per subsystem (each appends to the
shared errors/warnings/info lists) and returns a status dict; both
scripts/checks/check_dataset.py (CLI) and the HF Space pipeline use it.
"""

import json
from pathlib import Path

import numpy as np
import av

from grabette_postprocess.episode_manager import find_trajectory_csv

# A gripper joint whose whole-episode peak-to-peak range is below this never
# meaningfully moved (sensor noise / encoder quantum is ~0.0015 rad). Kept well
# above the noise floor so we warn *only* when a joint is genuinely stuck, not
# when it moved a little. ~0.6° — real recordings show motion an order of
# magnitude larger, stuck joints an order of magnitude smaller.
ANGLE_STATIC_RANGE_RAD = 0.01


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _samples(path: Path) -> list:
    """Load the flat {"samples": [...]} schema (timestamps / imu / angle files)."""
    return _load_json(path).get("samples", [])


def _check_video(path: Path, label: str, status: dict, *, required: bool) -> None:
    """Presence + non-empty + decodable check for one video.

    Cheap by design: it reads the container header (resolution + frame count) and
    only decodes a single frame when the header omits the count — never the whole
    stream. A missing/empty/corrupt file is an error when required, a warning
    otherwise."""
    sink = status["errors"] if required else status["warnings"]
    if not path.is_file():
        sink.append(f"missing {path.name} ({label})")
        return
    if path.stat().st_size == 0:
        sink.append(f"{path.name} ({label}) is empty (0 bytes)")
        return
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            w, h, n = stream.width, stream.height, stream.frames
            if not n:  # header lacked a frame count — confirm ≥1 decodable frame
                for _ in container.decode(stream):
                    n = 1
                    break
    except Exception as e:
        sink.append(f"{path.name} ({label}) is unreadable: {e}")
        return
    if not (w > 0 and h > 0 and n > 0):
        sink.append(f"{path.name} ({label}) has no frames")
        return
    status["info"].append(f"{label} {w}x{h} {n}f")


def _check_arducam(ep_dir: Path, status: dict) -> None:
    """Arducam observation camera (raw_video.mp4) — required for the dataset."""
    _check_video(ep_dir / "raw_video.mp4", "arducam", status, required=True)


def _check_oak_cameras(ep_dir: Path, require_right: bool, status: dict) -> None:
    """OAK left video + its timestamps (SLAM inputs). The right camera is not a
    SLAM/dataset input, so it's only checked (non-fatally) when require_right."""
    _check_video(ep_dir / "oakd_left.mp4", "oak_left", status, required=True)
    lt = ep_dir / "oakd_left_timestamps.json"
    if not (lt.is_file() and _samples(lt)):
        status["errors"].append("oakd_left_timestamps.json missing or empty")
    if require_right:
        _check_video(ep_dir / "oakd_right.mp4", "oak_right", status, required=False)


def _check_depth(ep_dir: Path, status: dict) -> None:
    """Depth stream (SLAM input): a packed lossless video (oakd_depth.mkv) or a
    legacy PNG dir, plus non-empty timestamps."""
    err, info = status["errors"], status["info"]
    depth_mkv = ep_dir / "oakd_depth.mkv"
    depth_dir = ep_dir / "oakd_depth"
    depth_ts = ep_dir / "oakd_depth_timestamps.json"
    has_mkv = depth_mkv.is_file() and depth_mkv.stat().st_size > 0
    has_dir = depth_dir.is_dir() and any(depth_dir.glob("*.png"))
    n_ts = len(_samples(depth_ts)) if depth_ts.is_file() else 0
    info.append(f"depth {'mkv' if has_mkv else 'png' if has_dir else 'none'}/{n_ts}ts")
    if not has_mkv and not has_dir:
        err.append("missing depth (oakd_depth.mkv or non-empty oakd_depth/)")
    if not n_ts:
        err.append("oakd_depth_timestamps.json missing or empty")


def _check_seq_overlap(ep_dir: Path, status: dict) -> None:
    """left∩depth seq overlap drives the SLAM frame count. Cheap (the timestamp
    JSONs are already small); warn only when the streams share no frames at all —
    the data is present but would yield an empty SLAM run."""
    lt = ep_dir / "oakd_left_timestamps.json"
    depth_ts = ep_dir / "oakd_depth_timestamps.json"
    if not (lt.is_file() and depth_ts.is_file()):
        return
    left_seqs = {int(s["seq"]) for s in _samples(lt)}
    depth_seqs = {int(s["seq"]) for s in _samples(depth_ts)}
    overlap = len(left_seqs & depth_seqs)
    status["info"].append(f"slam_frames~{overlap}")
    if left_seqs and depth_seqs and overlap == 0:
        status["errors"].append(
            "left and depth timestamps share no seq — SLAM would get 0 frames")


def _check_imu(ep_dir: Path, status: dict) -> None:
    """OAK IMU (oakd_imu.json): accel + gyro are required SLAM inputs; rotation is
    optional (VIO init aid), so its absence only warns."""
    err, warn, info = status["errors"], status["warnings"], status["info"]
    imu_path = ep_dir / "oakd_imu.json"
    if not imu_path.is_file():
        err.append("missing oakd_imu.json")
        return
    counts = {"accel": 0, "gyro": 0, "rotation": 0}
    for s in _samples(imu_path):
        kind = s.get("kind")
        if kind in counts:
            counts[kind] += 1
    info.append("imu " + "/".join(f"{k[:3]}:{n}" for k, n in counts.items()))
    for kind in ("accel", "gyro"):
        if not counts[kind]:
            err.append(f"oak imu: no {kind} samples")
    if not counts["rotation"]:
        warn.append("oak imu: no rotation samples (VIO init aid absent)")


_JOINTS = ((0, "distal"), (1, "proximal"))  # value = [distal, proximal]


def _angle_matrix(samples: list) -> np.ndarray | None:
    """(N, 2) [distal, proximal] array from angle samples, or None when the
    samples aren't usable [distal, proximal] pairs."""
    raw = [s.get("value") for s in samples]
    if not raw or not all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in raw):
        return None
    return np.asarray([v[:2] for v in raw], dtype=float)


def static_gripper_joints(ep_dir: Path) -> list[str]:
    """Names of the gripper joints (``"distal"`` / ``"proximal"``) that never move
    over the whole episode — peak-to-peak range below ANGLE_STATIC_RANGE_RAD.

    Empty when the gripper moved, or when angle_data.json is missing/unusable (we
    can't claim a joint is stuck if we can't read it). The single source of truth
    for the static-joint judgement, shared by the recording warning below and the
    `fixed_gripper` episode tag (see checks.tags)."""
    angle_path = Path(ep_dir) / "angle_data.json"
    if not angle_path.is_file():
        return []
    vals = _angle_matrix(_samples(angle_path))
    if vals is None:
        return []
    return [name for axis, name in _JOINTS
            if float(np.ptp(vals[:, axis])) < ANGLE_STATIC_RANGE_RAD]


def _check_gripper(ep_dir: Path, status: dict) -> None:
    """Gripper joint angles (angle_data.json): required for the dataset. Beyond
    presence/non-empty, warn only when a joint genuinely never moves — the distal
    or proximal angle stays within ANGLE_STATIC_RANGE_RAD over the whole episode."""
    err, warn, info = status["errors"], status["warnings"], status["info"]
    angle_path = ep_dir / "angle_data.json"
    if not angle_path.is_file():
        err.append("missing angle_data.json (gripper)")
        return
    ang = _samples(angle_path)
    info.append(f"angle:{len(ang)}")
    if not ang:
        err.append("angle_data.json has no samples")
        return

    vals = _angle_matrix(ang)
    if vals is None:
        warn.append("angle: samples aren't [distal, proximal] pairs (can't assess motion)")
        return
    static = set(static_gripper_joints(ep_dir))
    for axis, name in _JOINTS:
        if name in static:
            rng = float(np.ptp(vals[:, axis]))
            warn.append(
                f"angle: {name} joint appears static (range {np.degrees(rng):.2f}° "
                f"over the whole episode)")


def _check_calib(ep_dir: Path, status: dict) -> None:
    """Offline calibration (oakd_calib_offline.json): required keys + intrinsics."""
    err = status["errors"]
    calib = ep_dir / "oakd_calib_offline.json"
    if not calib.is_file():
        err.append("missing oakd_calib_offline.json")
        return
    c = _load_json(calib)
    required = ["width", "height", "fx", "fy", "cx", "cy", "baseline", "imu_to_cam"]
    miss = [k for k in required if k not in c]
    if miss:
        err.append(f"calib missing keys: {miss}")
    elif not (c["fx"] > 0 and c["fy"] > 0 and c["baseline"] > 0):
        err.append(f"calib: bad intrinsics fx={c['fx']} fy={c['fy']} baseline={c['baseline']}")


def _check_existing_trajectory(ep_dir: Path, status: dict) -> None:
    """If SLAM already ran, record a "traj:tracked/total (pct%)" summary.

    Resolves the CSV via episodes.find_trajectory_csv and reads it with
    trajectory.load_trajectory_csv (the single reader of the trajectory schema) —
    imported lazily so the common pre-SLAM path doesn't pull scipy."""
    traj = find_trajectory_csv(ep_dir)
    if traj is not None:
        from grabette_postprocess.trajectory import load_trajectory_csv
        df = load_trajectory_csv(traj)
        tracked = len(df) - int(df["is_lost"].sum())
        pct = 100 * tracked / len(df) if len(df) else 0
        status["trajectory"] = f"traj:{tracked}/{len(df)} ({pct:.0f}%)"


def check_recording(ep_dir: Path, require_right: bool = True) -> dict:
    """Check one raw episode directory for content completeness, return a status
    dict.

    Keys: name, errors, warnings, info (lists of strings), and optionally
    trajectory (a "traj:tracked/total (pct%)" string if SLAM already ran).

    This is a content check only: it confirms the SLAM inputs plus angle_data.json
    and raw_video.mp4 are present and non-empty. It intentionally does NOT run the
    heavier quality heuristics (frame-drop detection, count-vs-metadata
    reconciliation, IMU staleness/rate) — those belong to a separate audit, and
    keeping this fast is what lets the Space check every episode up front.

    require_right: when False, the right OAK camera is not checked at all. The
    pipeline never consumes oakd_right.mp4 (SLAM is RGB-D on left+depth), so a
    caller that intentionally skips downloading it (e.g. the Space) sets this to
    avoid a spurious "missing oakd_right.mp4" report.
    """
    ep_dir = Path(ep_dir)
    status = {"name": ep_dir.name, "errors": [], "warnings": [], "info": []}

    _check_arducam(ep_dir, status)
    _check_oak_cameras(ep_dir, require_right, status)
    _check_depth(ep_dir, status)
    _check_seq_overlap(ep_dir, status)
    _check_imu(ep_dir, status)
    _check_gripper(ep_dir, status)
    _check_calib(ep_dir, status)
    _check_existing_trajectory(ep_dir, status)

    return status
