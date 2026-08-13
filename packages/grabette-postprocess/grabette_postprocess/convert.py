"""Convert a grabette episode directory into the oak/ layout consumed by
run_oak_slam / docker/oak_vslam.

Our recording (grabette/hardware/oakd.py) stores compact mp4 + JSON sidecars
to save disk. The C++ offline_vslam expects per-file PNGs + CSVs. This module
produces <episode>/oak/{frames,depth,timestamps.csv,imu_acc.csv,imu_gyro.csv,
imu_rotation.csv,calib_offline.json} from <episode>/{oakd_left.mp4, oakd_depth.mkv
(or legacy oakd_depth/), oakd_left_timestamps.json, oakd_depth_timestamps.json,
oakd_imu.json, oakd_calib_offline.json}.

Depth is stored as a single lossless FFV1 16-bit video (oakd_depth.mkv) — one
file instead of ~600 PNGs, ~2× smaller, and bit-identical once decoded, so the
SLAM input is unchanged. Older recordings with an oakd_depth/ PNG directory are
still accepted.

Frame matching: left timestamps and depth timestamps share a seq number
(both come from the same StereoDepth node). We take seqs present in both,
in seq order, and assign consecutive idx = 0..N-1. mp4 frames are decoded
in encoding order (= seq order); we trim to whichever stream is shortest.

Timestamps in the output CSVs are in nanoseconds. Camera frames keep their
host_ms stamps — the clock the SLAM trajectory (and the downstream Arducam
alignment in dataset.py) lives on. IMU samples are placed on that same host
timeline via an affine fit of the OAK device clock → host clock (see
fit_device_to_host_s): the IMU shares the cameras' device clock, so mapping it
through the *frame* stream removes the ~tens-of-ms false camera-IMU offset that
using the IMU stream's own (differently-buffered over USB) host_ms injects.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _ms_to_ns(ms: float) -> int:
    return int(round(ms * 1e6))


def fit_device_to_host_s(left_ts_samples: list) -> tuple[float, float] | None:
    """Least-squares affine (slope, intercept) mapping the OAK device clock to
    the host clock, both in seconds: ``host_s ≈ slope * device_s + intercept``.

    Fitted on the OAK left frame stream (oakd_left_timestamps.json), which
    carries both ``device_us`` and ``host_ms`` per frame. Used to place the IMU
    — which shares the cameras' device clock — onto the frame host timeline
    instead of trusting the IMU stream's own host_ms (whose USB-transfer latency
    differs from the camera stream's by tens of ms). The same fit is reused by
    the sync check so it validates exactly what the pipeline produces.

    Returns None when device_us is absent (legacy recordings); callers then fall
    back to raw host_ms.
    """
    pts = [(s["device_us"], s["host_ms"]) for s in left_ts_samples
           if "device_us" in s and s.get("host_ms") is not None]
    if len(pts) < 2:
        return None
    dev_s = np.array([p[0] for p in pts], dtype=float) * 1e-6
    host_s = np.array([p[1] for p in pts], dtype=float) * 1e-3
    slope, intercept = np.polyfit(dev_s, host_s, 1)
    return float(slope), float(intercept)


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    """Run an ffmpeg command, surfacing its stderr on failure.

    A bare `check=True` raises a CalledProcessError whose message is only the
    argv + exit code — hiding the one line that says WHY ffmpeg failed. We
    capture stderr and fold its tail into the raised error so it reaches the
    caller's logs (incl. the HF Space run logs) instead of being swallowed.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-1000:] or "(no stderr)"
        raise RuntimeError(
            f"ffmpeg failed ({what}, exit {result.returncode}): {tail}\n"
            f"  command: {' '.join(cmd)}"
        )


# The PNG encoder defaults to frame-level multithreading. In a container with a
# restricted pids/threads cgroup (HF Spaces), spawning that thread pool fails
# with "ff_frame_thread_encoder_init failed" → "Error while opening encoder for
# output stream". Forcing a single thread bypasses the frame-thread wrapper
# entirely; output is byte-for-byte identical. MUST sit AFTER -i so it applies
# to the output (encoder), not the input (decoder) — verified via clone() count:
# after -i cuts the encoder pool, before -i does not.
_FFMPEG_PNG_THREADS = ["-threads", "1"]


def _extract_mp4_frames(mp4_path: Path, out_dir: Path) -> int:
    """Decode mp4 to GRAY8 6-digit PNGs starting at 000000.png. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mp4_path),
        *_FFMPEG_PNG_THREADS,
        "-pix_fmt", "gray",
        "-start_number", "0",
        str(out_dir / "%06d.png"),
    ]
    _run_ffmpeg(cmd, f"decode {mp4_path.name}")
    return len(list(out_dir.glob("*.png")))


def _extract_depth_video(mkv_path: Path, out_dir: Path) -> int:
    """Decode a lossless 16-bit depth video (FFV1 gray16le) to uint16 PNGs
    (000000.png…). Frame i is the i-th depth frame in encode order, which the
    packer (grabette.hf) writes in oakd_depth_timestamps.json order. No
    -pix_fmt on output: ffmpeg writes native 16-bit PNG, preserving the mm
    values exactly (verified bit-identical to the source PNGs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mkv_path),
        *_FFMPEG_PNG_THREADS,
        "-start_number", "0",
        str(out_dir / "%06d.png"),
    ]
    _run_ffmpeg(cmd, f"decode {mkv_path.name}")
    return len(list(out_dir.glob("*.png")))


def _split_imu_to_csvs(imu_json: Path, oak_dir: Path,
                       dev_to_host_s: tuple[float, float] | None = None) -> tuple[int, int, int]:
    """Split oakd_imu.json into the three CSVs offline_vslam reads
    (imu_acc.csv, imu_gyro.csv, imu_rotation.csv). Returns (n_accel, n_gyro, n_rot).

    rotation is the BNO086's fused orientation (IMU body → gravity-aligned world).
    RTAB-Map's Odometry can take this as orientation per IMU sample and use it for
    VIO initialization + drift constraint during rotations.

    When `dev_to_host_s` (slope, intercept) is given, each sample's device_us is
    mapped onto the frame host timeline so the IMU is correctly synced to the
    cameras; otherwise (or when a sample lacks device_us) the raw host_ms is used.
    """
    imu = json.loads(imu_json.read_text())["samples"]
    n_acc = n_gyr = n_rot = 0
    with (oak_dir / "imu_acc.csv").open("w") as f_a, \
         (oak_dir / "imu_gyro.csv").open("w") as f_g, \
         (oak_dir / "imu_rotation.csv").open("w") as f_r:
        f_a.write("timestamp_ns,ax,ay,az\n")
        f_g.write("timestamp_ns,wx,wy,wz\n")
        f_r.write("timestamp_ns,qx,qy,qz,qw\n")
        for s in imu:
            kind = s.get("kind")
            v = s.get("value")
            if dev_to_host_s is not None and "device_us" in s:
                slope, intercept = dev_to_host_s
                ts_ns = _ms_to_ns((slope * s["device_us"] * 1e-6 + intercept) * 1e3)
            else:
                ts_ns = _ms_to_ns(float(s["host_ms"]))
            if kind == "accel" and v and len(v) >= 3:
                f_a.write(f"{ts_ns},{v[0]},{v[1]},{v[2]}\n")
                n_acc += 1
            elif kind == "gyro" and v and len(v) >= 3:
                f_g.write(f"{ts_ns},{v[0]},{v[1]},{v[2]}\n")
                n_gyr += 1
            elif kind == "rotation" and v and len(v) >= 4:
                # oakd.py stores rotation as [i, j, k, real] = [qx, qy, qz, qw]
                f_r.write(f"{ts_ns},{v[0]},{v[1]},{v[2]},{v[3]}\n")
                n_rot += 1
    return n_acc, n_gyr, n_rot


def convert_episode(ep_dir: Path, force: bool = False) -> Path:
    oak_dir = ep_dir / "oak"
    if oak_dir.exists() and not force:
        print(f"  oak/ already exists at {oak_dir} (use --force to overwrite)")
        return oak_dir
    if force and oak_dir.exists():
        shutil.rmtree(oak_dir)

    # --- Required inputs ---
    # Depth comes either as a compact lossless video (oakd_depth.mkv, the format
    # the uploader now produces) or, for older recordings, a directory of
    # per-frame PNGs (oakd_depth/). Either is accepted.
    left_mp4 = ep_dir / "oakd_left.mp4"
    depth_dir = ep_dir / "oakd_depth"
    depth_mkv = ep_dir / "oakd_depth.mkv"
    left_ts_json = ep_dir / "oakd_left_timestamps.json"
    depth_ts_json = ep_dir / "oakd_depth_timestamps.json"
    imu_json = ep_dir / "oakd_imu.json"
    calib_src = ep_dir / "oakd_calib_offline.json"
    for p in (left_mp4, left_ts_json, depth_ts_json, imu_json, calib_src):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")
    if not depth_mkv.is_file() and not depth_dir.is_dir():
        raise FileNotFoundError(
            f"Missing depth: neither {depth_mkv} nor {depth_dir}")

    oak_dir.mkdir(parents=True)
    (oak_dir / "frames").mkdir()
    (oak_dir / "depth").mkdir()

    # --- Calib (already in expected schema) ---
    shutil.copyfile(calib_src, oak_dir / "calib_offline.json")

    # --- Build (seq, host_ms) pairs that exist in BOTH left and depth ---
    left_ts = json.loads(left_ts_json.read_text())["samples"]
    depth_ts = json.loads(depth_ts_json.read_text())["samples"]
    # device→host fit from the frame stream: lets the IMU below ride the same
    # host timeline as the frames (removes the false camera-IMU offset).
    dev_to_host_s = fit_device_to_host_s(left_ts)
    depth_seqs = {int(d["seq"]): d for d in depth_ts}
    matched = [
        (int(l["seq"]), float(l["host_ms"]))
        for l in left_ts if int(l["seq"]) in depth_seqs
    ]
    matched.sort(key=lambda x: x[0])  # ascending seq

    # --- Extract mp4 frames ---
    with tempfile.TemporaryDirectory() as tmp:
        tmp_frames = Path(tmp) / "frames"
        n_mp4 = _extract_mp4_frames(left_mp4, tmp_frames)

        # Trim to min of (matched pairs, decoded mp4 frames)
        n = min(len(matched), n_mp4)
        if n < len(matched):
            print(f"  WARNING: trimming {len(matched)-n} pairs (mp4 has {n_mp4} frames)")
        if n < n_mp4:
            print(f"  WARNING: ignoring {n_mp4-n} extra mp4 frames (no matching depth)")

        # Move to oak/frames/<idx>.png. shutil.move() (not Path.rename()) so
        # it works when tmp_frames is on tmpfs and oak_dir is on disk —
        # rename() raises EXDEV across filesystems.
        for idx in range(n):
            shutil.move(str(tmp_frames / f"{idx:06d}.png"),
                        str(oak_dir / "frames" / f"{idx:06d}.png"))

        # Resolve each depth seq to a source PNG. For the video format, decode
        # it once to temp PNGs; frame i ↔ depth_ts[i].seq (encode order). For
        # the legacy dir format, the PNG is named by seq.
        if depth_mkv.is_file():
            tmp_depth = Path(tmp) / "depth"
            _extract_depth_video(depth_mkv, tmp_depth)
            seq_to_png = {
                int(s["seq"]): tmp_depth / f"{i:06d}.png"
                for i, s in enumerate(depth_ts)
            }
        else:
            seq_to_png = {
                int(s["seq"]): depth_dir / f'{int(s["seq"]):08d}.png'
                for s in depth_ts
            }

        # Copy matched depth frames → oak/depth/<idx>.png
        for idx, (seq, _) in enumerate(matched[:n]):
            depth_png = seq_to_png.get(seq)
            if depth_png is None or not depth_png.exists():
                raise FileNotFoundError(f"depth frame missing for seq {seq}")
            shutil.copyfile(depth_png, oak_dir / "depth" / f"{idx:06d}.png")

        # --- Write timestamps.csv (idx, ns) ---
        with (oak_dir / "timestamps.csv").open("w") as f:
            f.write("idx,timestamp_ns\n")
            for idx, (_, host_ms) in enumerate(matched[:n]):
                f.write(f"{idx},{_ms_to_ns(host_ms)}\n")

    # --- Split IMU JSON → imu_acc.csv + imu_gyro.csv + imu_rotation.csv ---
    n_acc, n_gyr, n_rot = _split_imu_to_csvs(imu_json, oak_dir, dev_to_host_s)

    clk = "device→host fit" if dev_to_host_s is not None else "raw host_ms (no device_us)"
    print(f"  oak/ written: {n} frames, {n_acc} accel, {n_gyr} gyro, "
          f"{n_rot} rotation samples (IMU clock: {clk})")
    return oak_dir
