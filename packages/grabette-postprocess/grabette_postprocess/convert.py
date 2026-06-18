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

Timestamps in the output CSVs are in nanoseconds in the SyncManager (host_ms)
clock — same convention as my colleague's capture pipeline.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def _ms_to_ns(ms: float) -> int:
    return int(round(ms * 1e6))


def _extract_mp4_frames(mp4_path: Path, out_dir: Path) -> int:
    """Decode mp4 to GRAY8 6-digit PNGs starting at 000000.png. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mp4_path),
        "-pix_fmt", "gray",
        "-start_number", "0",
        str(out_dir / "%06d.png"),
    ]
    subprocess.run(cmd, check=True)
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
        "-start_number", "0",
        str(out_dir / "%06d.png"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))


def _split_imu_to_csvs(imu_json: Path, oak_dir: Path) -> tuple[int, int, int]:
    """Split oakd_imu.json into the three CSVs offline_vslam reads
    (imu_acc.csv, imu_gyro.csv, imu_rotation.csv). Returns (n_accel, n_gyro, n_rot).

    rotation is the BNO086's fused orientation (IMU body → gravity-aligned world).
    RTAB-Map's Odometry can take this as orientation per IMU sample and use it for
    VIO initialization + drift constraint during rotations.
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
    n_acc, n_gyr, n_rot = _split_imu_to_csvs(imu_json, oak_dir)

    print(f"  oak/ written: {n} frames, {n_acc} accel, {n_gyr} gyro, {n_rot} rotation samples")
    return oak_dir
