"""Bench the UVC stereo camera + hat BMI088 on the Pi before integrating them.

Answers four questions, each printed as OK / WARN / FAIL:
  1. Which /dev/i2c-N has the BMI088? (read-only chip-id probe)
  2. Does the camera hold its fps in MJPG passthrough (no decode, no encode)?
  3. Is CAP_PROP_POS_MSEC a kernel buffer timestamp on CLOCK_MONOTONIC?
     If so, frames can be dated at capture instead of at read() return.
  4. Can the SD card absorb the raw MJPG rate, with the IMU still at 200 Hz?

Run on the Pi (grabette service stopped, so nothing else holds the camera/I2C):
    uv run python scripts/stereo_bench.py --profile skl2mp220 --mode 3040x1080
Writes stereo.mkv, timestamps.json, left.png, right.png to --out.
"""

import argparse
import json
import logging
import resource
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

# Card-name substring (from /sys/class/video4linux/*/name) and eye order, from
# stereokit/profiles.py. The SKL sends [right|left].
PROFILES = {
    "skl2mp220": {"card_match": "SPCA2100", "swap_eyes": True},
    "ar0144": {"card_match": "CCB Camera", "swap_eyes": False},
}

BMI088_ACCEL_ADDRS = (0x18, 0x19)  # chip id 0x1E at reg 0x00
BMI088_GYRO_ADDRS = (0x68, 0x69)   # chip id 0x0F at reg 0x00


def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def section(name):
    print(f"\n=== {name} ===")


# --- 1. I2C probe ------------------------------------------------------------

def read_reg(i2c, addr, reg):
    buf = bytearray(1)
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto_then_readfrom(addr, bytes([reg]), buf)
    finally:
        i2c.unlock()
    return buf[0]


def probe_bmi088():
    """Return (bus, accel_addr, gyro_addr) of the first BMI088 found, or None.

    Only reads register 0x00 at the four possible BMI088 addresses, so it is
    harmless to other devices on the bus (e.g. the audio codec at 0x18 on
    i2c-1 just returns its page register).
    """
    from adafruit_extended_bus import ExtendedI2C

    section("BMI088 I2C probe")
    buses = sorted(int(p.name.removeprefix("i2c-")) for p in Path("/dev").glob("i2c-*"))
    found = None
    for bus in buses:
        try:
            i2c = ExtendedI2C(bus)
        except Exception as e:
            warn(f"i2c-{bus}: cannot open ({e})")
            continue
        accel = gyro = None
        for addr in BMI088_ACCEL_ADDRS:
            try:
                read_reg(i2c, addr, 0x00)  # accel needs a dummy read after power-up
                if read_reg(i2c, addr, 0x00) == 0x1E:
                    accel = addr
            except OSError:
                pass
        for addr in BMI088_GYRO_ADDRS:
            try:
                if read_reg(i2c, addr, 0x00) == 0x0F:
                    gyro = addr
            except OSError:
                pass
        i2c.deinit()
        if accel and gyro:
            ok(f"i2c-{bus}: BMI088 accel=0x{accel:02X} gyro=0x{gyro:02X}")
            found = found or (bus, accel, gyro)
        elif accel or gyro:
            warn(f"i2c-{bus}: partial match accel={accel} gyro={gyro}")
        else:
            print(f"         i2c-{bus}: no BMI088")
    if not found:
        fail("no BMI088 found on any bus")
    return found


# --- 2-4. Camera + IMU recording ---------------------------------------------

def find_video_node(card_match):
    for p in sorted(Path("/sys/class/video4linux").glob("video*"),
                    key=lambda q: int(q.name.removeprefix("video"))):
        name = (p / "name").read_text().strip()
        if card_match in name:
            # A UVC camera exposes a capture node and a metadata node with the
            # same card name; the capture node enumerates first.
            return int(p.name.removeprefix("video")), name
    return None


def open_camera(index, width, height, fps, exposure):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open /dev/video{index}")
    # FOURCC before size, or V4L2 ignores it.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Hand back the raw MJPG buffer: no decode on the Pi.
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    # Manual exposure in 100 us units. Auto on the SKL at 3040x1080 picks
    # ~60 ms and throttles 30 fps to 16.6.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


def record(cap, out_dir, duration_s, imu):
    """Pipe raw MJPG into ffmpeg (-c copy) for duration_s; return per-frame records."""
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "mjpeg", "-i", "pipe:0", "-c", "copy", str(out_dir / "stereo.mkv")],
        stdin=subprocess.PIPE,
    )
    frames = []
    write_s = 0.0
    first_jpeg = None
    if imu:
        imu.start_capture()
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        ok_, buf = cap.read()
        host_s = time.monotonic()
        if not ok_:
            frames.append({"ok": False, "host_s": host_s})
            continue
        data = buf.tobytes()
        if first_jpeg is None:
            first_jpeg = data
        t0 = time.monotonic()
        ffmpeg.stdin.write(data)
        write_s += time.monotonic() - t0
        frames.append({"ok": True, "host_s": host_s,
                       "pos_ms": cap.get(cv2.CAP_PROP_POS_MSEC), "bytes": len(data)})
    imu_samples = imu.stop() if imu else None
    ffmpeg.stdin.close()
    ffmpeg.wait()
    return frames, write_s, first_jpeg, imu_samples


def report_camera(frames, write_s, duration_s, fps):
    section("Camera")
    good = [f for f in frames if f["ok"]]
    bad = len(frames) - len(good)
    rate = len(good) / duration_s
    (ok if rate > 0.97 * fps else fail)(f"{len(good)} frames in {duration_s:.0f} s = {rate:.2f} fps (target {fps})")
    if bad:
        warn(f"{bad} failed read() calls")

    mb_s = sum(f["bytes"] for f in good) / duration_s / 1e6
    print(f"         MJPG rate {mb_s:.1f} MB/s, mean frame {statistics.mean(f['bytes'] for f in good) / 1e3:.0f} kB")
    # Time blocked writing to ffmpeg; near duration_s means the disk is the bottleneck.
    (ok if write_s < 0.3 * duration_s else warn)(f"blocked {write_s:.1f} s of {duration_s:.0f} s writing to ffmpeg")

    section("Frame timestamps (CAP_PROP_POS_MSEC)")
    pos = np.array([f["pos_ms"] for f in good])
    host = np.array([f["host_s"] for f in good]) * 1000.0
    if not pos.any():
        fail("POS_MSEC is always 0: no kernel timestamps, must fall back to host time")
        return
    period = 1000.0 / fps
    d = np.diff(pos)
    drops = int((d > 1.5 * period).sum())
    print(f"         interval median {np.median(d):.2f} ms, min {d.min():.2f}, max {d.max():.2f} (period {period:.2f})")
    (ok if drops == 0 else warn)(f"{drops} gaps > 1.5 periods (dropped frames)")
    # Same clock => host_ms - pos_ms is a small positive, steady latency.
    lat = host - pos
    print(f"         read latency host-pos: mean {lat.mean():.1f} ms, std {lat.std():.1f}, min {lat.min():.1f}, max {lat.max():.1f}")
    boot_minus_mono = (time.clock_gettime(getattr(time, "CLOCK_BOOTTIME", 7))
                       - time.monotonic()) * 1000.0
    if 0 <= lat.min() and lat.max() < 500:
        ok("POS_MSEC is on CLOCK_MONOTONIC")
    elif 0 <= (lat + boot_minus_mono).min() and (lat + boot_minus_mono).max() < 500:
        warn(f"POS_MSEC looks like CLOCK_BOOTTIME (offset {boot_minus_mono:.1f} ms)")
    else:
        fail("POS_MSEC is on an unknown clock")
    # Jitter of the kernel stamps vs the host stamps: the reason to prefer them.
    print(f"         interval std: kernel {d.std():.2f} ms vs host {np.diff(host).std():.2f} ms")


def report_imu(samples, duration_s):
    section("IMU (BMI088)")
    if samples is None:
        warn("skipped (no BMI088)")
        return
    cts = np.array([s["cts"] for s in samples.accel])
    if len(cts) < 2:
        fail(f"{len(cts)} samples")
        return
    rate = len(cts) / duration_s
    (ok if rate > 190 else fail)(f"{len(cts)} samples = {rate:.1f} Hz (target 200)")
    gaps = np.diff(cts)
    (ok if gaps.max() < 15 else warn)(f"max gap {gaps.max():.1f} ms, median {np.median(gaps):.2f} ms")
    acc = np.array([s["value"] for s in samples.accel])
    print(f"         mean accel {acc.mean(axis=0).round(2).tolist()} (|g| {np.linalg.norm(acc.mean(axis=0)):.2f})")


def save_eyes(jpeg, swap, out_dir):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    mid = img.shape[1] // 2
    left, right = img[:, :mid], img[:, mid:]
    if swap:
        left, right = right, left
    cv2.imwrite(str(out_dir / "left.png"), left)
    cv2.imwrite(str(out_dir / "right.png"), right)
    print(f"         decoded {img.shape[1]}x{img.shape[0]}; check left.png really is the LEFT eye")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=PROFILES, default="skl2mp220")
    ap.add_argument("--mode", default="3040x1080", help="combined WxH")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--exposure", type=int, default=300, help="100 us units (300 = 30 ms)")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds")
    ap.add_argument("--out", type=Path, default=Path.home() / "grabette-data" / "stereo_bench")
    ap.add_argument("--no-imu", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="         %(message)s")  # shows IMU drift factor
    args.out.mkdir(parents=True, exist_ok=True)
    width, height = map(int, args.mode.split("x"))
    prof = PROFILES[args.profile]

    imu = None
    if not args.no_imu:
        found = probe_bmi088()
        if found:
            from grabette.hardware.imu import BMI088Capture
            from grabette.hardware.sync import SyncManager
            sync = SyncManager()
            sync.start()
            bus, accel, gyro = found
            imu = BMI088Capture(sync, sample_rate_hz=200, i2c_bus=bus, accel_addr=accel, gyro_addr=gyro)
            imu.init_sensor()

    section("Camera open")
    node = find_video_node(prof["card_match"])
    if node is None:
        fail(f"no video node with card name containing {prof['card_match']!r}")
        return
    index, name = node
    cap = open_camera(index, width, height, args.fps, args.exposure)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    (ok if got == (width, height) else fail)(f"/dev/video{index} ({name}) negotiated {got[0]}x{got[1]}")
    for _ in range(30):  # let the stream and exposure settle
        cap.read()

    print(f"\nRecording {args.duration:.0f} s ...")
    t0 = resource.getrusage(resource.RUSAGE_SELF)
    try:
        frames, write_s, first_jpeg, imu_samples = record(cap, args.out, args.duration, imu)
    finally:
        # UVC controls persist on the device: give auto exposure back.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.release()
    t1 = resource.getrusage(resource.RUSAGE_SELF)

    report_camera(frames, write_s, args.duration, args.fps)
    report_imu(imu_samples, args.duration)

    section("Output")
    cpu = (t1.ru_utime - t0.ru_utime + t1.ru_stime - t0.ru_stime) / args.duration * 100
    print(f"         python process CPU {cpu:.0f}% of one core")
    if first_jpeg:
        save_eyes(first_jpeg, prof["swap_eyes"], args.out)
    (args.out / "timestamps.json").write_text(json.dumps({"frames": frames}))
    print(f"         wrote {args.out}/stereo.mkv, timestamps.json, left.png, right.png")


if __name__ == "__main__":
    main()
