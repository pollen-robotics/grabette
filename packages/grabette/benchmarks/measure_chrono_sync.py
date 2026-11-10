# /// script
# requires-python = ">=3.10"
# dependencies = ["rapidocr_onnxruntime>=1.3", "numpy"]
# ///
"""Measure stream and device synchronisation by filming an on-screen chronometer.

Bench: two (or more) grabettes record the same chronometer screen
(`HH:MM:SS.mmm`, e.g. an online stopwatch) with a synchronized group start.
Download the episodes (one folder per episode, a sub-folder per device, as the
dashboard's multi-device raw download makes them), then on a workstation:

    uv run benchmarks/measure_chrono_sync.py ~/Downloads/synchro-<...>/
    uv run benchmarks/measure_chrono_sync.py ~/Downloads/synchro-<...>/20261105_100911 --out /tmp/sync

`uv run` installs RapidOCR in a throw-away environment; nothing is added to the
project. The first run reads every frame (a few minutes for ten 30 s episodes);
OCR readings are cached in --out, so later runs only redo the analysis.

Method
------
The screen shows a new value only every ~57-73 ms, so one reading is too coarse
to date a frame. What is precise is a *transition*: the displayed value changes
from one frame to the next, so the moment value v appeared lies between the two
frames' timestamps. For each transition, chrono - local ≈ v - (t_i + t_i+1) / 2;
the median over all transitions of a stream gives that stream's offset from its
local clock (the episode's t=0) to the chronometer, to a few ms. The screen's
own display latency is the same for every camera filming it, so it cancels in
every comparison below.

Reported per episode:
  - RPi-Orbbec (per device): timestamp of the RPi camera minus timestamp of the
    Orbbec for the same chronometer instant. Checks that each frame of
    raw_video.mp4 got the right capture time (PR 239). Expected: a few ms,
    stable through the episode.
  - RPi-Orbbec drift (per device): spread of RPi-Orbbec over 5 s windows. A
    frame / timestamp shift in one stream (after a hole, or pictures written
    out of order) shows up as a ~20 ms step. The Pi's clock being adjusted by
    NTP moves both streams together and cancels here; it is kept in
    results.json as each stream's clock_drift_ms.
  - t=0 gap R-L: how much later device R started than device L, measured on
    the chronometer (once with the RPi camera, once with the Orbbec).
  - capture_started error: the gap predicted by metadata sync.capture_started_utc
    minus the measured gap. What aligning on capture_started_utc would get wrong.
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CHRONO = re.compile(r"(\d{2})\D?(\d{2})\D?(\d{2})\D?(\d{3})")
# A reading further than this from what the stream's clock predicts is a misread.
OUTLIER_MS = 250.0
# A crop whose mean grey change from the previous frame stays under this is the
# same displayed value: reuse the previous reading instead of running OCR.
SAME_CROP = 1.5
WINDOW_MS = 5000.0

STREAMS = {
    # name: (video file, timestamps file)
    "rgb": ("raw_video.mp4", "frame_timestamps.json"),
    "ir": ("dcam_left.mp4", "dcam_left_timestamps.json"),
}


def load_timestamps(path: Path) -> list[float] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):  # dcam_*_timestamps.json: {"samples": [{"host_ms": ...}]}
        return [s["host_ms"] for s in data.get("samples", [])]
    return [float(t) for t in data]


def parse_chrono(text: str) -> float | None:
    """'00:01:11.255' (any separators, as OCR reads them) -> ms."""
    m = CHRONO.search(text.replace(" ", ""))
    if not m:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    if mi > 59 or s > 59:
        return None
    return ((h * 60 + mi) * 60 + s) * 1000.0 + ms


def texts(result) -> list[str]:
    """Every string in a RapidOCR result, whatever its nesting."""
    out = []
    stack = [result]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    return out


class Reader:
    """Reads the chronometer frame after frame.

    Text detection costs ~0.6 s a frame on a CPU, recognition of a tight,
    upright crop ~8 ms. So the chronometer's outline is detected once, and each
    frame is warped to that outline and only recognised. Detection runs again
    only when a frame can't be read (the grabette moved, or a glare)."""

    REC_HEIGHT = 48  # the recogniser's input height

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 8))
        self.quad = None
        self.rotate = False
        self.detections = 0

    def locate(self, frame: np.ndarray) -> float | None:
        """Full detection: update the outline, return the value it reads."""
        self.detections += 1
        for rotate in ((self.rotate, not self.rotate)):
            img = cv2.rotate(frame, cv2.ROTATE_180) if rotate else frame
            result, _ = self.engine(img)
            for box, text, _score in result or []:
                v = parse_chrono(text)
                if v is not None:
                    tl, tr, br, bl = (np.array(p, dtype=np.float32) for p in box)
                    u, w = (tr - tl), (bl - tl)
                    # Pad: a little along the line, more across it.
                    self.quad = np.array([tl - 0.04 * u - 0.2 * w, tr + 0.04 * u - 0.2 * w,
                                          br + 0.04 * u + 0.2 * w, bl - 0.04 * u + 0.2 * w],
                                         dtype=np.float32)
                    self.rotate = rotate
                    return v
        return None

    def crop(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.rotate(frame, cv2.ROTATE_180) if self.rotate else frame
        tl, tr, _br, bl = self.quad
        width = max(8, int(self.REC_HEIGHT * np.linalg.norm(tr - tl) / max(1.0, np.linalg.norm(bl - tl))))
        dst = np.array([[0, 0], [width, 0], [width, self.REC_HEIGHT], [0, self.REC_HEIGHT]],
                       dtype=np.float32)
        m = cv2.getPerspectiveTransform(self.quad, dst)
        return cv2.warpPerspective(img, m, (width, self.REC_HEIGHT))

    def recognise(self, crop: np.ndarray) -> float | None:
        result, _ = self.engine(crop, use_det=False, use_cls=False, use_rec=True)
        for t in texts(result):
            v = parse_chrono(t)
            if v is not None:
                return v
        return None

    def read(self, frame: np.ndarray, crop: np.ndarray) -> float | None:
        v = self.recognise(crop)
        if v is None:  # dim infrared, glare: stretch the contrast
            grey = self.clahe.apply(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            v = self.recognise(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR))
        if v is None:  # moved? find the chronometer again
            v = self.locate(frame)
        return v


def read_video(reader: Reader, video: Path, n_frames: int) -> list[float | None]:
    """OCR value of every frame, in file order (frame i <-> timestamps[i]).

    Frames are decoded one by one, never resampled to a frame rate: the mp4's
    nominal fps is not the real one (see docs/data_format.md)."""
    cap = cv2.VideoCapture(str(video))
    values: list[float | None] = []
    prev, prev_v = None, None
    reader.quad, reader.detections = None, 0
    ok, f = cap.read()
    while ok:
        if reader.quad is None:
            v = reader.locate(f)
            prev = None
        else:
            crop = reader.crop(f)
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
            if prev is not None and prev.shape == grey.shape and np.abs(grey - prev).mean() < SAME_CROP:
                v = prev_v
            else:
                v = reader.read(f, crop)
            prev = grey if reader.quad is not None else None
        values.append(v)
        prev_v = v
        ok, f = cap.read()
    cap.release()
    if len(values) != n_frames:
        print(f"    ! {video.name}: {len(values)} frames decoded, {n_frames} timestamps",
              file=sys.stderr)
    if values and all(v is None for v in values):
        print(f"    ! {video.name}: chronometer not found", file=sys.stderr)
    print(f"    {video.name}: {sum(v is not None for v in values)}/{len(values)} frames read, "
          f"{reader.detections} detections", file=sys.stderr)
    return values


def stream_offset(ts: list[float], values: list[float | None]) -> dict | None:
    """chrono - local offset of one stream, from its value transitions."""
    n = min(len(ts), len(values))
    pairs = [(ts[i], values[i]) for i in range(n) if values[i] is not None]
    if len(pairs) < 10:
        return None
    rough = statistics.median(v - t for t, v in pairs)
    good = {i for i in range(n) if values[i] is not None
            and abs(values[i] - ts[i] - rough) < OUTLIER_MS}
    est = []  # (local time of the transition, offset)
    for i in range(n - 1):
        if i in good and i + 1 in good and values[i + 1] != values[i]:
            # Value values[i+1] appeared between frame i and frame i+1.
            if values[i + 1] < values[i]:
                continue  # a misread that slipped through
            mid = (ts[i] + ts[i + 1]) / 2
            est.append((mid, values[i + 1] - mid))
    if len(est) < 5:
        return None
    offsets = [o for _, o in est]
    off = statistics.median(offsets)
    windows = {}
    for t, o in est:
        windows.setdefault(int(t // WINDOW_MS), []).append(o)
    wmed = {k: statistics.median(w) for k, w in windows.items() if len(w) >= 5}
    q = statistics.quantiles(offsets, n=4) if len(offsets) >= 4 else [off, off, off]
    return {
        "offset_ms": off,
        "transitions": len(est),
        "iqr_ms": q[2] - q[0],
        # Includes the Pi's own clock being slewed (NTP): compare with
        # rpi_minus_orbbec_drift_ms, where the clock cancels out.
        "clock_drift_ms": (max(wmed.values()) - min(wmed.values())) if len(wmed) >= 2 else 0.0,
        "window_offsets_ms": {str(k * WINDOW_MS / 1000): v for k, v in sorted(wmed.items())},
        "misread": n - len(good),
        "frames": n,
    }


def pair_drift(rgb: dict | None, ir: dict | None) -> float | None:
    """Spread of RPi-Orbbec over 5 s windows. Both streams use the Pi's clock,
    so a clock adjustment cancels here; what is left is a frame/timestamp shift
    in one stream (e.g. after a hole, or pictures written out of order)."""
    if not rgb or not ir:
        return None
    a, b = rgb["window_offsets_ms"], ir["window_offsets_ms"]
    diffs = [b[k] - a[k] for k in a if k in b]
    return (max(diffs) - min(diffs)) if len(diffs) >= 2 else None


def capture_started(dev: Path) -> datetime | None:
    try:
        s = json.loads((dev / "metadata.json").read_text())["sync"]["capture_started_utc"]
        return datetime.fromisoformat(s)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def analyse_device(reader: Reader, dev: Path, cache_dir: Path) -> dict:
    out = {"device_id": None, "capture_started": capture_started(dev)}
    try:
        out["device_id"] = json.loads((dev / "metadata.json").read_text()).get("device_id")
    except (OSError, ValueError):
        pass
    for name, (video, tsfile) in STREAMS.items():
        ts = load_timestamps(dev / tsfile)
        if ts is None or not (dev / video).exists():
            continue
        cache = cache_dir / f"{dev.parent.name}_{dev.name}_{name}.ocr.json"
        if cache.exists():
            values = json.loads(cache.read_text())
        else:
            print(f"  OCR {dev.parent.name}/{dev.name}/{video} ({len(ts)} frames)…", file=sys.stderr)
            values = read_video(reader, dev / video, len(ts))
            cache.write_text(json.dumps(values))
        out[name] = stream_offset(ts, values)
    return out


def find_episodes(paths: list[Path]) -> list[Path]:
    """Episode folders: those holding one sub-folder per device."""
    eps = []
    for p in paths:
        cands = [p] + sorted(x for x in p.iterdir() if x.is_dir())
        for c in cands:
            devs = [d for d in c.iterdir() if d.is_dir() and (d / "raw_video.mp4").exists()]
            if devs:
                eps.append(c)
    return eps


def fmt(v, digits=1, signed=True):
    if v is None:
        return "—"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="+", type=Path,
                   help="session folders (episodes inside) or episode folders")
    p.add_argument("--left", default="left", help="sub-folder of the reference device")
    p.add_argument("--out", type=Path, help="folder for the OCR cache and results.json "
                   "(default: <first path>/chrono_sync)")
    args = p.parse_args()

    out_dir = args.out or (args.paths[0] / "chrono_sync")
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = find_episodes(args.paths)
    if not episodes:
        print("No episode found (expected <episode>/<device>/raw_video.mp4)", file=sys.stderr)
        return 1

    reader = Reader()
    results = []
    for ep in episodes:
        devs = {d.name: analyse_device(reader, d, out_dir)
                for d in sorted(ep.iterdir()) if d.is_dir() and (d / "raw_video.mp4").exists()}
        row = {"episode": ep.name, "devices": {}}
        for name, d in devs.items():
            rgb, ir = d.get("rgb"), d.get("ir")
            row["devices"][name] = {
                "device_id": d["device_id"],
                "rgb": rgb, "ir": ir,
                # Same chrono instant: RPi timestamp - Orbbec timestamp.
                "rpi_minus_orbbec_ms": (ir["offset_ms"] - rgb["offset_ms"]) if rgb and ir else None,
                "rpi_minus_orbbec_drift_ms": pair_drift(rgb, ir),
            }
        ref = devs.get(args.left)
        for name, d in devs.items():
            if name == args.left or ref is None:
                continue
            gap = {}
            for s in ("rgb", "ir"):
                if d.get(s) and ref.get(s):
                    gap[s] = d[s]["offset_ms"] - ref[s]["offset_ms"]
            predicted = None
            if d["capture_started"] and ref["capture_started"]:
                predicted = (d["capture_started"] - ref["capture_started"]).total_seconds() * 1000
            measured = gap.get("rgb", gap.get("ir"))
            row["devices"][name]["t0_gap_vs_" + args.left] = {
                "rgb_ms": gap.get("rgb"), "ir_ms": gap.get("ir"),
                "capture_started_predicted_ms": predicted,
                "capture_started_error_ms": (predicted - measured)
                if predicted is not None and measured is not None else None,
            }
        results.append(row)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # Summary table, one line per episode and non-reference device.
    print(f"{'episode':16s} {'device':6s} {'RPi-Orbbec L':>13s} {'RPi-Orbbec':>11s} "
          f"{'t0 gap rgb':>11s} {'t0 gap ir':>10s} {'cs error':>9s} "
          f"{'drift L/dev':>12s} {'transitions':>12s} {'misread':>8s}")
    for row in results:
        devs = row["devices"]
        ref = devs.get(args.left, {})
        for name, d in devs.items():
            if name == args.left:
                continue
            g = d.get("t0_gap_vs_" + args.left, {})
            dr = lambda x: fmt(x.get("rpi_minus_orbbec_drift_ms"), 1, False)
            tr = lambda x: (x.get("rgb") or {}).get("transitions", 0) + (x.get("ir") or {}).get("transitions", 0)
            mr = lambda x: (x.get("rgb") or {}).get("misread", 0) + (x.get("ir") or {}).get("misread", 0)
            print(f"{row['episode']:16s} {name:6s} {fmt(ref.get('rpi_minus_orbbec_ms')):>13s} "
                  f"{fmt(d.get('rpi_minus_orbbec_ms')):>11s} {fmt(g.get('rgb_ms')):>11s} "
                  f"{fmt(g.get('ir_ms')):>10s} {fmt(g.get('capture_started_error_ms')):>9s} "
                  f"{dr(ref) + ' / ' + dr(d):>12s} {tr(ref) + tr(d):>12d} {mr(ref) + mr(d):>8d}")
    print(f"\nDetail: {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
