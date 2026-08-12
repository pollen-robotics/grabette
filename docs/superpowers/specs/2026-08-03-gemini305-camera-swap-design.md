# Replacing the OAK-D SR with an Orbbec Gemini 305

**Date:** 2026-08-03
**Status:** design reviewed; **Phase 0 executed and passed** — see [Phase 0 results](#phase-0-results-executed-2026-08-03)
**Driver:** vendor risk — OAK-D SR supply, cost, and Luxonis single-sourcing

## Problem

The Grabette rig's depth/IMU sensing is a Luxonis OAK-D SR driven through
`depthai`. That is a single-source dependency on the critical path of the whole
data-collection pipeline. We want a second, independently sourced option.

The Orbbec Gemini 305 is the closest available analog:

| | OAK-D SR | Gemini 305 |
|---|---|---|
| Size | 56 × 36 × 25.5 mm | 42 × 42 × 23 mm |
| Weight | 72 g | 68 g |
| Baseline | 20 mm | 18.156 mm (measured) |
| Ideal depth | 0.3 – 1 m | 0.07 – 0.5 m (op. 0.04 – 1 m) |
| Power | — | 1.57 – 1.88 W |
| Price | ~$320 | $229 |
| IMU | BNO086 @ 200 Hz | **none** |

Smaller, lighter, cheaper, lower-power, comparable optics — and no IMU.

## Reframe: pluggable, not replaced

Swapping one sole-source vendor for another relocates vendor risk rather than
reducing it. Separately, the SLAM pipeline has no ground truth, so the only way
to validate a new camera is to A/B it against OAK-D recordings — which requires
both cameras to remain supported.

**The deliverable is therefore a pluggable depth-camera interface with the
Gemini 305 as a second implementation, not a replacement of the OAK-D path.**
This costs essentially nothing extra (see Phase 1) and is strictly better.

## What was verified on hardware

A Gemini 305 (serial `CV275610003C`, firmware `1.0.70`, USB 3.2,
`OBDeviceType.LIGHT_BINOCULAR`) was probed directly with `pyorbbecsdk2==2.1.1`.
These are measurements, not datasheet claims:

| Question | Result |
|---|---|
| IMU present? | **No** — `ACCEL_SENSOR`/`GYRO_SENSOR` both absent from the sensor list |
| Sensors | `COLOR`, `DEPTH`, `LEFT_IR`, `RIGHT_IR` |
| Baseline | 18.1562 mm |
| Depth rectified? | **Yes** — `depth_distortion` k1…k6, p1, p2 all exactly 0 across all 121 calibration sets |
| Color rectified? | **No** — `rgb_distortion` k1=−1.171, k2=0.670, k3=−0.109, k4=−1.153, k5=0.644, k6=−0.099 |
| Depth + LEFT_IR concurrent? | **Yes**, both 848×530 (and 640×400), `enable_frame_sync()` OK |
| Depth ↔ IR sync | **0 µs** device-timestamp delta, identical frame index |
| 640×400 depth available? | **Yes**, 5/10/15/20/30 fps (13 depth modes total, not the 5 the datasheet lists) |
| Depth units | `depth_scale = 0.1` → **raw is 0.1 mm, not mm** |
| Invalid pixels | `0` (36.4% of a test frame) — same convention as OAK-D; `65535` also present at 0.5% |
| Clocks per frame | device `get_timestamp_us`, host `get_system_timestamp_us`, synced `get_global_timestamp_us` |
| Host clock sync | `Device.timer_sync_with_host()` present and succeeds |
| Clock offsets | global ≈ device + 2.1 ms; system ≈ device + 28 ms |

One incidental but useful observation: on a normal indoor scene the 305 returned
valid depth well beyond its rated 1 m operating range — median 1.78 m, p99 5.05 m,
max 6.48 m. Those far returns are presumably lower-confidence than the 7–50 cm
ideal band, but RGB-D odometry benefits from mid-range structure, so the "max
100 cm" figure appears to be a spec of *guaranteed accuracy*, not of the
usable range. Worth confirming against the OAK-D in the same-motion A/B.

Two consequences worth stating plainly:

1. **`get_global_timestamp_us()` is the correct clock**, not
   `get_system_timestamp_us()`. The ~28 ms system-vs-device gap is exactly the
   delivery-latency error `docs/data_format.md` already warns about ("stamping
   at delivery made host_ms ~100 ms late vs the Arducam's sensor-capture
   timeline"). This is the direct analog of depthai's `msg.getTimestamp()`.
2. **Depth must be scaled by `depth_scale` before writing.** Skip it and
   RTAB-Map reads every depth value 10× too far, with no error anywhere.

## Why the pipeline fits better than expected

| Need | OAK-D SR today | Gemini 305 |
|---|---|---|
| Depth reference frame | `setDepthAlign(CAM_B)` (left) | depth origin *is* the left module's optical center |
| Left grayscale + depth | `stereo.rectifiedLeft` + `depth` | `LEFT_IR` (Y8) + `Depth` (Y16), hardware-synced |
| Rectified, undistorted frames | `stereo.rectifiedLeft` | zero-distortion depth intrinsics — **no host undistortion** |
| Depth resolution | 640×400 | 640×400 available — **mask unchanged** |
| Right stream | recorded but **never consumed** (`checks/recording.py:255`) | irrelevant |
| SDK on Pi 4 arm64 | depthai wheels | `pyorbbecsdk2` aarch64 wheels, Py 3.9–3.13 |

The datasheet's "Dual Color *xor* Depth+Color" preset exclusivity is a non-issue:
we need only depth + left, which is the `Depth + Color` preset.

## Phases

### Phase 0 — IMU ablation (gate; no hardware, no code change)

The 305 has no IMU. Before committing to it, measure how much RTAB-Map actually
depends on the one we have. This is nearly free: `loadImu` returns cleanly on a
missing file and the IMU block is guarded at `offline_vslam.cpp:217`.

1. Take 3–5 OAK-D episodes that currently grade GOOD.
2. Run `run_oak_slam.py` twice each — baseline, then with `imu_acc.csv`,
   `imu_gyro.csv`, `imu_rotation.csv` withheld, into
   `--output_csv camera_trajectory_noimu.csv`.
3. Grade both with `check_trajectory.py`; compare drift / jump / zigzag metrics.

**Gate:** IMU-free grades hold → proceed. They collapse → stop, and revisit
either a discrete BNO085 on the Pi's I2C (same sensor family as the OAK-D's
BNO086, so `oakd_imu.json` stays byte-compatible and `imu_to_cam` comes from the
URDF) or the Gemini 336 (has a hardware-synced 6-DoF IMU, but 124×29×27 mm /
133 g / up to 6.5 W peak).

Scope limit, stated explicitly: this measures RTAB-Map's IMU dependence *on this
rig's motion profile*. It says nothing about the 305's depth quality. Those are
independent risks and Phase 0 retires only the first.

**Input, resolved:** `pollen-robotics/test_gripette_050526` turned out to be the
*converted* LeRobot v3.0 output (`data/`, `meta/`, `videos/`; features are only
`observation.images.cam0/cam1`, `action[8]`, and index bookkeeping) — no depth,
IMU, or calibration, so SLAM cannot be re-run from it. Four complete raw
episodes were found locally under `~/Downloads` instead and used for the run.

## Phase 0 results (executed 2026-08-03)

**Verdict: PASS.** Removing the IMU perturbs camera-local odometry no more than
re-running the identical pipeline twice does.

Run on four real episodes found locally (`~/Downloads`), 112–446 frames each
(4.6–15.8 s at 30 fps), converted with `convert_episode_to_oak.py` and processed
with the `pollenrobotics/oak-vslam` image built from
`docker/oak_vslam/Dockerfile` (not on Docker Hub; must be built).

### The control that mattered

A naive IMU-vs-no-IMU comparison would have been misleading. RTAB-Map's F2M
odometry is RANSAC-based and **this pipeline is not reproducible run to run**, so
the baseline was run twice to establish a noise floor:

| Episode | mean step | IMU vs IMU (noise) | IMU vs no-IMU (effect) | effect/noise |
|---|---|---|---|---|
| 20260615_130020 | 3.715 mm | 2.309 mm | 2.145 mm | **0.93** |
| 20260609_150333 | 0.398 mm | 0.929 mm | 0.692 mm | **0.74** |
| 20260609_150402 | 0.296 mm | 0.392 mm | 0.403 mm | **1.03** |
| 20260611_075114 | 4.527 mm | 3.367 mm | 3.202 mm | **0.95** |

RMSE of camera-local translation deltas (`R[t]ᵀ(p[t+1]−p[t])`), which are
invariant to any rigid rotation of the world frame and therefore isolate
odometry quality from the gravity convention. Every ratio is ≈ 1.0: the IMU's
effect is indistinguishable from run-to-run noise.

Supporting results:

- `check_trajectory` grades **GOOD → GOOD on all four**, with **zero** tracking
  losses introduced (100% tracking both ways).
- Raw world-frame RMSE looks large (up to 0.457 m) but collapses to 0.3–7.6 mm
  after removing a single rigid rotation of ~109–127°. That rotation *is* the
  gravity alignment; it accounts for ~98% of the apparent difference.

### What the IMU actually does here

Two things, both world-frame conventions rather than odometry quality:

1. RTAB-Map's gravity-aligned initial pose
   (`Odometry.cpp:328 … Updated initial pose … with IMU orientation`).
2. `oak_slam._gravity_align_trajectory`, a host-side rigid world rotation
   applied after SLAM — a second IMU dependency not identified in the original
   design, which logs `gravity-align skipped` when the CSVs are absent.

Losing both means each episode's world frame becomes arbitrary rather than Z-up.
That is **irrelevant to the DiffusionPolicy `--proprioception relative` path**:
`convert_dataset.py:263` computes pose relative to episode start *expressed in
the start camera frame*, which cancels any world rotation exactly. It would
matter for a consumer of the absolute 8-D state, where gravity alignment is what
makes Z consistent across episodes.

### Consequence for the IMU decision

The requirement collapses. `_estimate_gravity_imu` needs only **accelerometer**
samples, and only their median over near-static parts of the episode. So if a
Z-up world is wanted, it needs a cheap 3-axis accelerometer measuring a static
gravity direction — not a 200 Hz VIO-grade 6-DoF IMU. If only relative
proprioception is consumed, nothing is needed at all.

### Limits of this result

- Four episodes, all short (≤ 15.8 s). Drift accumulates with time; this cannot
  speak to minute-long recordings.
- All four were easy — 100% tracking with IMU, no fast motion or feature-poor
  stretches, which is exactly where an IMU normally earns its place.
- It says nothing about the 305's depth quality, which remains unretired.

### Incidental finding worth its own attention

The SLAM pipeline is **non-deterministic** at ~2–3 mm local-delta RMSE between
identical runs. Regenerating a dataset will not reproduce previous trajectories
bit-for-bit. That is a pre-existing property, unrelated to this work, but it
bears on dataset reproducibility and on any future A/B that compares single runs.

## Config A verified end-to-end on hardware (2026-08-03)

A 305 sequence was recorded straight into the `oak/` layout (LEFT_IR → `frames/`,
depth×0.1 → `depth/`, SDK intrinsics → `calib_offline.json`, no IMU CSVs) and run
through the production `offline_vslam` image and `check_trajectory` grader.

### Geometry: CONFIRMED

The design's core assumption was tested with a falsifiable stereo check —
predict disparity `d = fx·B/Z` from the depth value, then score the LEFT_IR patch
against RIGHT_IR at that disparity:

| Depth-scale hypothesis | mean NCC | fraction > 0.8 |
|---|---|---|
| `raw × 0.1` → mm (**the design**) | **0.8754** | 77% |
| `raw` unscaled (10×) | 0.0782 | 3% |
| `raw × 0.01` (0.1×) | 0.0125 | 0% |

Both sharpness sweeps peak exactly at zero and fall off monotonically —
disparity offset `0 → 0.875`, decaying to `0.117` at ±8 px; row offset
`0 → 0.875`, symmetric to `0.564` at ±4 px.

Therefore, on real hardware: `LEFT_IR` **is** the rectified left image, it is
row-rectified against `RIGHT_IR`, depth shares its pixel grid, and
`depth_scale = 0.1 → mm` is metrically consistent with `fx·B/Z`. **No host
undistortion is required; the Config B fallback is not needed.**

Also confirmed incidentally: Depth + LEFT_IR + RIGHT_IR stream together at
640×400 @30, and `offline_vslam` crashes without `imu_to_cam` exactly as
predicted — `nlohmann::detail::type_error.302 "type must be number, but is null"`
— so the guard at `offline_vslam.cpp:118` is confirmed necessary.

### Trajectory quality: GOOD at realistic pace

Three captures were needed, because the first two were invalid — the first moved
3–4× too fast at room scale, the second turned out to be stationary (frame 0 vs
frame 1349 differed no more than adjacent frames). Motion was verified from pixel
statistics before trusting any grade thereafter.

| Capture | verdict | trk % | dist | median step | max step | jumps |
|---|---|---|---|---|---|---|
| **305 realistic** | **GOOD** | 97.8 | 5.40 m | **3.84 mm** | 13.3 mm | **0** |
| 305 realistic, rerun | GOOD | 97.8 | 5.42 m | 3.87 mm | 12.4 mm | 0 |
| 305 fast, room-scale | BAD | 100.0 | 29.92 m | 14.61 mm | 76.7 mm | 19 |
| OAK-D 20260615 | GOOD | 100.0 | 1.14 m | 3.38 mm | 10.5 mm | 0 |
| OAK-D 20260611 | GOOD | 100.0 | 2.01 m | 4.54 mm | 12.8 mm | 0 |

The realistic capture hit the target pace — 3.84 mm/frame sits inside the OAK-D
band of 3.38–4.54 mm — at 0.33 m median depth with 65.5% valid pixels, and graded
**GOOD with zero jumps**, reproducibly across two runs.

Run-to-run local-delta RMSE was 2.184 mm on a 3.968 mm mean step, the same
magnitude as the OAK-D noise floor measured in Phase 0 (2.309 mm on 3.715 mm).
The pipeline's non-determinism is inherent, not camera-dependent.

### The one real deficit: 97.8% vs 100% tracking

The 30 lost frames are a **single contiguous 1-second burst** (frames 332–361),
identical in both runs. Diagnosing that window against its own baseline:

| | baseline | during burst |
|---|---|---|
| brightness | ~56–66 | **30.8** |
| sharpness (Laplacian var) | 159–280 | **63** (motion blur) |
| depth valid | 65–73% | **26.8%** |

The camera swung through a dark, low-texture region while moving fast enough to
blur, depth coverage collapsed, and RGB-D odometry dropped out until it
recovered unaided. This is exactly the predicted passive-stereo failure mode of a
`LIGHT_BINOCULAR` device with an IR-cut filter and no projector — an
illumination and texture limitation, not a defect in the integration.

Practical implication: the 305 will need adequate workspace lighting and textured
scenes. The OAK-D SR is also passive, so this is a difference of degree; whether
it is worse in practice needs the same-motion A/B on the real rig.

**Not claimed:** the start-to-end distances (305: 0.106 m over 5.40 m; OAK-D:
0.104 m over 1.14 m) are *not* comparable drift figures — none of these
recordings is a verified closed loop, so they measure where the operator happened
to stop, not accumulated error.

Depth coverage is strongly scene-dependent: 18.9% valid at room scale (2.37 m
median), 48.6% at 0.24 m, 65.5% at 0.33 m. The ideal 7–50 cm band is real.

### Phase 1 — pluggable interface

The 88 `oakd` references in `backend/rpi.py` are almost entirely lifecycle
plumbing (keepalive powerdown, teleop mutual exclusion, enable/disable). They
funnel through a 10-method surface:

```
init_device  shutdown  is_initialized  is_recording  wait_until_ready
start_recording  stop_recording  get_latest_imu  get_depth_jpeg  imu_sample_count
```

- Add `hardware/depth_camera.py` defining a `DepthCameraCapture` Protocol over
  exactly those 10 members. `OakdCapture` satisfies it **unchanged**.
- `backend/rpi.py` changes in one place: `_init_oakd()` selects the class from a
  new config field `depth_camera`.

**Built 2026-08-06.** `hardware/depth_camera.py` defines the Protocol over the
10 members; `OakdCapture` satisfies it unchanged. Config gained
`depth_camera: Literal["oakd", "gemini305"] = "oakd"` — **`"none"` was dropped
from the spec's proposed Literal** because `enable_oakd=False` already expresses
it, and two overlapping off-switches invite drift. `depth_camera` selects the
model; `enable_oakd` still controls power. Both branches were exercised: each
degrades to a logged warning and `_oakd = None` when its SDK is absent.
`tests/test_depth_camera_protocol.py` pins the contract, checking signatures
rather than just member names (isinstance against a runtime_checkable Protocol
would pass a `start_recording` with the wrong arity).
- `get_latest_imu()` returns `None` on the 305. The existing contract already
  permits this ("Returns None until both first accel and first gyro packets have
  arrived"); verify the dashboard and `/charts/*` paths tolerate a permanent
  `None` rather than a transient one.

**Keep the `oakd_*` filenames and the `/api/oakd` route.** `convert.py`,
`checks/*`, `dataset.py`, `trajectory.py`, the OpenArm integration, and every
existing HF dataset key on them. Renaming is a separate mechanical PR with a
compatibility shim, not part of a hardware spike.

### Phase 2 — `OrbbecCapture`

New `hardware/orbbec.py`, mirroring `OakdCapture`'s always-on-pipeline +
per-capture-writers structure.

- **Streams:** `Depth` (Y16) + `LEFT_IR` (Y8) at 640×400 @30, D2C **off**,
  `enable_frame_sync()` on. Verified concurrent with 0 µs skew.
- **Clock:** call `Device.timer_sync_with_host()` at init; stamp frames with
  `get_global_timestamp_us()` mapped onto the `SyncManager` timeline, mirroring
  `_capture_host_ms`. Emit the `oakd_clock_pairs.json` analog.
- **Depth normalisation** (both required, both silent if missed):
  - multiply raw by `depth_scale` (0.1) to get millimetres;
  - zero out values outside the valid range so `65535` does not become a
    phantom 6.5 m return — grabette's `wait_until_ready` gates on
    `(depth > 0).mean()` and the body mask multiplies by 0, so `0` must remain
    the sole invalid marker.
- **Encoding — the FFV1 suggestion above was wrong.** The left stream has two
  consumers: SLAM input (`convert.py` explodes `oakd_left.mp4` into
  `oak/frames/*.png`) and **a policy observation** (`dataset.py`:
  `observation.images.cam1`, "OAK left, resized to image_size like cam0"). The
  OAK-D encodes it on-ASIC at 8 Mbit/s: 640×400 Y8 raw is 7.7 MB/s (~460 MB/min),
  H.264 brings it to ~60 MB/min. Lossless FFV1 only manages ~2:1, i.e. ~230
  MB/min — 4× the SD-card *and* WiFi-upload cost, since the only consumer is the
  remote Space. And losslessness buys nothing anyone values: the stream is
  already lossy H.264 today, SLAM reads it as grayscale, and the dataset
  re-encodes it to 720×960 anyway. **The real question is narrower: can the Pi 4
  H.264-encode one extra 640×400@30 stream**, on the hardware encoder alongside
  picamera2's 1296×972@46, or in software? That is a measurement needing a Pi.
  Fallback if it cannot: the 305 emits MJPEG natively, but only on the *color*
  stream — which is not rectified, so that trades away Config A rather than
  dropping in.
- **`calib_offline.json`:** fx/fy/cx/cy read from the device (the real values
  differ from the datasheet's nominal table — measured fx=622.79 vs 620 nominal
  at 1280×800, and cx/cy are not exactly W/2, H/2), plus `baseline = 0.018156`.
  Omit `imu_to_cam`.
- **Required fix — DONE (2026-08-06).** `offline_vslam.cpp` dereferenced
  `calib["imu_to_cam"]` unguarded, aborting with `type_error.302` on any
  IMU-free calib. Now probed with `contains()`, falling back to identity (unused
  when the IMU buffers are empty). Verified: the same input that previously
  aborted now completes.
- Add `pyorbbecsdk2` to the `[rpi]` extra alongside `depthai` — both installed,
  one selected at runtime.

## On-device validation on the Pi (2026-08-10)

Run against `grabette-01` — Raspberry Pi 4 Model B Rev 1.5, Debian 13 (Trixie),
Python 3.13.5, aarch64, camera on USB 3.2 (896 mA), `grabette` service active
throughout. **Both remaining Pi-side unknowns are now retired.**

### SDK: installs cleanly, but needs a udev rule

`pyorbbecsdk2==2.1.1` installed from a real aarch64 wheel for Python 3.13 — no
source build. Import and device enumeration work immediately.

**But opening the device fails out of the box** with
`OBError: usbEnumerator openUsbDevice failed!`. The USB node is `crw-rw-r--
root root`, and the only rules present are `80-movidius.rules` (installed by
`make install-udev-oak`, for the OAK-D) and `99-rpi-keyboard.rules`. Being in
`plugdev`/`video` does not help.

The SDK ships the fix: `pyorbbecsdk/shared/99-obsensor-libusb.rules` contains an
explicit entry for our PID —
`ATTRS{idVendor}=="2bc5", ATTRS{idProduct}=="0840" ... SYMLINK+="Gemini_305"`.
Installing it gives `crw-rw-rw- root video` plus a `/dev/Gemini_305` symlink, and
the device opens.

**Action for Phase 2:** add a `make install-udev-orbbec` target mirroring
`install-udev-oak`, and wire it into `install-rpi`. Without it a fresh flash
cannot open the camera, and the failure message names neither udev nor
permissions.

### Streaming: full rate, no drops

640×400 @30 depth + LEFT_IR, 30 s sustained: **29.90 fps, ~0 dropped frames**
(device index span 1..902 for 901 delivered), 0 wait timeouts.

### Encoding: solved, and it does not need the hardware encoder

The open question was whether the Pi could absorb the H.264 encoding the OAK-D
used to do on its own ASIC. Measured on 300 real captured Y8 frames:

| encoder | realtime | CPU | MB/min |
|---|---|---|---|
| libx264 ultrafast, **1 thread** | **4.07×** | **111%** | **54.0** |
| libx264 ultrafast (default threads) | 6.48× | 193% | 54.0 |
| libx264 veryfast | 2.58× | 324% | 54.0 |
| h264_v4l2m2m (hardware) | — | — | rejects `gray`, wants `yuv420p` |
| FFV1 lossless | 6.38× | 268% | 162.0 |

**One core encodes at 4× realtime.** So the stream can go to software x264 on a
single core while picamera2 keeps the Pi 4's hardware V4L2 encoder for
`raw_video.mp4` — no contention, which was the specific worry. At 54 MB/min it
is also marginally *smaller* than the OAK-D's ~60 MB/min reference.

The hardware encoder refusing `gray` input is therefore moot; converting to
yuv420p just to use it would cost CPU for no benefit.

FFV1 measures 162 MB/min — 3× H.264, confirming that recommendation was wrong,
though the earlier ~230 MB/min estimate was pessimistic.

Thermals stayed healthy: 50.6 °C idle → 62.3 °C after encoding, `throttled=0x0`.

Caveat: measured with the `grabette` service active but not mid-recording. The
true worst case is capture + picamera2 encode + x264 concurrently. Given one
core at 111% out of four, and picamera2 using the hardware encoder rather than
CPU, the margin is large — but it is a margin, not a measurement.

### SDK gotcha worth knowing

`ob.Context().query_devices()` fails with
`NULL pointer passed for argument "deviceMgr"` — the temporary Context is
collected mid-call. Both the `Context` **and** the device list must be held in
named variables. `OrbbecCapture` should keep both as instance attributes.

### Deployment gap: the SLAM Space vendors its own copy

SLAM does not run on the Pi — episodes are uploaded and processed by the
`pollen-robotics/grabette-slam` Space (`grabette/slam.py`, a Docker-SDK Space).
That Space **vendors a full copy of `grabette-postprocess/`**, including its own
`docker/oak_vslam/offline_vslam.cpp`, `convert.py`, and `checks/*`.

Consequence, verified 2026-08-06 by fetching the Space's file: its copy still has
the unguarded `auto& itc = calib["imu_to_cam"];` (line 108, zero occurrences of
the `contains()` guard). **An IMU-free Gemini 305 episode pushed through
`/api/hf/slam` will abort in the Space with `type_error.302`**, even though this
repo is fixed. The fix has to be propagated to the Space repo and the Space
rebuilt before the 305 path works in production.

This is a standing hazard, not a one-off: any Phase 2 change to `convert.py` or
`checks/*` needed to handle a 305 episode must be propagated the same way. The
duplication is invisible from this repo — nothing here fails when the two drift.
Worth considering whether the Space should consume `grabette-postprocess` as a
dependency rather than a vendored copy, though that is out of scope here.

## Phase 2 results (implemented and validated 2026-08-10)

`hardware/orbbec.py` (`OrbbecCapture`) mirrors `OakdCapture`'s structure and
satisfies the same Protocol. Run on `grabette-01` against real hardware:

| check | result |
|---|---|
| 20 s recording | 600 frames, 30.02 fps |
| mp4 frames == left timestamps == depth timestamps | 600 / 600 / 600 |
| depth decoded from the FFV1 mkv | uint16 **mm**, median 1.047 m, max 6.519 m |
| invalid handling | no `65535` survives, nothing above the 10 m clamp |
| `get_latest_imu()` / `imu_sample_count` | `None` / `0` |
| `oakd_calib_offline.json` | fx/fy/cx/cy + baseline, **no `imu_to_cam`** |
| `wait_until_ready()` | True in 0.38 s |
| end-to-end convert → offline_vslam | `No imu_to_cam ... using identity`, 600/600 tracked, **GOOD** |

The end-to-end run was with the camera stationary on a desk (0.164 m over 20 s),
so it validates the *plumbing*, not trajectory quality — that was established
separately by the properly-paced workstation capture.

### Two further required fixes found while implementing

- **`convert.py` treated `oakd_imu.json` as mandatory** and raised
  `FileNotFoundError`, rejecting every 305 episode before SLAM could run. Now
  optional, printing `no IMU (IMU-less camera)` and skipping the CSV split.
  **This must also be propagated to the SLAM Space**, which vendors its own copy.
- **`make install-udev-orbbec`** added and wired into `install-rpi`. Both udev
  rules install unconditionally: `depth_camera` is a runtime setting, so gating
  provisioning on it would force a re-provision to switch cameras, and a rule for
  an absent device is inert.

### Storage: the one real regression

Measured over the same 20 s recording, versus OAK-D episodes packed identically:

| stream | Gemini 305 | OAK-D SR |
|---|---|---|
| left (H.264) | **57.3 MB/min** | ~60 MB/min |
| depth (FFV1) | **185.7 MB/min** | **40.8 MB/min** |
| total useful | **243 MB/min** | ~101 MB/min |

The left stream is fine — slightly *better* than the OAK-D. Depth is **4.5×
larger**, and the cause is entropy, not a bug: an OAK-D frame carries ~844 unique
depth values (median 341 mm, max 2976 mm), a 305 frame ~1469, with native 0.1 mm
precision and returns out to 6.5 m. FFV1 is lossless, so that noise is paid for
in full. On a battery Pi uploading over WiFi, 2.4× the bytes is a real cost.

Levers measured on 200 real frames:

| variant | MB/min | valid px |
|---|---|---|
| as recorded | 185.9 | 45.4% |
| clamp > 3 m | 132.5 | 34.7% |
| **clamp > 2 m** | **94.7** | 28.2% |
| clamp > 1 m | 64.8 | 23.2% |
| quantise 5 mm | 166.7 | 45.4% |
| quantise 10 mm | 144.5 | 45.4% |
| clamp 2 m + 5 mm | 76.0 | 28.2% |

Clamping the far field works; quantisation barely does. `_MAX_VALID_DEPTH_MM` is
currently 10000, i.e. effectively unclamped.

**Open decision, deliberately not taken here.** Clamping trades storage against
mid-range structure that RGB-D odometry may use, and this spec earlier argued
those far returns are useful. The 305 is only *rated* to 1 m, so a 2 m clamp
keeps margin while halving the cost — but that is a quality judgement, and it is
measurable: re-run SLAM on clamped depth and compare with
`compare_trajectories.py --noise`. Until measured, the conservative default
(no clamp) stands.

## Pre-merge TODO: generic camera naming in the dashboard

Requested 2026-08-12, to be done before this branch merges. The UI hardcodes
"OAK-D" in user-visible strings and should instead show whichever depth camera
is configured.

The clean seam is `app/routers/oakd.py::_status()`, which already returns
`supported/enabled/initialized/initializing` and is what the dashboard polls.
Adding a `model` field (`"oakd"` / `"gemini305"`) — or better, a display label —
lets the UI name itself with no new endpoint.

Known user-visible sites:

- `ui/app.py:264-272` — five `_badge("OAK-D", ...)` calls (N/A / Connected /
  Starting… / Error / Off).
- `ui/app.py:484` — toggle button appearance and the OAK data-row visibility.
- `app/routers/oakd.py:42,53` — `HTTPException(501, "backend has no OAK-D")`.
- `app/routers/camera.py:51,60` — docstrings for the depth endpoints (these
  surface in the OpenAPI schema).

**Not** in scope for that change, deliberately: the `/api/oakd/*` route paths and
the `oakd_*` output filenames. Renaming routes breaks the UI client and any
external caller; renaming files breaks convert.py, checks/*, dataset.py, the
OpenArm integration and every existing HuggingFace dataset. Both are separate
mechanical PRs with a compatibility shim, not part of a hardware swap.

## First real recordings exposed two bugs (2026-08-12)

Five episodes recorded through the daemon on `grabette-01` — the first time
`OrbbecCapture` ran under real concurrent load, with picamera2 encoding
`raw_video.mp4` at the same time. Both bugs were introduced by this work; the
standalone bench had passed at 30 fps because nothing else was running.

### Bug 1 — 71% of depth frames dropped

| | OAK-D reference | Orbbec, as first written |
|---|---|---|
| depth stream | 30.02 fps, seq span 307 = 307 frames, **0 dropped** | 8.61 fps, seq span 266 = 77 frames, **189 dropped (71%)** |
| arducam | 49.7 fps | 28.1 fps |

Causes, in order of cost:

1. A 640×400 uint16 **PNG encode per frame**, on the acquisition thread.
2. A float32 depth conversion allocating a 1 MB array and walking it four times,
   where integer `raw // 10` does the job in one pass.
3. `(mask > 0).astype(...)` recomputed **every frame** for a value that never
   changes.
4. Everything on a single thread, where `OakdCapture` splits video and depth.

Fixed by piping depth straight into a second ffmpeg FFV1 encoder instead of
writing PNGs (which also deletes the `_pack_depth_video` packing pass), the
integer conversion, and precomputing the mask.

### Bug 2 — timestamps on the wrong clock

`sync.monotonic_s_to_ms()` expects **CLOCK_MONOTONIC** — depthai's
`getTimestamp()` is monotonic, which is what it was written for. Orbbec's
`get_global_timestamp_us()` is **CLOCK_REALTIME (epoch)**. Feeding epoch in gave
`host_ms ≈ 1.79e12`; `camera_trajectory.csv` then showed `1786530000.0` on every
row, because ~1.79e9 exceeds float32's ~7 significant digits and every stamp
collapsed to one value. Episodes looked zero-length, so `check_trajectory`
reported "Unrealistic avg speed: 9.70 m/s (total 0.97m in 0.0s)".

Worse than the cosmetic grade: cross-stream alignment with the arducam and angle
data was silently wrong. Fixed by measuring `time.monotonic() - time.time()` once
at init and applying it per frame.

### Bug 3 — the always-on pipeline cooked the board

Even when *not* recording, the loop converted and masked every frame at 30 fps
just to keep a preview warm. Measured cost: **78.8 °C enabled-and-idle vs 60.8 °C
disabled**, which put the Pi into thermal throttling (`throttled=0xe0008`, soft
temp limit active) *before* a capture began. Idle now converts one frame in six
(~5 fps preview, `_IDLE_PREVIEW_EVERY`).

### Result

| | as first written | after fixes |
|---|---|---|
| depth/IR | 8.61 fps, 71% dropped | **29.63 fps, 1.3% dropped** |
| arducam | 28.1 fps | **40.0 fps** |
| idle temp, camera on | 78.8 °C | **55.5 °C** |
| thermal throttling | active | **none** |
| `host_ms` | ~1.79e12 (epoch) | 10.8 → 19896.4 ✓ |

End-to-end on a fixed episode: depth decodes as uint16 mm (median 0.308 m, no
`65535`), convert → SLAM **600/600 tracked, GOOD**.

Caveat: that episode was near-stationary (0.19 m, 0.3 mm/frame), so it validates
plumbing, not trajectory quality. The 1.3% residual drop is not yet explained —
the OAK-D reference dropped none.

### Bug 4 — check_dataset anchored on the IMU

`find_episodes(dataset_dir, anchor="oakd_imu.json")` made every 305 episode
invisible ("No episodes found"). Now anchors on `oakd_left.mp4`, the SLAM input.
That is the fourth file needing the IMU-optional treatment, after
`offline_vslam.cpp`, `convert.py` and `checks/recording.py` — all four vendored
by the SLAM Space.

## RESOLVED: daemon memory growth was glibc arena retention (2026-08-12)

**Symptom.** Daemon RSS climbed to 1.66 GB in normal use, and capture degraded
with it — from 1.3% of depth frames dropped on a fresh daemon to 41-49% on a
bloated one, at 15-17 fps instead of 29. A restart restored full performance.

**Ruled out first:** thermal (51 C, no active throttling), disk (22.1 MB/s
available against ~5.3 MB/s needed), and the dashboard preview streams — an A/B
had captures *with* two preview WebSockets doing better (26.4% dropped) than
without (48.8%), refuting that hypothesis outright.

**Root cause.** Not a leak. Cycling `OrbbecCapture` in a standalone process with
`tracemalloc` showed the **Python heap completely flat** — 4.2 MB across 8
cycles, gc objects +3, writer thread exiting cleanly every time — while RSS still
grew. `malloc_trim(0)` then handed back 6-8 MB per cycle and held RSS at a
plateau, which is the signature of memory freed by the application but retained
by the allocator.

The capture threads churn roughly 1 MB of numpy buffers per frame (a 512 KB
depth conversion plus a 512 KB mask multiply) at up to 30 fps. glibc keeps freed
memory in per-thread arenas rather than returning it, and on a 4-core Pi it will
open up to `8 * cores = 32` arenas of up to 64 MB each — about 2 GB of headroom,
which is why RSS could reach 1.66 GB without anything leaking.

**Fix.** `Environment=MALLOC_ARENA_MAX=2` in `systemd/grabette.service`.

**Verified** across a six-capture session on `grabette-01`:

| capture | RSS | fps | dropped |
|---|---|---|---|
| 1 | 306 MB | 29.09 | 3.0% |
| 2 | 307 MB | 29.19 | 2.7% |
| 3 | 310 MB | 29.12 | 3.0% |
| 4 | 310 MB | 29.28 | 2.5% |
| 5 | 317 MB | 28.95 | 3.6% |
| 6 | 313 MB | 29.20 | 2.7% |

RSS plateaus at ~310 MB and the drop rate holds at 2.5-3.6% instead of climbing.
Enable/disable cycling likewise flattens: 281 → 309 MB over six cycles, against
318 → 478 MB before.

**Note this was never Orbbec-specific.** The allocation churn comes from the
capture path generally, so the OAK-D almost certainly shares it — the Orbbec
merely adds a second encoder and a per-frame depth conversion, reaching the
cliff sooner. Worth confirming with `depth_camera=oakd`, but the fix is
camera-agnostic either way.

**Residual:** 2.5-3.6% of frames still drop, stably, where the OAK-D reference
dropped none. Stable rather than degrading, so it is a separate and much smaller
question.

## Clean result: 100% tracking on every episode (try5, 2026-08-12)

Six episodes, two batches of three, kitchen scene, freshly restarted daemon.
**6/6 GOOD, 100% tracking on every one — zero losses across 1,256 frames.**

| batch | tracking | median step | dropped |
|---|---|---|---|
| try5a | 100 / 100 / 100 % | 5.0, 7.0, 3.6 mm | 4.6% |
| try5b | 100 / 100 / 100 % | 3.6, 3.6, 4.5 mm | 4.2% |

Median step 3.6-7.0 mm brackets the OAK-D reference band of 3.4-4.5 mm, so the
motion is representative, and drop rates are matched between batches.

### The exposure experiment did not actually run

try5b was meant to test pinned exposure, and did not: the feature touches four
files and only `orbbec.py` had ever been scp'd to the Pi, which was still at
`4911872`. `GRABETTE_ORBBEC_IR_EXPOSURE_US` was silently ignored, because
pydantic-settings drops unknown env vars without complaint.

So try5b is a **second auto-exposure batch**, which makes the pair a
repeatability measurement rather than an A/B — and both hit 100%.

**The consequence is that the exposure knob is not needed.** Auto-exposure
reaches 100% tracking once the daemon is healthy and the scene is textured; the
motion blur seen in try4 was an artefact of that batch's 23% frame drops
stretching the interval between surviving frames, not of the exposure itself.
The setting stays in the codebase, off by default, for workspaces where
lighting cannot be improved. `grabette-01` has been reverted to AE.

### What each batch actually taught us

| batch | drops | scene | result | limiting factor |
|---|---|---|---|---|
| try1 | 71% | desk | 1 WARN / 4 BAD | frame drops + wrong clock |
| try2 | 41-49% | desk | 4 GOOD / 1 WARN | daemon RSS at 1.66 GB |
| try3 | 5.4% | desk | 5/5 GOOD, 90% tracked | dark objects starving depth (63% → 17%) |
| try4 | 23% | kitchen | 5/5 GOOD, 81% tracked | drops; depth stayed healthy at 68-75% |
| **try5** | **4.4%** | **kitchen** | **6/6 GOOD, 100% tracked** | **none observed** |

Two conditions have to hold together: a **freshly restarted daemon** (RSS growth
is still unattributed) and a **lit, textured workspace** (passive stereo, IR-cut
filter, no projector). With both, the Gemini 305 matches what the OAK-D produces
on this rig.

Still outstanding: the same-motion A/B against the OAK-D itself, which is the
only thing that can say whether the 305 is as good, not merely good.

### Phase 3 — mechanical, explicitly deferred

- New CAD mount (42×42×23 vs 56×36×25.5 mm) in Onshape.
- New URDF `oak_l` / `oak_r` link poses, feeding `frames.json` and
  `T_camera_in_oak_l`.
- `hardware/oakd_teleop.py` (`OakdTeleop`, its own depthai pipeline, mutually
  exclusive with recording) stays OAK-D-only unless teleop needs the 305.

## Risks

| Risk | Status | Mitigation |
|---|---|---|
| RTAB-Map degrades without IMU | **RETIRED** — effect/noise ≈ 1.0 on 4 episodes | Phase 0 ablation, passed |
| World frame no longer Z-up without IMU | real, but scoped | Irrelevant to relative proprioception; a cheap accelerometer suffices if absolute states are needed |
| Result may not hold on long or hard episodes | open | Re-run the ablation on a minute-long, fast-motion episode |
| `LEFT_IR` not pixel-exact with depth | **RETIRED** — stereo NCC test confirms rectification, row alignment and depth scale on hardware | n/a |
| 305 trajectory quality under realistic motion | **open — the main remaining unknown** | Capture at ~3–5 mm/frame, 1–2 m path, 15–40 cm from textured clutter |
| `pyorbbecsdk2` on the Pi | **RETIRED** — installs and streams at 29.90 fps on Pi 4 / Trixie / Py 3.13 | Needs a `make install-udev-orbbec` target |
| Pi 4 CPU cost of encoding Y8 | **RETIRED** — libx264 ultrafast does 4.07× realtime on one core, 54 MB/min | Software x264; picamera2 keeps the hardware encoder |
| 305 depth quality on the real rig vs OAK-D | unretired | Same-motion A/B once mounted |
| Passive stereo, IR-cut, no projector | inherent (`LIGHT_BINOCULAR`) | Needs ambient light and scene texture; OAK-D SR is also passive, so roughly comparable |

## Verification

There is no test suite in this repo, and unit tests for a USB camera driver
would have little value. The existing checks scripts are the harness, used as
acceptance gates:

1. `check_recording.py` — expected files present.
2. `check_sync.py` — cross-stream timestamp alignment within existing tolerance.
3. `check_trajectory.py` — GOOD/WARN on a 305 episode.
4. Same-motion A/B: one scripted trajectory recorded with each camera back to
   back, trajectories compared.

New unit-testable seams worth covering, since they are pure functions with
silent failure modes:

- depth raw → millimetre conversion and invalid-pixel normalisation;
- device → host timestamp mapping.

## Decisions taken

- Pluggable interface rather than in-place swap.
- Measure IMU-free SLAM quality before adding any IMU hardware. **Done — passed,
  so the 305 proceeds with no IMU hardware.**
- Any future IMU-vs-no-IMU comparison must include a same-config rerun as a
  noise floor; the pipeline is not deterministic.
- Keep `oakd_*` filenames and API routes in this work.
- 640×400 @30 depth, preserving the current resolution and body mask.
- `LEFT_IR` + `Depth` with D2C off, rather than Color + D2C + undistortion —
  now verified on hardware rather than inferred.
