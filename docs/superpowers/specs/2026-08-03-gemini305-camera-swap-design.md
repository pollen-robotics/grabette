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
  new config field `depth_camera: Literal["oakd", "gemini305", "none"]`.
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
- **Encoding:** no on-device H.264. Reuse the existing lossless FFV1 path
  (`_pack_depth_video` is a ready-made pattern) for the Y8 left stream rather
  than contending with picamera2 for the Pi 4's hardware encoder. Revisit if
  file size hurts.
- **`calib_offline.json`:** fx/fy/cx/cy read from the device (the real values
  differ from the datasheet's nominal table — measured fx=622.79 vs 620 nominal
  at 1280×800, and cx/cy are not exactly W/2, H/2), plus `baseline = 0.018156`.
  Omit `imu_to_cam`.
- **Required fix:** `offline_vslam.cpp:118` does `auto& itc = calib["imu_to_cam"]`
  **unguarded**. An IMU-free calib JSON will crash there. Guard it and fall
  through to the existing no-IMU `setIMU(gyro, acc, ...)` branch at
  `offline_vslam.cpp:280`.
- Add `pyorbbecsdk2` to the `[rpi]` extra alongside `depthai` — both installed,
  one selected at runtime.

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
| `pyorbbecsdk2` on Pi 4 / Bookworm / Py 3.11 | **untested** — probe ran on x86_64 / Py 3.12 | Install spike on the Pi early; udev rules likely needed |
| Pi 4 CPU cost of encoding Y8 (was free in OAK-D hardware) | **unquantified** — 640×400 Y8 @30 ≈ 7.7 MB/s raw | Measure FFV1 encode load on-device; fall back to V4L2 M2M or lower fps |
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
