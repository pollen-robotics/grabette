# Bi-manual Data Fusion via Casquette — Design Analysis

**Status:** Open design — not yet implemented. This document captures the analysis from the SLAM bring-up sessions (May 2026) and frames the open questions to resolve before committing.

## 1. Goal

Collect bi-manual manipulation demonstrations using **two Grabettes simultaneously**, with both gripper-camera trajectories expressed in a common spatial reference frame so a single policy (or coordinated policies) can be trained on the data.

Constraint: the user does NOT want to rely on a workspace-tape ArUco marker like the original UMI does. The casquette (head-mounted POV camera) is the chosen alternative.

## 2. Why this is non-trivial

A single Grabette's SLAM trajectory is in its own SLAM-internal world frame, established by whatever orientation the device had at recording start. Two Grabettes recording simultaneously have **two independent SLAM worlds** that are unrelated until you provide some cross-observation. Concretely:

```
Grabette A: pose_A(t) in frame W_A   ← rtabmap world for device A
Grabette B: pose_B(t) in frame W_B   ← rtabmap world for device B

No information about how W_A and W_B are related → cannot compute
the inter-gripper relative pose pose_A^{-1} · pose_B without an
external reference.
```

For training a bi-manual policy, the inter-gripper relative pose is what the model needs to coordinate the two arms. (Each gripper's *own* camera-local deltas are fine; see `[grabette/lerobot] project_grabette_design.md §10` for why the deltas live in camera-local frame.)

## 3. Reference architecture: Casquette + ArUco-on-Grabettes

The user's plan, summarized:

```
                                      Casquette
                                  (head-mounted POV)
                                          │
                          camera sees ArUco markers on each grabette
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
            Grabette A                                      Grabette B
            (own SLAM via OAK)                              (own SLAM via OAK)
            marker on body                                  marker on body
```

The casquette plays the role of **UMI's table tag** — except it moves with the user's head, so it must also have its own pose-tracking to remain a usable spatial reference.

### Fusion math (simplified)

For each timestep `t` where the casquette observes grabette A's marker:

```
tx_casqW_grabAW(t)  =  tx_casqW_casqCam(t)  ·  tx_casqCam_grabA(t)  ·  inv(tx_grabAW_grabA(t))
                       └──────┬─────────┘    └────────┬──────────┘    └────────┬──────────┘
                       from casquette SLAM     from ArUco detection    from grabette A SLAM
```

This produces a **fixed 6-DoF offset** between the two SLAM worlds (`tx_casqW_grabAW`). Once established, both grabettes' trajectories can be expressed in the casquette's SLAM world:

```
pose_grabA_in_casqW(t)  =  tx_casqW_grabAW  ·  pose_grabA(t)
pose_grabB_in_casqW(t)  =  tx_casqW_grabBW  ·  pose_grabB(t)
```

And the inter-gripper relative pose is:

```
tx_grabA_grabB(t)  =  inv(pose_grabA_in_casqW(t))  ·  pose_grabB_in_casqW(t)
```

Three SLAM trajectories (casquette, grabette A, grabette B) plus per-observation ArUco detections from the casquette's camera give us everything we need.

## 4. ArUco visibility requirements

| Regime | What you get | Caveats |
|---|---|---|
| **Once per demo, at the start** | Fixed inter-SLAM-world offset valid for that demo. Each grabette's own SLAM provides ego-motion in its own frame, transformed via the constant offset. | Relies on each grabette's SLAM not drifting during the demo. For ~30s demos, rtabmap VIO drift is typically <1 cm, <1° — fine. |
| **Periodically (a few times per demo)** | Offset re-anchored, cancelling small drift. Most robust against per-device tracking glitches. | Requires the user to "look at" each grabette occasionally. Reasonable for some tasks, unnatural for others. |
| **Continuously** | Direct measurement at every timestep. No reliance on per-device SLAM drift behavior. | Often blocked by the user's own arms or grabette body orientation. Not realistic to maintain. |

**Recommendation:**
- **Minimum**: each grabette's marker visible for ~1–2 s at the **start** of every demo. The user wears the casquette and looks briefly at each grabette before pressing the start button. This gives multiple noise-averaged ArUco detections to establish a low-noise `tx_casqW_grabAW`.
- **Better**: visible a few times throughout the demo, so SLAM drift gets caught.
- **Multiple markers per grabette** (top + each side) so at least one is in view regardless of orientation. UMI puts two tags on each gripper for this reason.

## 5. Open question: Casquette SLAM capability

The casquette runs on a **Pi Zero 2W** (quad-core Cortex-A53 @ 1 GHz, 512 MB RAM) — much weaker than the Pi 4 + OAK-D setup we use on Grabette. Historical context:

- We previously tried ORB-SLAM3 on this hardware class (RPi 4 + RPi camera + IMU) and got 93–98% tracking when everything was tuned, but it proved **not robust enough** under typical conditions (frame drops, IMU drift, lighting changes). That fragility is why Grabette V2 switched to the OAK-D + rtabmap stack.
- The casquette has even less compute (Pi Zero 2W < Pi 4) and uses the same RPi-camera-class lens + BMI088 IMU as the old V1 Grabette.

**Live on-device SLAM on the casquette is almost certainly unfeasible.** ORB-SLAM3 typically wants 1–2 GB RAM and meaningful CPU; the Pi Zero 2W has 512 MB total. Tracking even at 240p would be marginal.

**Offline SLAM (post-process on workstation) is fine.** Each casquette session records `raw_video.mp4` + `imu_data.json` + `metadata.json` (already designed for this); we run SLAM on the workstation after the fact. This matches the Grabette OAK-vslam Docker offline pipeline.

So the question becomes:

### Do we even need casquette SLAM?

It depends on how we use the casquette's observations:

#### Option A — Casquette has its own SLAM (current plan)

Casquette pose tracked over time via its own VIO. ArUco observations of the grabettes give cross-world offsets as described in §3.

- ✅ Most general: doesn't constrain head movement
- ✅ Works even when only ONE grabette is in view at a time
- ❌ Requires the casquette VIO to be robust (the very thing that drove us off RPi+ORBSLAM3 historically)
- ⚠ Might be salvageable with rtabmap (different algorithm) or with looser robustness requirements (offline only, less time pressure than online), but **needs experimental validation**

#### Option B — Casquette as a "shared-view" device only (no own SLAM)

Casquette is just a multi-marker observer. Only timestamps where it sees **both grabettes' markers in the same frame** are usable — those give a direct measurement of `tx_grabA_grabB` (independent of any SLAM):

```
tx_grabA_grabB(t)  =  inv(tx_casqCam_grabA(t))  ·  tx_casqCam_grabB(t)
```

The casquette's own motion cancels out because both observations share the same frame.

- ✅ No casquette SLAM needed at all → side-steps the hardware-too-weak question
- ✅ Direct, low-noise inter-gripper relative pose
- ❌ Requires both grabettes to be **simultaneously visible** in the casquette's field of view. Hard during typical manipulation where one hand crosses behind the other or one is held above the workspace.
- ❌ Multiple "snapshots" per demo (when both are visible) are needed; in between, the per-grabette ego-SLAMs interpolate — but they must be aligned, which is exactly the problem we're trying to solve.

In practice Option B alone is brittle. It can serve as a **calibration anchor at the start of a session** (look at both grabettes simultaneously for a few seconds to nail down `tx_grabA_grabB(t=0)`), then per-grabette SLAMs handle ego-motion from there. This degrades gracefully to a "once at start" regime.

#### Option C — Joint multi-agent SLAM (the ambitious path)

Treat all three devices as a **single multi-camera SLAM system** with cross-device factors from ArUco observations. Formulated as a factor graph (e.g., GTSAM):

- Nodes: each device's pose at each keyframe (so 3 trajectories' worth of nodes)
- Within-device edges: ego-motion from per-device VIO (rtabmap), IMU pre-integration
- Cross-device edges: ArUco observations from casquette of each grabette's marker = relative-pose factor between casquette pose and grabette pose at that timestamp
- Optionally: grabette gripper cameras can see ArUco markers on the casquette or workspace too → more cross-device factors
- Optimize globally → all three trajectories in one consistent frame

- ✅ Most accurate: every observation contributes to the global optimization
- ✅ Self-correcting: large per-device drift gets compensated by cross-device anchor points
- ✅ Can incorporate workspace ArUco loop closures if added later
- ❌ Significant engineering: needs a multi-agent factor-graph backend, custom calibration, careful time alignment
- ❌ Runs **only offline** on a workstation
- ⚠ Existing frameworks (CCM-SLAM, COSLAM) target live multi-agent; for offline post-process, GTSAM + custom Python is more typical

Conceptually this is what the user is asking about — "fuse all knowledge from all cameras and ArUco into a single big SLAM". It is real, it is done, and it is non-trivial.

#### Option D — Improve each individual SLAM with ArUco loop closures (lighter version of C)

Run per-device SLAM as today, but **inject ArUco observations as loop-closure factors** into each device's SLAM independently. Specifically:

- For each grabette: if its OWN gripper-camera sees a known ArUco marker (e.g., one on the casquette, one on the table, or one on the OTHER grabette), the per-grabette rtabmap gets a loop-closure constraint.
- rtabmap supports custom landmark observations; ORB-SLAM3 also has marker-based loop closure extensions.
- No cross-device joint optimization, but each grabette's SLAM becomes more accurate and consistent across long demos.

This is a more incremental improvement and could be done independently per device, then combined with Option A or B for cross-device fusion.

#### Option B′ — ArUco-only direct relative pose (refinement of B, May 2026)

Option B above assumes simultaneous visibility is rare and treats it as a one-shot calibration anchor. If, instead, **dual-marker visibility is the norm rather than the exception** during typical bimanual manipulation, Option B becomes the *primary* mechanism — not just an anchor — and the entire bimanual pipeline simplifies sharply.

The key observation: for policy training, the only spatial quantity that actually matters is the **inter-gripper relative pose** `T_grabA_grabB(t)`. When both markers are visible in the casquette frame at time `t`, that quantity is directly measured by composing two ArUco detections:

```
T_grabA_grabB(t) = inv(T_casq_grabA(t)) · T_casq_grabB(t)
```

The casquette's own pose appears on both sides and inverts away. Implications:

- **No casquette SLAM is needed at any timestep** — Option A's hardest dependency (Pi Zero 2W VIO) vanishes.
- **No common world frame is needed** — only relative pose is consumed downstream (consistent with §8).
- **No grabette SLAM is needed for this quantity at the dual-visibility timesteps** — although it remains useful for filling the gaps (see below).

**Handling visibility gaps.** When only one or zero markers are visible at `t`, fall back to per-grabette ego-SLAM (the OAK-D rtabmap stack already in production for Grabette V2):

```
T_grabA_grabB(t) ≈ T_grabA_grabB(t_anchor) · (per-grabette ego-motion between t_anchor and t)
```

where `t_anchor` is the most recent dual-visibility frame. Per-grabette rtabmap drift over a typical demo length (~30 s) is ~1 cm; re-anchoring every time both markers are simultaneously visible prevents drift from accumulating.

**Required visibility rate.** For a target inter-gripper error `ε` with per-grabette drift `d`, the longest tolerable gap between dual-visibility anchors is roughly `ε / d`. With `d ≈ 0.03 cm/s` and `ε = 1 cm`, that's ~30 s — i.e. one anchor per demo is the bare minimum. In practice even sporadic dual visibility (a handful of times per demo) keeps error well below 1 cm. The "most-of-the-time" regime the user posits is comfortably sufficient.

**Marker layout implications.** Each grabette needs **multiple markers** biased toward facing the head-mounted casquette during normal use. The back of the grabette body (the user-facing side, opposite the gripper-cam-pointing direction) is the most reliable face during hand-in-front-of-user poses; a top-facing marker covers "user looking down at hand" poses. Two markers per grabette with distinct ArUco IDs (one per device, one per face) is a reasonable starting layout.

**Failure modes.**

| Failure | Effect |
|---|---|
| Both markers occluded for a long stretch | Per-grabette ego-SLAM drift accumulates. Detectable from missing observations — can flag the demo as low-quality and drop or retrain on subset. |
| Only one marker visible | No new direct measurement, but the most-recent `T_grabA_grabB` remains valid; per-grabette SLAM bridges with bounded error. |
| Dual-visible but pose noisy (distance, motion blur) | Average ArUco detections over short windows when both are visible; gate on IMU motion (the same approach used in the teleop bridge) to reject blurry frames. |
| Mistaken marker (left/right swap) | Use distinct ArUco IDs per grabette; verify ID matches expected device at detection time. |

**Prerequisites that move to the critical path.**

- **Casquette camera intrinsic calibration** becomes a hard requirement (solvePnP accuracy depends on it directly). A ChArUco-target calibration session of the casquette's fisheye camera, using the same OpenCV recipe as the Grabette, is the next concrete blocker.
- **Marker → grabette body extrinsics** (one per marker): the small mechanical offset between each marker's center and the grabette's reference frame, measured once and baked into per-device config.

This is essentially **Option B + per-grabette-SLAM gap-filling**, with the casquette-VIO leg of Option A removed entirely. It is a strict simplification of A whenever the dual-visibility assumption holds, and a strict generalization of B (whose "anchor at start only" stance was overly conservative).

## 6. Time synchronization (orthogonal but related)

Whichever option above, the three devices need a common timebase:

- Each Pi runs its own clock. The `metadata.json` already records `wall_clock_start_utc` per device — combined with NTP (the Pis can sync to a shared NTP source on the same WiFi), this gives ~10 ms-class alignment.
- SLAM trajectories and ArUco observations are timestamped on the device that records them; cross-device fusion needs to align these timestamps. 10 ms at 50 fps is half a frame — acceptable for slow manipulation, may need finer sync for fast motions.
- Alternative: a shared optical sync flash at recording start (LED visible to all three cameras) gives frame-accurate offset.

## 7. Recommended path

Without committing yet. **Updated recommendation (May 2026)**: start with Option B′ (§5), which sidesteps the casquette-VIO question if dual-marker visibility is frequent enough. Treat the steps below as the original investigation path; the casquette-VIO experiment (step 1) becomes optional rather than blocking.

1. **First experiment**: validate that casquette VIO (option A, offline) actually works on the casquette's recorded mp4+IMU using the same Docker rtabmap stack as Grabette. If yes — Option A is viable.
2. **In parallel**: try Option B as a calibration anchor: collect a short test where the user holds both grabettes in casquette view at the start, then measure `tx_grabA_grabB` from those frames and from per-grabette SLAMs continued from there. Check drift over a 30s demo.
3. **If both A and B are noisy individually but the data is informative**: Option C (factor-graph joint optimization) becomes worth the engineering cost. The data structure is the same; the analysis layer changes.
4. **Independent improvement**: Option D (ArUco loop closures inside each grabette's SLAM) can be added regardless of the cross-device approach.

Open questions to resolve before deciding:

- [ ] **Casquette VIO feasibility**: does rtabmap (or any modern VIO) running offline on the casquette's recorded data give stable trajectory? Test with a short hand-held recording.
- [ ] **ArUco detection robustness from head distance**: at typical head-to-hand distance (~50 cm), with the casquette's fisheye lens, what's the smallest marker that gives reliable 6-DoF estimation? Pilot test needed.
- [ ] **Marker layout on grabette**: how many markers per device, where to mount them, what dictionary, what size. Constrained by what doesn't interfere with the gripper mechanics.
- [ ] **Time-sync sufficiency**: do we need optical-flash synchronization or is NTP good enough at our typical motion speeds?
- [ ] **Required demo duration**: longer demos make drift matter more. What's our typical episode length?
- [ ] **Failure mode tolerance**: what happens during training if a few demos have bad cross-device alignment? Does the policy still learn coordination?

## 8. Relationship to the delta-based action representation

A crucial property of the recommended LeRobot delta representation
(`[grabette/lerobot] project_grabette_design.md §10.3`) is that **per-gripper deltas
live in each gripper's camera-local frame** — they don't depend on any shared world. The
cross-device alignment is therefore only needed at the **state level** (so the policy
can observe the other gripper's position relative to itself):

```
state_for_grabA(t) = [
    grabA's own pose deltas,
    grabB's position in grabA's frame,  ← needs cross-device alignment
    grabB's orientation in grabA's frame, ← needs cross-device alignment
    gripper widths, etc.
]
```

This means even relatively noisy cross-device alignment (Option A or B at 1–2 cm
precision) is **usable for training**, as long as the noise is consistent. The policy
can learn coordination from "approximate" relative poses; it doesn't need
millimeter accuracy.

## 9. Summary table

| Option | Casquette SLAM needed? | Marker visibility needed | Engineering cost | Accuracy | Notes |
|---|---|---|---|---|---|
| A — Casquette VIO + per-device SLAM, glued by ArUco | Yes (offline) | Once per demo + sparse retries | Moderate | Good | Needs casquette VIO to work |
| B — Casquette as multi-marker observer (no SLAM) | No | Both grabettes simultaneously visible | Low | Lower (sparse measurements) | Good as one-shot calibration anchor |
| **B′ — ArUco-only, gaps filled by per-grabette SLAM** | **No** | **Most-of-the-time dual visibility + per-grabette SLAM bridges gaps** | **Low** | **Good (direct measurement when visible, bounded drift in gaps)** | **Currently recommended starting point** |
| C — Joint multi-agent factor-graph SLAM | All devices integrated | Whatever is available enters as factors | High | Best | Most general, most engineering |
| D — Per-device SLAM + ArUco loop closures | No casquette involvement | Each device sees markers | Moderate | Improved per-device | Independent of cross-device options |

## 10. References

- UMI's tag-based fusion: `[grabette]/../universal_manipulation_interface/scripts_slam_pipeline/06_generate_dataset_plan.py` (lines 88–128 for `tx_slam_tag`, 451–535 for left/right disambiguation)
- UMI's per-camera calibration: `[grabette]/../universal_manipulation_interface/scripts/calibrate_slam_tag.py` (geometric-median tag pose estimation)
- Delta-action convention (camera-local): `[lerobot]/docs/project_grabette_design.md §10.3`
- rtabmap offline VIO pipeline (currently used for Grabette OAK): `[grabette]/../grabette-data/docker/oak_vslam/`
- Multi-agent SLAM literature: CCM-SLAM, COSLAM, Kimera-Multi (all live; for offline-only see GTSAM tutorials)
