# The grasp projection

**Status:** design + results, 2026-08-06. Code:
`packages/gripette/gripette/grasp_projection.py`. To record data for it, see
[grasp_projection_recording_procedure.md](grasp_projection_recording_procedure.md).

## The problem it solves

The Gripette has two gripper joints (proximal, distal). Train a policy to regress
those two angles and it inherits a problem it cannot solve: **the demonstrated
angle is where the human's fingers sat while pressing the object.** A
position-controlled servo replaying that same angle stops just short and touches
nothing. Measured across the first three real datasets, a demonstrated grasp uses
only **38–60% of the proximal range**.

The angle that works is a property of the *object*, not of the task — so asking
the model to predict it is asking it to measure object width from an image, and
be right to within a couple of degrees, every time.

## The reparameterisation

The same two angles, re-expressed:

| | meaning | drives |
|---|---|---|
| `s` **strategy** | the SHAPE of the grasp: 0 = pure proximal (flat finger), 1 = pure distal (curled fingertip) | the distal joint |
| `c` **closure** | how far along that shape: 0 = open, 1 = fully closed | the proximal joint |

```
decode:  prox = clamp01(c) * REACHABLE_PROXIMAL      # 93.5°
         dist = clamp01(s) * REACHABLE_DISTAL        # 102.0°
encode:  s = clamp01(dist / lim_dist),  c = clamp01(prox / lim_prox)
```

Each joint is normalised independently against its own **measured reachable
travel** — deliberately not `settings.motor*_max`, which are the server's
accepted-command bounds and are looser (`motor2_max` is 116° where no device
reaches past ~102°). A normaliser larger than the real travel understates that
channel and biases every strategy toward the other joint.

The policy now commands `c = 1` — "close all the way" — and the **object** decides
the final angle, with the servo's torque cap doing the stopping. The
object-dependent angle is no longer something the model has to predict.

### Why closure drives the proximal joint

At the grasp, proximal sits at 52% of its range (48.5° of 93.5°) — *that is the
shortfall* — while distal sits at 13%, because that is where the human chose to
put it. In a proximal-dominant grasp the object blocks the proximal motion and the
distal joint is free. Driving proximal to its limit and passing distal through is
both better conditioned and closer to the physics.

### Why not polar

The first version made `s` the polar angle `atan2(v, u)` and `c` the radius. It
was removed rather than kept as an option, because it is measurably worse on 199
real mustard grasps:

- it **couples** the channels (`ds/dv = u/(u²+v²)`), so noise in the small channel
  (distal, 13.6° ± 14.7) leaks into the shape coordinate with a gain that grows as
  the pose approaches the origin — i.e. worst during the approach;
- because `c = 1` scales along the ray, that noise is then multiplied by
  `k ≈ 1/cos α` on the way out.

The humans produced **35.8°** of distal spread; polar commanded **96.8°**
(×2.71). The independent map reproduces it at **35.4°** (×0.99). Polar
manufactured variation that was not in the demonstrations, in the exact channel
the policy has to learn.

## The asymmetry is deliberate

In a converted dataset:

- **action** closure reaches **1.0** on grasp frames — a commanded full close;
- **observation** closure never does (~0.82 max) — it is where the fingers
  actually stopped.

That gap is the only signal for *"is something in the hand"*. Do not normalise it
away, and do not rewrite the state to match the action. Everything outside the
grasp — approach, open, rest — passes through untouched; an early version that
zeroed closure on open frames shifted the approach aperture by 76°.

## How a dataset declares which representation it is in

Range alone cannot answer this — a raw dataset's gripper q99 measured **0.894**,
below 1.0 — so three things carry it, in descending authority:

| where | says | survives |
|---|---|---|
| `features.*.names` in `info.json` | `proximal`/`distal` (raw) vs `strategy`/`closure` (projected) | yes — `features` is a declared LeRobot field |
| `meta/grasp_projection.json` | the exact **calibration**: `lim_prox_rad`, `lim_dist_rad`, source root | yes — outside LeRobot's schema, never rewritten |
| card tag `grasp-projection` | queryable from the Hub API without downloading | yes, on the Hub |

The sidecar exists because names say *which* representation but not *which
calibration*, and decode must use the limits encode used. `REACHABLE_DISTAL` is
explicitly provisional (it came from a torque-capped stall, not a measured stop) —
re-measure it and every existing dataset decodes differently, silently, since the
commands stay in range and only the grasp is wrong. `check_publish` compares the
recorded limits against the running build and errors on any drift.

**Do not put this in `info.json` as a custom key.** That file is parsed into a
fixed-field dataclass which drops unknown keys on the next write (verified:
`DatasetInfo.from_dict` warns and discards), and our own converter triggers a
rewrite via `recompute_stats` — so the key would survive until then and vanish
after, leaving some copies annotated and others not.

None of this reaches the trained **checkpoint**, which keeps only
`{"action": {"type": "ACTION", "shape": [11]}}` — no names. That is why
`evaluate.py` has to sniff the saved normaliser, and why remote inference needs
`--grasp_projection on` explicitly.

## Using it

**Convert a dataset** (never in place — the raw dataset is the archive):

```bash
uv run python -m grabette_postprocess.grasp_projection_convert \
    --src <raw-dataset-root> --dst <converted-root> [--rest-is-closed]
```

**Evaluate.** `evaluate.py --grasp_projection {auto,on,off}`. `auto` inspects the
checkpoint's saved normaliser (`action.max[-2:] ≤ 1.0`) — channel *names* are not
in a checkpoint, so the range is the only available evidence, and `q99` is
unusable because a raw dataset's q99 (0.894) is also below 1. With `--policy_addr`
there is no local checkpoint to inspect, so **pass `on` explicitly**; `auto` exits
rather than guess, because guessing "raw" would send a closure of 1.0 as 1.0
*radian* and under-close every grasp with nothing in the log to say so.

## Results on hardware

| policy | grasps | commanded proximal | measured proximal | notes |
|---|---|---|---|---|
| Diffusion (mustard) | 5/5 | 93.5° | 60.2 / 60.6 / 68.2° | vs 48–51° demonstrated, no assist |
| π0.5 (pick3, remote) | 2/2 + reruns | 93.5° | 58.0 / 61.9° | grasped the top of the bottle in one run |

Closure behaved as designed in every run: flat at 0.12–0.24 through the whole
approach, then a single-step jump to ~1.0 held for the rest of the episode.

The strategy channel is genuinely used, not collapsed to a per-object constant —
two π0.5 runs on the *same* bottle produced `s → 0.00` (measured distal 0.8°) and
`s ≈ 0.20` (19.6°), i.e. two different grasp shapes. Across objects it is
differentiated too: mustard ≈ 0.001, can ≈ 0.11, cup ≈ 0.27–0.29.

## Known limits

- **Grip force is the torque cap, and the cap is the grip force.** Because `c = 1`
  drives into a stop, the servo sits at its torque ceiling for the whole hold
  (observed: proximal load pegged for 5–6 s). The ceiling is
  `settings.torque_ceiling` (0.5), and an unset per-command limit resolves *to*
  the ceiling. 0.5 crushes a paper cup; ~0.25 was already sufficient for almost
  anything. Lower it with `--grip_torque_limit`, and note the stop becomes
  force-limited rather than geometry-limited as you do, so grip depth on heavy
  objects gets shallower.
- **Replan boundaries can un-commit the close.** At a chunk boundary the fresh
  draw at grasp onset is genuinely bimodal (pre-grasp ~0.2 vs commit ~1.0) and was
  observed dipping back for 3–4 steps before recovering. It self-corrected in
  every run because the gripper's own lag carried the fingers through, but that
  margin is servo lag, not design. The lever is chunk length, not a latch.
- **The distal travel figure may be slightly short.** It came from a
  torque-capped stall, not a hand-measured geometric stop.
- **Segmentation depends on the recording convention.** `--rest-is-closed`
  changes what is detectable; see the recording procedure.

## Tests

| file | covers |
|---|---|
| `packages/gripette/tests/test_grasp_projection.py` | the map: round trips both directions, channel independence, saturation above 1.0, clamping, sign normalisation, median filter edges |
| `packages/gripette/tests/test_grip_segmentation.py` | event detection: ramps vs drift vs twitches, both rest conventions |
| `packages/grabette-postprocess/tests/test_grasp_projection_convert.py` | the conversion: full close commanded, state left honest, strategy latched, non-grasp frames untouched |
| `packages/gripette/tests/test_torque_ceiling.py` | the ceiling is enforced, including for an unset per-command limit |
| `packages/gripette/tests/test_limit_tolerance.py` | a goal exactly at a joint limit survives float32 quantisation |
