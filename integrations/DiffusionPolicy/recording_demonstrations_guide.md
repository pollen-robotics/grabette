# Recording great demonstrations (Grabette → Gripette)

**Audience:** the operator recording demonstrations with the hand-held Grabette.
**Status:** starting draft — distilled from the Gripette debugging campaign (Jun 2026).
**Companion:** for the *why* (covariate shift, DAgger, action-manifold theory), see
[`data_collection_for_imitation_learning.md`](data_collection_for_imitation_learning.md).
This file is the practical "what to do with your hands."

---

## The one principle

> **A policy reproduces the *statistics* of your demonstrations. It keeps the
> variation it can *condition on* (predict from what the camera sees) and
> *averages* the variation it can't.**

Two rules follow, and almost every recording mistake is a violation of one:

1. **Be consistent** where the policy can't read your intent from the scene
   (free / stylistic choices) — otherwise it averages your spread into a single
   compromise, which is often a *worse* behavior than any demo you gave it.
2. **Be diverse** where you need generalization **and** the cause is visible to
   the camera.

A diffusion policy *can* represent multiple modes — but only ones it can tell
apart from the observation. An unconditioned spread doesn't get sampled; it gets
collapsed to the mean.

---

## Part A — Be CONSISTENT (the policy averages these)

These are choices the camera can't disambiguate, so variation here gets averaged
into one (often bad) mode.

1. **Grasp approach angle.** Pick **one natural, easy angle** and stay close to
   it. A *tight* spread (≈ ±10°, centered on that easy angle) is fine and gives a
   little robustness. Do **not** grasp the same object from widely different
   angles: the policy can't tell which to use, averages them, and the mean is
   often a bad/hard grasp (e.g. straight top-down). *Center the variation on a
   good angle; keep the spread inside the "all of these still work" band.*
   *(If you genuinely need angle robustness, see Part B — make the angle track
   the object, don't free-vary it.)*

2. **Reach decisively and SEAT the object.** Drive the gripper so the object ends
   up **deep in the jaw**, in one confident continuous motion. Do **not** glide
   to a gentle stop the instant the fingers touch. A decelerating "settle" teaches
   the policy "near the object → tiny motion → stop," and it then creeps and
   **stalls ~1 cm short** at deployment. Reach *through* to a seated depth.

3. **Close firmly and decisively.** One committed, firm close. Avoid slow,
   tentative, or partial closures — the policy imitates hesitation and ends up
   *touching but not grasping*.

4. **Move smoothly.** No jitter, no nervous micro-corrections. The policy
   reproduces shakiness (diffusion turns noisy demos into erratic action
   samples), *and* fast/jerky motion blurs frames and jumps the view between
   frames, making the SLAM tracker lose lock and emit pose glitches (see Part E).
   Steady, moderate pace.

5. **Keep a consistent pace** and record at the rate you'll deploy at.

---

## Part B — Be DIVERSE (the policy generalizes over these)

These are visible to the camera, so variation here becomes genuine
generalization.

1. **Object position** — cover the whole workspace area you'll deploy in. The
   policy only works where it has seen the object.
2. **Object orientation** — diversify **only if you grasp aligned with it**, so
   the camera-visible orientation *determines* the grasp angle. Then the angle is
   conditioned and learnable, and you get real orientation robustness. A grasp
   angle unrelated to the object's orientation is the Part-A mistake in disguise.
3. **Start poses / approach directions** you'll actually encounter.
4. **Appearance** (lighting, background) for real-world robustness — the camera
   sees it, so it's conditioned. (Training-time color-jitter adds more.)

---

## Part C — Failures & retries (handle deliberately, or not at all)

- **Default: clean, first-try successes.** A sloppy miss-then-fix that sneaks
  into an otherwise-"clean" demo is poison — the policy imitates the miss and
  reproduces your near-miss as if it were the plan.
- **If** you want recovery robustness (valuable on the real arm), record it
  **on purpose**, not by accident:
  - **Small fraction** (~10% of episodes).
  - **Diverse miss directions** — not always the same offset, or the policy
    learns one biased approach (it will literally re-create that offset).
  - **Make the failure observable** so the policy learns *"detect miss →
    recover,"* not *"always miss."* A clearly sensable failure state (e.g. the
    gripper closing fully on nothing) is what lets it gate the recovery.

---

## Part D — Coverage & quantity

- The policy is reliable **only within the distribution you demonstrate** —
  positions, orientations, lighting, start poses.
- Diverse variables need **enough samples to populate**. Don't spread a small
  episode budget thin across many dimensions of variation; either add episodes or
  cut dimensions.

---

## Part E — Protect the SLAM tracking (avoid pose glitches)

The hand pose comes from visual SLAM on the wrist camera. When the camera can't
see enough of the **static scene**, tracking degrades and the trajectory
**teleports** — those frames get zeroed, or the whole episode dropped, and it's
usually the **grasp phase** (the most important part) that's lost.

1. **Don't let the grasped object occlude the camera.** As you close on and lift
   the object, keep some background/scene in the camera's view — don't bring the
   object right up to the lens or let it fill the frame. This is the single
   biggest cause of glitches we see: the object blanks the wrist camera through
   the grasp + lift, and SLAM relocalizes with a jump.
2. **No fast swings.** Fast motion = motion blur + large frame-to-frame jumps =
   lost tracking. Keep a steady, moderate pace (this is the tracking side of
   Part A's "move smoothly").

*Symptom to recognize in QA* (`clean_dataset.py` / the postprocess trajectory
checks): a burst of periodic, same-size position jumps ≈ the object occluded the
camera through the grasp and lift.

---

## Part F — Pre-flight (engineer, once)

- **Camera/device identical to deployment** (mount, FOV, calibration). The policy
  keys off the camera view; a different view at deployment is a different task.
- **Record the state you'll have at deployment** (on the real arm: the *measured*
  gripper position, not a command).
- **Verify the recorded reference frame matches the deployment control frame**
  end-to-end (this single mismatch caused the largest errors we hit).

---

## Quick checklist

- [ ] One natural, easy grasp angle; tight ±10°, **not** hard-top-down.
- [ ] Reach *through* to seat the object deep in the jaw — no gentle settle.
- [ ] Firm, decisive close.
- [ ] Smooth motion, no jitter; no fast swings (also protects SLAM tracking).
- [ ] Object never blanks the camera — keep the scene in view through grasp + lift.
- [ ] Diverse object **positions** (and orientations *only if* grasp-aligned).
- [ ] Clean first-try successes; no accidental misses.
- [ ] Consistent pace; record at deployment rate.
- [ ] Camera/frame/state verified vs deployment.

---

## Why each rule (the failure it prevents)

| Recording habit | Failure mode it prevents |
|---|---|
| Tight, centered grasp angle | Mode-collapse to a bad average grasp (top-down) |
| Reach through / seat the object | Under-reach: policy creeps and stalls ~1 cm short |
| Firm decisive close | Touch-but-don't-grasp; hesitant/abortable close |
| Smooth motion / no fast swings | Erratic deployment behavior **and** motion-blur SLAM tracking loss |
| Object never occludes the camera | SLAM tracking loss → pose glitches; filtered / dropped episodes |
| Diverse object positions | Fails outside the demonstrated workspace |
| Clean successes by default | Policy imitating an injected/accidental miss |
| Consistent camera/frame/state | Large systematic offsets (frame mismatch) |
