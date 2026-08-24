"""Grasp projection: the gripper's two joint angles <-> (strategy, closure).

WHY THIS EXISTS
---------------
The Gripette is not an open/close gripper: it has two joints (proximal, distal)
whose *ratio* is the shape of the grasp — flat-fingered at one extreme, curled
fingertip at the other. A policy trained to regress those two angles directly
inherits a problem it cannot solve: the demonstrated angle is where the human's
fingers sat *while pressing the object*, and a position servo replaying that
angle stops just short and touches nothing. Measured on the real datasets, the
demonstrated grasp uses only 38-60% of the proximal range.

This module reparameterises the same two angles as

    strategy s in [0, 1]   the SHAPE of the grasp (0 = pure proximal / flat,
                           1 = pure distal / curled fingertip)
    closure  c in [0, 1]   how far the gripper has travelled along that shape
                           (0 = fully open, 1 = as closed as the joint can get)

so that a policy can command `c = 1` — "close all the way" — and let the OBJECT
decide the final angle, with the servo's torque cap doing the stopping. The
precise, object-dependent angle stops being something the model has to predict.

THE MAP
-------
Each joint is normalised independently against its own reachable travel:

    closure  drives the PROXIMAL joint      prox = c * lim_prox
    strategy IS the normalised DISTAL target dist = s * lim_dist

Trivially invertible, defined everywhere including the fully-open pose, and the
two channels are independent — noise in one cannot leak into the other.

Why proximal carries the closure: at the grasp, proximal sits at 52% of its
range (48.5 deg of 93.5) — that is the shortfall this module exists to remove —
while distal is at 13%, because that is where the human CHOSE to put it. In a
proximal-dominant grasp the object blocks the proximal motion and the distal
joint is free, so driving proximal to its limit and passing distal through is
both better conditioned and closer to the physics.

An earlier version made strategy a polar ANGLE atan2(v, u) and closure the
radius. It was removed, not merely disabled, because it was measurably worse on
199 real mustard grasps: polar COUPLES the channels (ds/dv = u/(u^2+v^2)), so
noise in the small channel (distal, 13.6 deg +/- 14.7) leaks into the shape
coordinate with a gain that grows as the pose approaches the origin — worst
during the approach — and because `c = 1` scales along the ray, that noise is
then multiplied by k ~ 1/cos(alpha) on the way out. The humans produced 35.8 deg
of distal spread; polar commanded 96.8 deg (x2.71). The map above reproduces it
at 35.4 deg (x0.99). Recorded here so the experiment is not repeated blindly.

ASYMMETRY, BY DESIGN
--------------------
Converted datasets are deliberately asymmetric: the ACTION closure reaches 1.0
(a commanded full close), while the OBSERVED closure never does (it is where the
fingers actually stopped, ~0.82 at most). That gap is the signal "something is
in the hand" and must be preserved — do not normalise it away.

CONVENTIONS
-----------
Angles are in RADIANS in the gripette ROBOT FRAME: 0 = fully open, positive =
closing, command limits from `gripette.config.settings`. Datasets recorded with
an older convention may close NEGATIVE — normalise them with
`normalize_closing_sign` before encoding. The server's own `PROXIMAL_CMD_SIGN`
is a wire-level detail below this layer and must not be applied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gripette.config import settings

# Reachable joint travel, MEASURED on hardware (rgripette-v2, 2026-07-30).
#
# Deliberately NOT `settings.motor{1,2}_max`. Those are the server's
# accepted-command bounds, which are a safety envelope and may be loose:
# motor2_max is 116 deg while no device reaches past ~102. Using a bound larger
# than the real travel as the normaliser understates `v`, which biases every
# strategy toward the proximal end.
#
# These are shared by the Grabette recorder and the Gripette gripper because they
# are the same linkage, so one joint angle means one finger shape on both. That
# is what keeps `s` device-independent: encode runs on recorded angles, decode
# emits gripper commands, and they must agree about what a shape is. Per-device
# variation (zero offset, how far a given torque actually gets) is then a
# clamping concern at the edges rather than a bias in the shape coordinate.
#
# Provenance: proximal from moving the joint by hand to its collision with the
# distal open; distal from a torque-capped stall with the proximal open, so the
# distal figure may be slightly short of the true geometric stop.
#
# Err HIGH rather than low. `c = 1` is meant to command INTO a hard stop: if the
# value exceeds the real travel the servo simply stalls against its torque cap,
# which is the intended behaviour. Too low silently reintroduces the under-close.
REACHABLE_PROXIMAL = math.radians(93.5)
REACHABLE_DISTAL = math.radians(102.0)

# Guard against a zero-variance (synthetic) input in the segmenter's noise floor.
_EPS = 1e-12

# Where a converted dataset records the projection it was built with, relative to
# the dataset root. NOT a key in meta/info.json: that file is parsed into a
# fixed-field dataclass whose unknown keys are dropped on the next write (verified
# — `DatasetInfo.from_dict` warns and discards), so a custom key there would
# survive until the first LeRobot rewrite and then vanish, leaving some copies of
# the dataset annotated and others not. A sidecar is outside that schema and is
# copied with the rest of meta/.
PROJECTION_SIDECAR = "meta/grasp_projection.json"

# Value of the sidecar's "representation" field for a projected dataset. The raw
# alternative is not written at all — absence means raw angles.
REPRESENTATION = "strategy_closure"

# How far a recorded limit may differ from this module's constants before decode
# would land somewhere materially different. 1e-4 rad is ~0.006 deg, far below the
# servo's resolution, so anything above it is a real change rather than float noise.
LIMIT_MATCH_TOL = 1e-4


@dataclass(frozen=True)
class GraspProjection:
    """Invertible map between joint angles and (strategy, closure).

    Frozen because a projection is a calibration, not state: sharing one
    instance between the dataset converter and the eval loop is the point —
    an encode/decode pair that disagree would silently corrupt every grasp.
    """

    lim_prox: float = REACHABLE_PROXIMAL
    lim_dist: float = REACHABLE_DISTAL

    def __post_init__(self) -> None:
        if not (self.lim_prox > 0 and self.lim_dist > 0):
            raise ValueError(f"joint limits must be positive, got "
                             f"{self.lim_prox}, {self.lim_dist}")

    def decode(self, s: float, c: float) -> tuple[float, float]:
        """(strategy, closure) -> (proximal, distal) angles in radians.

        Both inputs are clamped to [0, 1]: a policy trained on this
        parameterisation routinely predicts slightly past 1.0 (measured up to
        1.04 on the real arm), which is the model saturating the channel as
        intended, not an error.
        """
        return _clamp01(c) * self.lim_prox, _clamp01(s) * self.lim_dist

    def encode(self, prox: float, dist: float) -> tuple[float, float]:
        """(proximal, distal) angles in radians -> (strategy, closure).

        Inputs are clamped into the joint limits: real recordings DO exceed them
        (one measured set reaches 101% of the proximal limit, because a human
        hand outranges the servo), and an unclamped value would encode to c > 1
        and decode back to an unreachable target.
        """
        u = _clamp01(prox / self.lim_prox)
        v = _clamp01(dist / self.lim_dist)
        return v, u

    def to_metadata(self, **extra) -> dict:
        """The record a converted dataset carries, so decode can be checked.

        The channel NAMES in info.json say which representation a dataset is in;
        they cannot say which CALIBRATION produced it. Decode must use the same
        limits encode did, and those limits are constants in this module — the
        distal one explicitly provisional, since it came from a torque-capped stall
        rather than a measured geometric stop. Re-measure it and every existing
        dataset silently decodes differently, with nothing recording the change.
        This is that record.
        """
        return {
            "representation": REPRESENTATION,
            "lim_prox_rad": float(self.lim_prox),
            "lim_dist_rad": float(self.lim_dist),
            **extra,
        }

    def matches_metadata(self, meta: dict) -> str | None:
        """None if `meta` describes this projection, else why it does not.

        Compares the limits, not just the representation label: a dataset built
        with different travel constants decodes to different angles, which is
        silent — the commands stay in range and the grasp is simply wrong.
        """
        if meta.get("representation") != REPRESENTATION:
            return (f"representation is {meta.get('representation')!r}, "
                    f"expected {REPRESENTATION!r}")
        for key, mine in (("lim_prox_rad", self.lim_prox),
                          ("lim_dist_rad", self.lim_dist)):
            theirs = meta.get(key)
            if theirs is None:
                return f"missing {key}"
            if abs(float(theirs) - float(mine)) > LIMIT_MATCH_TOL:
                return (f"{key}={float(theirs):.6f} but this build uses "
                        f"{float(mine):.6f} — decode would not match encode "
                        f"({math.degrees(abs(float(theirs) - float(mine))):.2f} deg off)")
        return None

    def full_close(self, s: float) -> tuple[float, float]:
        """Joint angles (rad) for a complete close along strategy `s`.

        This is what a `close` command decodes to: deliberately PAST where any
        demonstration stops, because the torque cap — not the angle — is what
        stops the fingers on the object.
        """
        return self.decode(s, 1.0)


@dataclass(frozen=True)
class GripEvent:
    """A detected closing or opening intention.

    `from_c` is the local REST level the hand was sitting at before the ramp —
    not a global constant. The comfortable posture drifts between and within
    episodes, and measurements on the real datasets showed the closure histogram
    is flat, so a dataset-wide "rest value" is not a real quantity.
    """

    kind: str          # "close" or "open"
    onset: int         # first frame of the ramp — where intention starts
    end: int           # frame the ramp settles at
    from_c: float      # local rest level immediately before the ramp
    to_c: float        # level reached
    amplitude: float   # closure travelled by the ramp


def segment_grip(
    closure: list[float],
    rest_is_closed: bool = False,
    min_amplitude: float = 0.08,
    hold_frames: int = 25,
    hold_tol: float = 0.06,
    smooth: int = 5,
    merge_gap: int = 8,
    rest_window: int = 10,
    slope_frac: float = 0.2,
) -> tuple[list[str], list[GripEvent]]:
    """Label each frame "open" / "rest" / "close", by detecting intention ramps.

    Why ramps rather than a level threshold: the operator does not hold the
    device fully open while approaching — the extreme is uncomfortable — so
    approach-at-rest and grasp-holding are at similar angles and similar (near
    zero) velocity. No level test separates them. Measured on the real datasets,
    though, the closing ramp is 9-15x the hand tremor (amplitude 0.18-0.25 in
    closure against a tremor sd of ~0.018), so the *event* is unambiguous even
    where the *level* is not.

    REST is the default: it is a human comfort posture, not a robot requirement,
    and is kept only so the observed gripper stays in distribution. Nothing has
    to fire to enter it — it is every frame not inside an event.

    The two thresholds that matter are physical, not statistical:
    `min_amplitude` (how far a real intention moves the aperture) and
    `hold_frames`/`hold_tol` (a grasp is HELD). See the amplitude/sustained
    comment in the loop below for why a noise-scaled threshold was abandoned.

    Returns per-frame labels and the events found, in order.
    """
    n = len(closure)
    if n == 0:
        return [], []
    if n < max(smooth, 3):
        return ["rest"] * n, []

    c = _median_filter(list(closure), width=smooth)
    # Noise floor from the frame-to-frame STEPS, robustly (MAD): the median is
    # dominated by the many quiet frames, so a real ramp cannot inflate the
    # threshold it is about to be tested against.
    #
    # NOT the residual of the median filter, which was the first attempt and was
    # wrong by an order of magnitude: a median filter preserves smooth signals,
    # so it leaves almost no residual and every micro-wiggle then cleared the
    # bar. That reported ramps at "241 sigma" and found 300+ spurious events.
    diffs = [c[i] - c[i - 1] for i in range(1, n)]
    med = _median(diffs)
    sigma_step = 1.4826 * _median([abs(d - med) for d in diffs])
    if sigma_step <= _EPS:
        sigma_step = 1e-4        # perfectly clean (synthetic) input

    # Monotone runs, ignoring steps below the noise floor so tremor does not
    # fragment a ramp into dozens of micro-runs.
    step_floor = sigma_step
    runs: list[tuple[int, int, int]] = []   # (direction, start, end)
    i = 1
    while i < n:
        d = c[i] - c[i - 1]
        if abs(d) < step_floor:
            i += 1
            continue
        sign = 1 if d > 0 else -1
        start = i - 1
        while i < n:
            d = c[i] - c[i - 1]
            if abs(d) < step_floor or (1 if d > 0 else -1) != sign:
                break
            i += 1
        runs.append((sign, start, i - 1))

    # Merge same-direction runs separated by a short pause: a deliberate close
    # often has a hesitation in the middle, and splitting there would report two
    # half-amplitude events instead of one real one.
    merged: list[tuple[int, int, int]] = []
    for sign, s, e in runs:
        if merged and merged[-1][0] == sign and s - merged[-1][2] <= merge_gap:
            merged[-1] = (sign, merged[-1][1], e)
        else:
            merged.append((sign, s, e))

    events: list[GripEvent] = []
    for sign, s, e in merged:
        amp = abs(c[e] - c[s])
        # Two physical requirements, because noise-scaled thresholds do NOT work
        # here: hand motion is correlated drift, not white noise, so any bar
        # built from per-frame steps is cleared by slow posture wobble. Measured
        # on real data that produced 250-500 spurious events per dataset.
        #
        #   1. AMPLITUDE — an intention moves the aperture by a large fraction of
        #      the range. Note this is a threshold on the ramp's SIZE, not on its
        #      level: it is invariant to where the hand happens to be resting,
        #      which is what broke the level-threshold approach.
        #   2. SUSTAINED — the level reached is then HELD. A grasp is held while
        #      the object is transported; drift is followed by more drift. This
        #      is the discriminator that needs no noise model at all.
        if amp < min_amplitude:
            continue
        # The hold requirement applies to CLOSES ONLY. Its job is to separate a
        # grasp (held while the object is transported) from a twitch. An open is
        # inherently transient — open, position, close — so demanding a long hold
        # after it rejects exactly the normal case, which is what the rest-closed
        # trajectories exposed.
        if sign > 0:
            hold_to = min(n, e + 1 + hold_frames)
            held = c[e + 1:hold_to]
            if len(held) < hold_frames:
                # Too close to the end to confirm a hold: accept only if the
                # episode simply ends here, which is how a pick-and-lift finishes.
                if e < n - 1 - hold_frames:
                    continue
            elif max(abs(v - c[e]) for v in held) > hold_tol:
                continue
        # Refine the onset to where the motion actually becomes deliberate.
        #
        # The hand CREEPS during the approach — measured on mustard, closure
        # drifts 0.077 -> 0.126 over ~2 s before the grasp ramps to 0.54 in 30
        # frames. Both are monotone increases, so they merge into one run and the
        # raw onset lands at the start of the creep, firing the close command two
        # seconds early. The two differ by ~20x in RATE, so the onset is walked
        # forward to where the slope first reaches a fraction of the run's peak.
        onset = _refine_onset(c, s, e, slope_frac)
        lo = max(0, onset - rest_window)
        rest_level = _median(c[lo:onset]) if onset > lo else c[onset]
        events.append(GripEvent(
            kind="close" if sign > 0 else "open",
            onset=onset, end=e,
            from_c=float(rest_level), to_c=float(c[e]),
            amplitude=float(amp),
        ))

    if rest_is_closed:
        # REST-CLOSED convention: the operator holds the gripper CLOSED when idle.
        # Then rest and grasp are the SAME command — "drive to full close" — and
        # differ only in whether an object blocks the fingers. So the close side
        # needs no interpretation at all, which removes the one genuinely
        # unresolvable ambiguity: relaxing back to rest after a release and
        # closing onto a wide object are kinematically identical, and here they
        # are also the same command, so confusing them costs nothing.
        #
        # Only OPEN needs detecting, and it is the easy direction: opening to
        # clear an object is large and deliberate.
        #
        # Any closing run TERMINATES an open — no amplitude or hold test, because
        # a closing motion is being used merely as "the open is over", not to
        # decide what kind of state follows.
        labels = ["close"] * n
        for ev in (e for e in events if e.kind == "open"):
            stop = n
            for sign, rs, _re in merged:
                if sign > 0 and rs > ev.end:   # the next CLOSING run ends the open
                    stop = rs
                    break
            for t in range(ev.onset, stop):
                labels[t] = "open"
        return labels, [e for e in events if e.kind == "open"]

    # Intermediate-rest convention (how the first datasets were recorded).
    # A close is LATCHED from its onset — the command should fire when the human
    # committed, so the robot's timing matches the demonstration — and holds
    # until an opening event releases it. An open is transient: it exists to let
    # the object go, after which the hand returns to rest.
    labels = ["rest"] * n
    state = "rest"
    ptr = 0
    for ev in events:
        for t in range(ptr, ev.onset):
            labels[t] = state
        if ev.kind == "close":
            state = "close"
            for t in range(ev.onset, min(n, ev.end + 1)):
                labels[t] = "close"
        else:
            for t in range(ev.onset, min(n, ev.end + 1)):
                labels[t] = "open"
            state = "rest"
        ptr = min(n, ev.end + 1)
    for t in range(ptr, n):
        labels[t] = state
    return labels, events


def _refine_onset(c: list[float], start: int, end: int, slope_frac: float) -> int:
    """First frame in [start, end] whose slope reaches `slope_frac` of the peak.

    Separates a deliberate grasp from the slow creep that precedes it. Returns
    `start` unchanged if the run is too short to have a meaningful slope profile.
    """
    if end - start < 3 or slope_frac <= 0.0:
        return start
    slopes = [abs(c[i + 1] - c[i]) for i in range(start, end)]
    peak = max(slopes) if slopes else 0.0
    if peak <= 0.0:
        return start
    bar = slope_frac * peak
    for k, sl in enumerate(slopes):
        if sl >= bar:
            return start + k
    return start


def _median(xs) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def clamp_to_command_limits(prox: float, dist: float) -> tuple[float, float, bool]:
    """Clamp a decoded target into what the gripper server will actually accept.

    `settings.motor1_max` was raised to the measured reachable travel (93.5 deg)
    precisely so a full close is commandable, so in the normal case this clamps
    nothing. It still matters at the edges: the distal command bound (116 deg) is
    looser than the reachable travel this module normalises against, the minimum
    bounds reject the small negative angles that a calibration offset produces,
    and a goal sitting exactly ON a limit can quantise just above it (goals cross
    gRPC as float32 while the limits are float64 — see the motor layer's
    tolerance).

    Returns (proximal, distal, was_clamped).
    """
    lo_p, hi_p = settings.motor1_min, settings.motor1_max
    lo_d, hi_d = settings.motor2_min, settings.motor2_max
    cp = min(max(prox, lo_p), hi_p)
    cd = min(max(dist, lo_d), hi_d)
    return cp, cd, (cp != prox or cd != dist)


def normalize_closing_sign(values: list[float]) -> tuple[list[float], bool]:
    """Force a joint channel to the "closing is positive" convention.

    Older recordings close NEGATIVE. Encoding those unflipped puts every pose on
    the wrong side of the open pose, so this is not cosmetic.

    Decided from the direction the channel TRAVELS, not from its centre: an
    episode is mostly open, so its median sits at ~0 and the sign of that is
    noise. Measured on real data, a median test mislabelled 51 of 199 mustard
    episodes.

    So: despike with a median filter, then take the sign at the largest
    remaining excursion — the grasp, where the direction is unambiguous. The
    filter is what stops an isolated glitch from deciding an episode, and
    percentiles are NOT usable instead: in episodes where the closing phase is a
    small fraction of the frames, the grasp itself falls outside the 95th
    percentile and gets discarded as if it were the outlier.

    Prefer stating the convention explicitly when it is known; this is for the
    converter, where a mixed corpus has to be handled without hand-labelling.

    Returns the corrected values and whether a flip was applied.
    """
    if not values:
        return list(values), False
    clean = _median_filter(values, width=5)
    peak = max(clean, key=abs)
    if peak < 0:
        return [-v for v in values], True
    return list(values), False


def _median_filter(values: list[float], width: int = 5) -> list[float]:
    """Odd-width median filter. Width 5 removes spikes up to two frames long,
    which is what real glitches look like in these recordings.

    At the edges the window is SHIFTED INWARD rather than truncated. Truncating
    shrinks the window to 2-3 samples there, so a spike pair at the very end of
    an episode — a common recording artifact — becomes the majority of its own
    window and survives the filter untouched.
    """
    n = len(values)
    if width < 3 or n < width:
        return list(values)
    half = width // 2
    out = []
    for i in range(n):
        lo = min(max(0, i - half), n - width)
        window = sorted(values[lo:lo + width])
        out.append(window[half])
    return out


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
