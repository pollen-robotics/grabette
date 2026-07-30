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
    closure  c in [0, 1]   how far along that shape the gripper has travelled
                           (0 = fully open, 1 = as closed as this shape can
                           mechanically get)

so that a policy can command `c = 1` — "close all the way along this shape" —
and let the OBJECT decide the final angle, with the servo's torque cap doing the
stopping. The precise, object-dependent angle stops being something the model
has to predict.

The mapping is pure geometry: no fitting is required and none of it is specific
to a dataset. Two optional refinements are exposed as parameters so the shape
can be made more faithful later without changing any call site:

    p       the full-close BOUNDARY. p = inf means the joints reach their limits
            independently (a box corner). Real fingers foul the thumb first, so
            the true boundary is a curve; p = 2 is a quarter circle. Fit from
            fully-closed poses.
    a, b    the PATH exponents. a = b = 1 is a straight ray, i.e. both joints
            move in fixed proportion. a < b makes the proximal joint lead early
            and the distal curl late, which is what a human hand actually does.

Defaults (p = inf, a = b = 1) are the plain geometric map, so behaviour is
well-defined before anything is fitted.

CONVENTIONS
-----------
Angles are in RADIANS in the gripette ROBOT FRAME: 0 = fully open, positive =
closing, limits from `gripette.config.settings`. Datasets recorded with an older
convention may close NEGATIVE — normalise them with `normalize_closing_sign`
before encoding, or `atan2` lands in the wrong quadrant and every strategy is
wrong. The server's own `PROXIMAL_CMD_SIGN` is a wire-level detail below this
layer and must not be applied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gripette.config import settings

# Half-turn of the strategy sweep: s = 0 is pure proximal, s = 1 pure distal.
_QUARTER = math.pi / 2

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
# Numerical guards. _EPS keeps division and root-finding away from 0/0 at the
# fully-open pose; _TOL is the bisection tolerance on s (~1e-4 of full sweep,
# far below the servo's resolution, so it is exact for our purposes).
_EPS = 1e-12
_TOL = 1e-9


@dataclass(frozen=True)
class GraspProjection:
    """Invertible map between joint angles and (strategy, closure).

    Frozen because a projection is a calibration, not state: sharing one
    instance between the dataset converter and the eval loop is the point —
    an encode/decode pair that disagree would silently corrupt every grasp.
    """

    lim_prox: float = REACHABLE_PROXIMAL
    lim_dist: float = REACHABLE_DISTAL
    p: float = math.inf   # full-close boundary exponent
    a: float = 1.0        # proximal path exponent
    b: float = 1.0        # distal path exponent

    def __post_init__(self) -> None:
        if not (self.lim_prox > 0 and self.lim_dist > 0):
            raise ValueError(f"joint limits must be positive, got "
                             f"{self.lim_prox}, {self.lim_dist}")
        if self.p < 1:
            raise ValueError(f"boundary exponent p must be >= 1, got {self.p}")
        if self.a <= 0 or self.b <= 0:
            raise ValueError(f"path exponents must be > 0, got a={self.a}, b={self.b}")

    # ---- geometry ------------------------------------------------------

    def boundary(self, s: float) -> tuple[float, float]:
        """The fully-closed NORMALISED pose (u, v) for strategy `s`.

        Walks out from the open pose along the direction `s` selects, until it
        meets the boundary `u**p + v**p = 1`. With p = inf that is the unit box,
        so one joint always ends exactly at its limit.
        """
        alpha = _QUARTER * _clamp01(s)
        ca, sa = math.cos(alpha), math.sin(alpha)
        if math.isinf(self.p):
            k = 1.0 / max(ca, sa, _EPS)
        else:
            k = (ca ** self.p + sa ** self.p) ** (-1.0 / self.p)
        return k * ca, k * sa

    def full_close(self, s: float) -> tuple[float, float]:
        """Joint angles (rad) for a complete close along strategy `s`.

        This is what a `close` command decodes to: deliberately PAST where any
        demonstration stops, because the torque cap — not the angle — is what
        stops the fingers on the object.
        """
        return self.decode(s, 1.0)

    # ---- the map -------------------------------------------------------

    def decode(self, s: float, c: float) -> tuple[float, float]:
        """(strategy, closure) -> (proximal, distal) angles in radians."""
        u_end, v_end = self.boundary(s)
        c = _clamp01(c)
        u = u_end * c ** self.a
        v = v_end * c ** self.b
        return u * self.lim_prox, v * self.lim_dist

    def encode(self, prox: float, dist: float) -> tuple[float, float]:
        """(proximal, distal) angles in radians -> (strategy, closure).

        Returns s = nan at the fully-open pose, where the strategy is genuinely
        undefined (both joints at zero carry no shape information). Callers
        working on trajectories should hold the last defined value — see
        `encode_trajectory`, which does exactly that.

        Inputs are clamped into the joint limits: real recordings DO exceed them
        (one measured set reaches 101% of the proximal limit, because a human
        hand outranges the servo), and an unclamped value would encode to c > 1
        and decode back to an unreachable target.
        """
        u = _clamp01(prox / self.lim_prox)
        v = _clamp01(dist / self.lim_dist)
        if u <= _EPS and v <= _EPS:
            return math.nan, 0.0

        if abs(self.a - self.b) <= _TOL:
            # Equal path exponents: the joint ratio is constant along the path,
            # so the direction alone gives s and the radius gives c. Closed form.
            alpha = math.atan2(v, u)
            s = alpha / _QUARTER
            u_end, v_end = self.boundary(s)
            end = math.hypot(u_end, v_end)
            frac = math.hypot(u, v) / max(end, _EPS)
            return s, _clamp01(frac ** (1.0 / self.a))

        # Unequal exponents: the ratio drifts along the path, so s must be
        # solved for. `_residual` is monotone decreasing in s (raising s pulls
        # the boundary toward the distal axis), so bisection is safe.
        lo, hi = 0.0, 1.0
        f_lo, f_hi = self._residual(lo, u, v), self._residual(hi, u, v)
        if f_lo <= 0.0:
            s = lo
        elif f_hi >= 0.0:
            s = hi
        else:
            while hi - lo > _TOL:
                mid = 0.5 * (lo + hi)
                if self._residual(mid, u, v) > 0.0:
                    lo = mid
                else:
                    hi = mid
            s = 0.5 * (lo + hi)
        return s, _clamp01(self._closure_at(s, u, v))

    def _residual(self, s: float, u: float, v: float) -> float:
        """Disagreement between the closure implied by each joint, at strategy s.

        Zero exactly when (u, v) lies on strategy s's path. Positive means s is
        too low (the distal joint is further along than this strategy allows).
        """
        u_end, v_end = self.boundary(s)
        c_u = (u / max(u_end, _EPS)) ** (1.0 / self.a)
        c_v = (v / max(v_end, _EPS)) ** (1.0 / self.b)
        return c_v - c_u

    def _closure_at(self, s: float, u: float, v: float) -> float:
        """Closure along strategy s, averaging both joints' estimates.

        Averaging rather than picking one keeps the round-trip error symmetric
        when (u, v) sits slightly off the path — which it will, since real
        recordings are noisy.
        """
        u_end, v_end = self.boundary(s)
        est = []
        if u_end > _EPS:
            est.append((u / u_end) ** (1.0 / self.a))
        if v_end > _EPS:
            est.append((v / v_end) ** (1.0 / self.b))
        return sum(est) / len(est) if est else 0.0

    # ---- trajectories --------------------------------------------------

    def encode_trajectory(
        self, prox: list[float], dist: list[float], close_at: float = 0.5
    ) -> tuple[list[float], list[float], list[bool]]:
        """Encode a whole episode -> (strategy, closure, close_flag) per frame.

        `s` is undefined while the gripper is fully open, so it is filled from
        the nearest frame where it IS defined (backwards first, then forwards).
        Without this the open frames would carry nan into training.

        `close_flag` is `closure >= close_at`. This threshold is the one lossy
        part of the projection: it decides WHEN the jaws shut, and a wrong onset
        moves the close to a point on the trajectory the human never closed at.
        Sweep it against real episodes rather than trusting the default.
        """
        if len(prox) != len(dist):
            raise ValueError(f"length mismatch: {len(prox)} prox vs {len(dist)} dist")
        pairs = [self.encode(p, d) for p, d in zip(prox, dist)]
        s_raw = [s for s, _c in pairs]
        closure = [c for _s, c in pairs]

        s_filled = list(s_raw)
        last = math.nan
        for i, val in enumerate(s_filled):          # carry backwards
            if not math.isnan(val):
                last = val
            elif not math.isnan(last):
                s_filled[i] = last
        nxt = math.nan
        for i in range(len(s_filled) - 1, -1, -1):  # then forwards, for a
            if not math.isnan(s_filled[i]):         # leading run of open frames
                nxt = s_filled[i]
            elif not math.isnan(nxt):
                s_filled[i] = nxt
        return s_filled, closure, [c >= close_at for c in closure]


def clamp_to_command_limits(prox: float, dist: float) -> tuple[float, float, bool]:
    """Clamp a decoded target into what the gripper server will actually accept.

    `decode(s, 1.0)` intentionally targets the REACHABLE travel, which is larger
    than the server's accepted-command bounds (`settings.motor{1,2}_max`, 85 and
    116 deg): the proximal joint reaches ~93.5 deg but commands above 85 are
    rejected outright. Without this, a full close would raise instead of closing.

    Clamping here loses the last few degrees of proximal travel rather than the
    grasp. Raising `motor1_max` toward the measured reachable value would recover
    it, but that widens what the hardware accepts and is not this module's call.

    Returns (proximal, distal, was_clamped).
    """
    lo_p, hi_p = settings.motor1_min, settings.motor1_max
    lo_d, hi_d = settings.motor2_min, settings.motor2_max
    cp = min(max(prox, lo_p), hi_p)
    cd = min(max(dist, lo_d), hi_d)
    return cp, cd, (cp != prox or cd != dist)


def normalize_closing_sign(values: list[float]) -> tuple[list[float], bool]:
    """Force a joint channel to the "closing is positive" convention.

    Older recordings close NEGATIVE. Encoding those unflipped puts every pose in
    the wrong quadrant, so this is not cosmetic.

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
