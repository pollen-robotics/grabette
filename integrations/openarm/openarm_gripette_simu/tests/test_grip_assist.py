"""Regression tests for evaluate.py's GripAssist (the --grip_assist primitive).

No hardware, no simulator: synthetic (lag, load, advance) traces drive the REAL
class — deliberately not a reimplementation, because an earlier mirrored-logic
test grew a bug of its own and passed while the real code was wrong.

Contact is a STALL: the fingers have stopped advancing, short of the pose we
commanded, while drawing load. Every term is load-bearing and every one of them
is here because it failed on the real robot first:
  - SETTLED  — transit to a fresh command lags ~0.09 rad and draws cap current;
               without this the assist latched 3 ticks in with ZERO top-up.
  - LAG      — settled short of the commanded pose means something is in the way.
  - LOAD     — a limp/disabled servo is also settled+lagging but pushes nothing.
And slip is LOAD-ONLY: a gripped finger keeps creeping while the policy raises
its own command, which the stall test read as slip and ratcheted the offset
0.18 -> 0.24 -> 0.30 on the arm.

Reference numbers (real gripper, 25% torque cap, 0.02 rad steps): free motion
lag 0.002-0.004 with load 0-24; from contact lag 0.014+ and growing while load
climbs 72 -> 250 (the cap).

Needs the `eval` extra (evaluate.py imports lerobot/torch):
    uv sync --package openarm-gripette-simu --extra eval --extra test
    uv run pytest integrations/openarm/openarm_gripette_simu/tests
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_EVALUATE = Path(__file__).resolve().parents[1] / "examples" / "evaluate.py"

REF = (0.40, 0.30)          # --start_gripper on the real arm
THRESH = 50.0               # load floor (only excludes the limp-servo case)
CLOSED = (0.65, 0.88)       # a typical policy close (mustard campaign)
KW = dict(ref=REF, min_close=0.15, stable_ticks=3, stable_eps=0.01, step=0.05,
          max_extra=0.30, dwell_ticks=1, confirm_ticks=2, lag=0.010, squeeze=0.10)

# Synthetic tick flavours: (lag behind the sent pose, load, advance since last tick)
FREE = dict(lag=0.003, load=10.0, adv=0.000)      # arrived, nothing in the jaws
STUCK = dict(lag=0.030, load=200.0, adv=0.000)    # settled, blocked, pushing
TRANSIT = dict(lag=0.090, load=250.0, adv=0.050)  # still travelling — NOT contact
LIMP = dict(lag=0.050, load=5.0, adv=0.000)       # stuck but drawing nothing
LOADED_MOVING = dict(lag=0.003, load=250.0, adv=0.000)   # tracking fine, loaded
CREEP_LOADED = dict(lag=0.030, load=250.0, adv=0.010)    # gripped and creeping


@pytest.fixture(scope="module")
def GripAssist():
    """Load the class straight out of examples/evaluate.py."""
    spec = importlib.util.spec_from_file_location("_evaluate", _EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GripAssist


def drive(assist, seq):
    """Feed [(cmd, flavour), ...]; synthesize the measured position from the
    flavour's lag/advance relative to what the assist last SENT.
    Returns [(sent, state, offset), ...]."""
    out = []
    for i, (cmd, kind) in enumerate(seq):
        base = np.asarray(assist._sent if assist._sent is not None else cmd, float)
        meas = tuple(base - kind["lag"])
        if assist._meas is not None and kind.get("adv"):
            meas = tuple(np.asarray(assist._meas) + kind["adv"])
        sent = assist.update(cmd, (kind["load"],) * 2, step=i, meas=meas)
        out.append((sent, assist.state, round(assist.offset, 3)))
    return out


def test_under_close_then_contact_latches(GripAssist):
    """The headline case: policy settles closed but short -> top up -> latch."""
    a = GripAssist(THRESH, **KW)
    r = drive(a, [(CLOSED, FREE)] * 8 + [(CLOSED, STUCK)] * 12)
    assert r[0][1] == "IDLE"                       # not yet judged
    assert r[3][1] == "ASSISTING"                  # settled + empty -> engage
    assert a.state == "GRIPPED" and a.n_gripped == 1
    assert 0 < a.offset <= KW["max_extra"]


def test_extra_closure_follows_the_policys_posture(GripAssist):
    """With a 2-DoF index vs a fixed thumb the prox/distal ratio IS the grasp
    type, so the assist must deepen the policy's posture, not impose one."""
    a = GripAssist(THRESH, **KW)
    r = drive(a, [(CLOSED, FREE)] * 8 + [(CLOSED, STUCK)] * 12)
    extra = np.array(r[-1][0]) - np.array(CLOSED)
    direction = np.array(CLOSED) - np.array(REF)
    cos = float(extra @ direction / (np.linalg.norm(extra) * np.linalg.norm(direction)))
    assert cos > 0.999


def test_squeeze_presses_past_contact(GripAssist):
    """Contact is only a touch; grip force needs pressing further."""
    no_sq = GripAssist(THRESH, **{**KW, "squeeze": 0.0})
    drive(no_sq, [(CLOSED, FREE)] * 8 + [(CLOSED, STUCK)] * 12)
    sq = GripAssist(THRESH, **{**KW, "squeeze": 0.10})
    drive(sq, [(CLOSED, FREE)] * 8 + [(CLOSED, STUCK)] * 12)
    assert sq.state == no_sq.state == "GRIPPED"
    assert sq.offset > no_sq.offset
    assert abs((sq.offset - no_sq.offset) - 0.10) < 0.06


def test_transit_is_not_contact(GripAssist):
    """Fingers travelling to a fresh command lag hard AND draw cap current.
    Real-arm bug 2026-07-28: latched at tick 3 with zero top-up."""
    a = GripAssist(THRESH, **KW)
    drive(a, [(CLOSED, TRANSIT)] * 10)
    assert a.state != "GRIPPED" and a.n_gripped == 0


def test_load_while_tracking_is_not_contact(GripAssist):
    a = GripAssist(THRESH, **KW)
    drive(a, [(CLOSED, FREE)] * 4 + [(CLOSED, LOADED_MOVING)] * 8)
    assert a.state != "GRIPPED"


def test_limp_servo_is_not_contact(GripAssist):
    """Not moving and lagging, but pushing nothing — must not read as a grasp."""
    a = GripAssist(THRESH, **KW)
    drive(a, [(CLOSED, FREE)] * 4 + [(CLOSED, LIMP)] * 8)
    assert a.state != "GRIPPED"


def test_single_stall_tick_is_debounced(GripAssist):
    """The load register is noisy (spikes to ~88 measured); one tick must not
    latch a grasp that isn't there."""
    a = GripAssist(THRESH, **KW)
    drive(a, [(CLOSED, FREE)] * 5 + [(CLOSED, STUCK)] + [(CLOSED, FREE)] * 5)
    assert a.state != "GRIPPED" and a.n_gripped == 0


def test_empty_jaws_clamp_release_and_rearm(GripAssist):
    """Nothing in reach -> stop at max_extra, release, and do NOT chatter:
    re-arm only after the policy opens again."""
    a = GripAssist(THRESH, **KW)
    r = drive(a, [(CLOSED, FREE)] * 40)
    assert a.state == "EXHAUSTED" and a.n_exhausted == 1
    assert a.offset == 0.0
    assert max(x[2] for x in r) <= KW["max_extra"]
    held = drive(a, [(CLOSED, FREE)] * 5)
    assert a.state == "EXHAUSTED" and all(s[0] == CLOSED for s in held)
    drive(a, [(REF, FREE)] * 2)                    # policy opens -> re-arm
    assert a.state == "IDLE"
    drive(a, [(CLOSED, FREE)] * 5)
    assert a.state == "ASSISTING"


def test_already_gripping_is_left_untouched(GripAssist):
    """The safety property: a working grasp must never be perturbed."""
    a = GripAssist(THRESH, **KW)
    r = drive(a, [(CLOSED, STUCK)] * 12)
    assert a.state == "GRIPPED" and a.offset == 0.0
    assert all(s[0] == CLOSED for s in r)


def test_slip_when_load_falls_reengages(GripAssist):
    a = GripAssist(THRESH, **KW)
    drive(a, [(CLOSED, STUCK)] * 12)
    assert a.state == "GRIPPED"
    drive(a, [(CLOSED, FREE)] * 6)                 # load gone = genuine slip
    assert a.state == "ASSISTING" and a.offset > 0.0


def test_creeping_at_full_load_is_not_slip(GripAssist):
    """Real-arm bug 2026-07-28: a gripped finger creeps while the policy raises
    its command; treating that as slip ratcheted the offset 0.18 -> 0.24 -> 0.30
    with load pinned at the cap."""
    a = GripAssist(THRESH, **KW)
    drive(a, [(CLOSED, FREE)] * 8 + [(CLOSED, STUCK)] * 12)
    assert a.state == "GRIPPED"
    latched = a.offset
    assert latched > 0.0, "need a non-zero latched offset to detect ratcheting"
    drive(a, [(CLOSED, CREEP_LOADED)] * 10)
    assert a.state == "GRIPPED"
    assert a.offset == latched


def test_policy_opening_releases_the_assist(GripAssist):
    a = GripAssist(THRESH, **KW)
    drive(a, [(CLOSED, FREE)] * 40)
    out = a.update(REF, (10.0, 10.0), step=99, meas=REF)
    assert a.state == "IDLE" and a.offset == 0.0 and out == REF


def test_moving_close_never_engages(GripAssist):
    """Mid-approach the command is still changing: hands off."""
    a = GripAssist(THRESH, **KW)
    moving = [((0.42 + 0.03 * i, 0.32 + 0.03 * i), FREE) for i in range(12)]
    r = drive(a, moving)
    assert a.state == "IDLE" and a.offset == 0.0
    assert all(abs(s[0][0] - c[0][0]) < 1e-9 for s, c in zip(r, moving))


def test_joint_limits_are_clamped(GripAssist):
    a = GripAssist(THRESH, ref=REF, min_close=0.15, stable_ticks=1,
                   stable_eps=0.01, step=0.5, max_extra=5.0, dwell_ticks=0,
                   confirm_ticks=2, lag=0.010)
    for sent, _, _ in drive(a, [((1.45, 2.0), FREE)] * 6):
        assert -1.484 - 1e-9 <= sent[0] <= 1.484 + 1e-9
        assert -2.025 - 1e-9 <= sent[1] <= 2.025 + 1e-9


def test_legacy_negative_closing_model(GripAssist):
    """Pre-flip datasets close proximal NEGATIVE. The assist must engage and
    extend that way too — a signed 'closing' test would never fire, and the
    server-side PROXIMAL_CMD_SIGN bridge never reaches this client-side logic."""
    leg_ref, leg_closed = (0.0, 0.0), (-0.45, 0.02)
    a = GripAssist(THRESH, ref=leg_ref, min_close=0.15, stable_ticks=3,
                   stable_eps=0.01, step=0.05, max_extra=0.30, dwell_ticks=1,
                   confirm_ticks=2, lag=0.010)
    r = drive(a, [(leg_closed, FREE)] * 8)
    assert a.state == "ASSISTING"
    assert (np.array(r[-1][0]) - np.array(leg_closed))[0] < 0


def test_falls_back_to_load_only_without_position(GripAssist):
    """Callers that cannot supply a measured position still get a (weaker)
    load-only test rather than silently never triggering."""
    a = GripAssist(THRESH, **KW)
    for i in range(8):
        a.update(CLOSED, (200.0, 200.0), step=i)   # meas omitted
    assert a.state == "GRIPPED"


def test_every_gate_reports_a_reason(GripAssist):
    """--grip_assist logs WHY it acted or not; a silent no-show on the arm is
    undebuggable, so every gate must set `why`."""
    a = GripAssist(THRESH, **{**KW, "max_extra": 0.20, "squeeze": 0.06})
    seen = set()

    def run(seq):
        for i, (cmd, kind) in enumerate(seq):
            base = np.asarray(a._sent if a._sent is not None else cmd, float)
            meas = tuple(base - kind["lag"])
            if a._meas is not None and kind.get("adv"):
                meas = tuple(np.asarray(a._meas) + kind["adv"])
            a.update(cmd, (kind["load"],) * 2, step=i, meas=meas)
            seen.add(a.why.split(" (")[0].split(" —")[0])

    run([(REF, FREE)] * 2)
    run([(CLOSED, FREE)] * 2)
    run([(CLOSED, FREE)] * 12)
    run([(CLOSED, FREE)] * 3)
    run([(REF, FREE)] * 2 + [(CLOSED, FREE)] * 4 + [(CLOSED, STUCK)] * 8)
    run([(CLOSED, FREE)] * 4)

    for expected in ("open", "cmd still moving", "dwell", "stepping to",
                     "no contact within max_extra", "disarmed", "contact confirmed",
                     "squeezing to", "latched at", "holding", "grip lost?", "slip"):
        assert any(s.startswith(expected) for s in seen), f"no diagnostic for {expected!r}"
