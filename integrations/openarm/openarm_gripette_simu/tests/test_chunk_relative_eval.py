"""Tests for evaluate.py's chunk-relative wiring.

The interesting ones drive the REAL run_episode() loop against fake gRPC stubs
rather than re-implementing its decisions, because the thing most likely to be
wrong is *when* the chunk reference resets — and a test that mirrors that logic
would agree with a broken loop. Everything run_episode needs is already
injected, so faking it needs no seams that do not exist.

Needs the `eval` extra (evaluate.py imports lerobot/torch).
"""

import importlib.util
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_EVALUATE = Path(__file__).resolve().parents[1] / "examples" / "evaluate.py"

torch = pytest.importorskip("torch")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
chunkrel = pytest.importorskip("grabette_chunkrel.chunk_relative")


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("_evaluate", _EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K = 6  # actions per chunk; short so the test stays quick


def make_chunks(n_chunks=3, seed=0):
    """n_chunks absolute trajectories, each encoded chunk-relative."""
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n_chunks):
        pos = np.cumsum(r.normal(scale=0.004, size=(K, 3)), axis=0) + np.array([1.0, 0.5, 0.3])
        rot = np.cumsum(r.normal(scale=0.008, size=(K, 3)), axis=0) + np.array([0.2, -0.9, 1.1])
        grip = r.uniform(0.05, 0.95, size=(K, 2))
        absolute = np.concatenate([pos, rot, grip], axis=1)
        rel, rep = chunkrel.to_chunk_relative(absolute)
        assert rep["n_jumps_spliced"] == 0 and rep["n_rot_jumps_spliced"] == 0
        out.append(rel)
    return out


# ── fakes ───────────────────────────────────────────────────────────────

class FakePolicy:
    """pi05's select_action() contract: refill the queue from the next chunk when
    it runs dry, then popleft. Including the dead `_queues` attribute pi05 also
    carries, so the test would catch a helper that reads the wrong one."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._served = 0
        self._action_queue = deque(maxlen=K)
        self._queues = {"action": deque(maxlen=K)}   # created, never drained
        self.config = SimpleNamespace(n_action_steps=K)

    def select_action(self, batch):
        if not self._action_queue:
            chunk = self._chunks[min(self._served, len(self._chunks) - 1)]
            self._served += 1
            self._action_queue.extend(torch.from_numpy(np.asarray(a, np.float32))
                                      for a in chunk)
        return self._action_queue.popleft().unsqueeze(0)


class FakeArm:
    def __init__(self):
        self.deltas = []
        self.joint_cmds = []

    def SendCartesianDelta(self, msg):
        self.deltas.append((np.array([msg.dx, msg.dy, msg.dz]), np.array(msg.dr6d)))
        return SimpleNamespace(success=True)

    def SendJointCommand(self, msg):
        self.joint_cmds.append(msg)
        return SimpleNamespace(success=True)

    def GetSuccessStatus(self, _req):
        return SimpleNamespace(goal_reached=False, cube_displacement=0.0)

    def GetArmState(self, _req):
        return SimpleNamespace(joint_positions=[0.0] * 7)


class FakeGripper:
    def __init__(self):
        self.goals = []

    def SendMotorCommand(self, msg):
        self.goals.append((msg.motor1_goal, msg.motor2_goal))
        return SimpleNamespace(success=True)


class FakeCamera:
    """Returns a constant frame; the policy ignores it (it serves a fixed chunk),
    and the loop only needs the timestamp to be monotonic."""

    def __init__(self):
        self.t = 0.0

    def get(self):
        self.t += 20.0
        import time
        return (np.zeros((8, 8, 3), np.uint8), np.array([0.3, 0.2], np.float32),
                time.perf_counter() * 1000.0)


def fake_pb2():
    return SimpleNamespace(
        CartesianDelta=lambda dx, dy, dz, dr6d: SimpleNamespace(dx=dx, dy=dy, dz=dz, dr6d=dr6d),
        JointCommand=lambda joint_positions: SimpleNamespace(joint_positions=joint_positions),
        SuccessStatusRequest=lambda: None,
        GetArmStateRequest=lambda: None,
    )


def fake_grip_pb2():
    return SimpleNamespace(
        MotorCommand=lambda motor1_goal, motor2_goal, motor1_torque_limit,
        motor2_torque_limit: SimpleNamespace(
            motor1_goal=motor1_goal, motor2_goal=motor2_goal))


def drive(ev, chunks, n_chunks, grasp_projection=None, chunk_relative=True):
    arm, grip, cam = FakeArm(), FakeGripper(), FakeCamera()
    conv = chunkrel.ChunkRelativeDeltas() if chunk_relative else None
    result = ev.run_episode(
        policy=FakePolicy(chunks),
        preprocessor=lambda b: b,
        postprocessor=lambda a: a,
        arm_stub=arm, gripper_stub=grip,
        arm_pb2=fake_pb2(), gripper_pb2=fake_grip_pb2(),
        camera=cam, device=torch.device("cpu"),
        max_steps=K * n_chunks, fps=500.0, success_check_freq=1000,
        debug=False, use_relative_proprio=False,
        start_pos=np.zeros(3), start_rot=Rotation.identity(),
        task="pick up the object",
        grasp_projection=grasp_projection,
        chunk_relative=conv,
    )
    return arm, grip, result


def replay(deltas, start_pos, start_rot):
    """The arm server's own algebra (grpc_server_real.SendCartesianDelta)."""
    def r6d(v):
        a1, a2 = np.asarray(v[:3]), np.asarray(v[3:6])
        b1 = a1 / np.linalg.norm(a1)
        a2p = a2 - b1.dot(a2) * b1
        b2 = a2p / np.linalg.norm(a2p)
        return np.stack([b1, b2, np.cross(b1, b2)], axis=1)

    pos, R, out = np.asarray(start_pos, float).copy(), start_rot, []
    for dp, dr in deltas:
        pos = pos + R.as_matrix() @ dp
        R = Rotation.from_matrix(R.as_matrix() @ r6d(dr))
        out.append((pos.copy(), R))
    return out


# ── the loop, end to end ────────────────────────────────────────────────

def test_every_chunk_is_executed_relative_to_where_the_arm_is(ev):
    """The property the robot depends on, over MULTIPLE chunks.

    Each chunk carries its own reference pose, so a chunk is executed as a fresh
    relative segment anchored wherever the arm happens to be at replan time —
    that is the semantics, not a defect. Within a chunk, the arm's poses
    expressed in the chunk-start frame must equal the policy's offsets exactly.
    """
    n = 3
    chunks = make_chunks(n)
    arm, _grip, _res = drive(ev, chunks, n)
    assert len(arm.deltas) == K * n, f"expected {K * n} arm commands, got {len(arm.deltas)}"

    traj = replay(arm.deltas, np.array([5.0, -3.0, 2.0]),      # arbitrary arm frame
                  Rotation.from_rotvec([0.4, 1.1, -0.7]))
    prev_pos, prev_rot = np.array([5.0, -3.0, 2.0]), Rotation.from_rotvec([0.4, 1.1, -0.7])
    for c in range(n):
        seg = traj[c * K:(c + 1) * K]
        inv = prev_rot.inv()
        for i, (pos, R) in enumerate(seg):
            want = chunks[c][i]
            got_pos = inv.apply(pos - prev_pos)
            got_rot = inv * R
            assert np.abs(got_pos - want[:3]).max() < 1e-5, f"chunk {c} action {i} position"
            err = np.rad2deg((got_rot.inv() * Rotation.from_rotvec(want[3:6])).magnitude())
            assert err < 1e-3, f"chunk {c} action {i} rotation ({err:.4f} deg)"
        prev_pos, prev_rot = seg[-1]


def test_a_missing_reset_would_be_caught_by_that_test(ev):
    """Teeth check: with the reference deliberately never reset, chunk 2's first
    command is a large spurious jump instead of a small offset. Confirms the
    test above is actually sensitive to the boundary logic."""
    chunks = make_chunks(2, seed=3)
    conv = chunkrel.ChunkRelativeDeltas()
    correct = [conv.step(a)[0] for a in chunks[0]]
    conv.reset()
    correct += [conv.step(a)[0] for a in chunks[1]]

    conv2 = chunkrel.ChunkRelativeDeltas()
    never_reset = [conv2.step(a)[0] for a in chunks[0] + list(chunks[1])] \
        if isinstance(chunks[0], list) else \
        [conv2.step(a)[0] for a in np.concatenate([chunks[0], chunks[1]])]

    boundary = K
    assert np.linalg.norm(correct[boundary] - never_reset[boundary]) > 0.005, (
        "the boundary must matter, or the end-to-end test proves nothing"
    )


def test_the_gripper_channel_is_the_last_two_dims_not_a_rotation_component(ev):
    """An 8D action sliced as if it were 11D (a_np[9:]) would run off the end and
    send rotation numbers to the servo. Assert the servo saw the policy's gripper."""
    chunks = make_chunks(2)
    _arm, grip, _res = drive(ev, chunks, 2)
    want = [tuple(a[6:8]) for c in chunks for a in c]
    assert len(grip.goals) == len(want)
    for (g1, g2), (w1, w2) in zip(grip.goals, want):
        assert abs(g1 - w1) < 1e-5 and abs(g2 - w2) < 1e-5


def test_chunk_relative_and_the_grasp_projection_together(ev):
    """The combination Steve actually runs, and the one nothing had exercised:
    the motion is converted from chunk offsets while the gripper channels are
    decoded from (strategy, closure) to joint angles."""
    from gripette.grasp_projection import GraspProjection, clamp_to_command_limits

    proj = GraspProjection()
    chunks = make_chunks(2, seed=11)
    arm, grip, _res = drive(ev, chunks, 2, grasp_projection=proj)

    # gripper: decoded angles, not the raw (s, c)
    want = []
    for c in chunks:
        for a in c:
            g1, g2 = proj.decode(float(a[6]), float(a[7]))
            g1, g2, _ = clamp_to_command_limits(g1, g2)
            want.append((g1, g2))
    assert len(grip.goals) == len(want)
    for (g1, g2), (w1, w2) in zip(grip.goals, want):
        assert abs(g1 - w1) < 1e-5 and abs(g2 - w2) < 1e-5
    # a closure of ~1 must reach a real close, i.e. radians not a unit scalar
    assert max(g for g, _ in grip.goals) > 0.5, "decode did not produce joint angles"

    # motion: still exactly the chunk offsets
    traj = replay(arm.deltas[:K], np.zeros(3), Rotation.identity())
    for i, (pos, _R) in enumerate(traj):
        assert np.abs(pos - chunks[0][i][:3]).max() < 1e-5


# ── detection ───────────────────────────────────────────────────────────

def _write_cfg(tmp_path, steps):
    import json
    d = tmp_path / "ckpt"
    d.mkdir(exist_ok=True)
    (d / "policy_preprocessor.json").write_text(
        json.dumps({"name": "policy_preprocessor",
                    "steps": [{"registry_name": s, "config": {}} for s in steps]}))
    return str(d)


def test_detection_finds_our_step(ev, tmp_path):
    d = _write_cfg(tmp_path, ["to_batch_processor",
                              "grabette_chunk_relative_actions",
                              "normalizer_processor"])
    assert ev.detect_chunk_relative(d) is True


def test_detection_is_not_fooled_by_lerobots_own_relative_step(ev, tmp_path):
    """The baseline checkpoint carries lerobot's `relative_actions_processor`
    (disabled). Matching that would turn chunk-relative on for every existing
    checkpoint — this is the discrimination that matters."""
    d = _write_cfg(tmp_path, ["to_batch_processor", "relative_actions_processor",
                              "normalizer_processor"])
    assert ev.detect_chunk_relative(d) is False


def test_detection_gives_up_rather_than_guessing(ev, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ev.detect_chunk_relative(str(empty)) is None
    assert ev.detect_chunk_relative(None) is None
    assert ev.detect_chunk_relative("") is None


# ── the queue accessor (the bug this wiring depends on) ─────────────────

def test_the_live_queue_is_the_one_select_action_drains(ev):
    """pi05 defines both `_action_queue` (drained) and `_queues[ACTION]`
    (created in reset(), never touched). Reading the dead one reports "chunk
    boundary" on every single tick."""
    p = SimpleNamespace(_action_queue=deque([1, 2, 3]), _queues={"action": deque()})
    assert len(ev.policy_action_queue(p)) == 3


def test_act_style_policies_still_resolve(ev):
    p = SimpleNamespace(_queues={"action": deque([1, 2])})
    assert len(ev.policy_action_queue(p)) == 2
    assert ev.policy_action_queue(SimpleNamespace()) is None
