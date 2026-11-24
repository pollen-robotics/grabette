"""Rustypot wrapper for two Feetech STS3215 servos.

Architecture: owner-thread pattern. A single background thread owns the serial
bus and drives a fixed-rate control cycle (read positions, apply pending goal
/ torque). All RPC handlers interact only with in-memory slots, so they never
block on serial I/O.

This decouples the camera streaming rate from the motor bus rate and keeps
SendMotorCommand latency at sub-millisecond regardless of how often a
sync_read_present_position reply is dropped by the STS3215.

Falls back to a mock (simulated positions) when rustypot is unavailable.

Rustypot API:
  controller = Sts3215PyController(serial_port, baudrate, timeout)
  controller.sync_read_present_position(ids)  -> list[float]  (radians)
  controller.sync_write_goal_position(ids, values)
  controller.sync_write_torque_enable(ids, values)
"""

import logging
import math
import serial
import threading
import time

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 0.5  # seconds

# STS3215 replies take <2 ms at 1 Mbps. A 50 ms read timeout is ~25x the
# normal reply time — plenty of margin for a healthy bus, and a dropped
# reply costs at most 50 ms instead of stalling the bus thread for a full
# second.
DEFAULT_SERIAL_TIMEOUT = 0.05

# Bus thread control rate. 100 Hz = one read + one optional write per 10 ms,
# which is well within the STS3215's capability at 1 Mbps and leaves plenty
# of headroom. The cache is refreshed faster than the 10 Hz StreamState
# consumer, so clients never see stale data.
DEFAULT_BUS_HZ = 100.0

# present_load is force telemetry / feedback, not the position control signal,
# so it doesn't need the full bus rate. Reading it costs one extra sync round
# trip; do it every Nth tick to keep the position read on the hot path.
LOAD_READ_DECIMATION = 2

# STS3215 running (RAM) torque-limit register range: 0..1000 = 0..100% of the
# servo's max torque. This is the *running* limit, NOT the EEPROM max-torque
# register — the running one is safe to write repeatedly.
TORQUE_LIMIT_MAX = 1000

# present_load's direction bit gives the hardware effort sign. We map it to the
# robot convention — positive = CLOSING effort, matching "positive angle closes"
# (true for both left and right gripette) — by applying the per-motor position
# `sign` (so left/right read consistently, exactly like the angle handling)
# plus this calibrated polarity. The sign is PRESERVED (no abs): an external
# force driving the joint the other way flips it, which is meaningful.
# Calibrated on a RIGHT unit (commanded close drives decoded load negative, so
# we flip). VERIFY this polarity on a LEFT unit before trusting its load sign.
LOAD_CLOSING_SIGN = -1

# --- Servo-side zero re-centring (STS3215 EEPROM "Ofs" register) ----------
#
# The STS3215 position register spans one turn, 0..4095 (rustypot maps it to
# -pi..+pi rad, 2048 = 0 rad), and in position mode a goal cannot cross the
# 0/4095 seam. If the gripper's open pose lands near that seam (encoder
# reading close to +-pi), the closing travel runs into it: an open pose at
# +2.908 rad leaves only pi - 2.908 = 13 deg of travel, and the motor stops
# there whatever goal it is sent.
#
# The servo's offset register shifts its own position frame (present = actual
# - Ofs, goals and the seam move with it), so writing it moves the open pose
# to mid-range (2048) without remounting the horn. The software offset
# (GRIPPER_MOTOR*_OFFSET) then only carries the residual of a few steps.
STS_STEPS_PER_TURN = 4096
STS_CENTER_STEPS = 2048          # raw position read as 0 rad by rustypot
STS_ADDR_OFFSET = 31             # EEPROM, 2 bytes, sign-magnitude (bit 11)
STS_ADDR_LOCK = 55               # 0 = EEPROM writable, 1 = locked
STS_OFFSET_SIGN_BIT = 1 << 11
STS_OFFSET_MAX = STS_OFFSET_SIGN_BIT - 1   # +-2047 steps
# An open pose already within this of mid-range is left alone (saves an EEPROM
# write); the post-write check also accepts this much (the fingers are held by
# hand while it runs). 32 steps = 2.8 deg, against >=64 deg of spare travel.
CENTER_TOLERANCE_STEPS = 32
CENTER_TIMEOUT_S = 5.0


def _flat(value) -> list[int]:
    """rustypot single-servo reads come back wrapped in a list (one entry per
    servo, e.g. [2031] or [[lo, hi]]); flatten to the bare register values."""
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in _flat(item)]
    return [int(value)]


def _single(value) -> int:
    """One register value from a single-servo read (see _flat)."""
    (v,) = _flat(value)
    return v


def closing_travel(offset: float, sign: int) -> float:
    """Robot-frame closing travel (rad) left before the encoder seam, for a
    software offset/sign: goals go to encoder = robot * sign + offset, which
    must stay within +-pi."""
    return math.pi - offset * sign


def travel_problem(offsets, signs, maxima) -> str | None:
    """Explain why calibration offsets cannot reach the closing limits, or None.

    Guards the calibration: an offset leaving less travel than the joint limit
    looks fine at rest and only shows up as a gripper that stops closing early.
    """
    short = []
    for i, (off, sign, lim) in enumerate(zip(offsets, signs, maxima), start=1):
        travel = closing_travel(off, sign)
        if travel < lim:
            short.append(f"motor {i} can only close {math.degrees(travel):.1f} deg "
                         f"of {math.degrees(lim):.1f} deg (offset {off:+.3f} rad)")
    if not short:
        return None
    return ("; ".join(short) + ". The open pose sits too close to the servo's "
            "encoder seam: re-centre the servo zero (CenterServos) and calibrate again.")


def encode_sts_offset(steps: int) -> int:
    """Signed steps -> STS3215 Ofs register value (sign-magnitude, bit 11)."""
    if not -STS_OFFSET_MAX <= steps <= STS_OFFSET_MAX:
        raise ValueError(f"servo offset {steps} outside +-{STS_OFFSET_MAX} steps")
    return (STS_OFFSET_SIGN_BIT | -steps) if steps < 0 else steps


def decode_sts_offset(raw: int) -> int:
    """STS3215 Ofs register value -> signed steps."""
    mag = raw & STS_OFFSET_MAX
    return -mag if raw & STS_OFFSET_SIGN_BIT else mag


def centering_offset(present_raw: int, current_offset: int) -> int:
    """Servo offset that makes the current pose read STS_CENTER_STEPS.

    present = actual - offset (mod one turn), so actual = present + offset and
    the wanted offset is actual - center, wrapped to a single turn. -2048 and
    +2048 are the same turn but neither fits the 11-bit magnitude; 2047 is one
    step off, which the software offset absorbs.
    """
    actual = present_raw + current_offset
    half = STS_STEPS_PER_TURN // 2
    wanted = (actual - STS_CENTER_STEPS + half) % STS_STEPS_PER_TURN - half
    return max(-STS_OFFSET_MAX, min(STS_OFFSET_MAX, wanted))

try:
    from rustypot import Sts3215PyController

    _HAS_RUSTYPOT = True
except ImportError:
    _HAS_RUSTYPOT = False


class MotorController:
    """Owner-thread controller for two STS3215 servos on a shared serial bus.

    Public API (all non-blocking w.r.t. the serial bus):
      - read_positions(): atomic read of the cached last-known positions.
      - write_goal_positions(pos1, pos2): atomic write to a pending-goal slot.
      - set_torque(enable): atomic write to a pending-torque slot.

    A dedicated background thread is the only code that talks to the serial
    bus. It runs at DEFAULT_BUS_HZ and, on each tick:
      1. reads positions and updates the cache,
      2. applies a pending goal write if one has been deposited,
      3. applies a pending torque write if one has been deposited.

    If a bus operation raises, it's logged and skipped — next tick retries.
    """

    def __init__(
        self,
        port: str = "/dev/ttyS0",
        baudrate: int = 1_000_000,
        id_1: int = 1,
        id_2: int = 2,
        timeout: float = DEFAULT_SERIAL_TIMEOUT,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
        signs: tuple[int, int] = (1, 1),
        offsets: tuple[float, float] = (0.0, 0.0),
        bus_hz: float = DEFAULT_BUS_HZ,
    ):
        self.port = port
        self.baudrate = baudrate
        self.ids = [id_1, id_2]
        self.timeout = timeout
        # limits: ((m1_min, m1_max), (m2_min, m2_max)) in radians, ROBOT FRAME
        # (0 = open, positive = closing).
        self.limits = limits
        # Robot frame <-> encoder frame mapping (per motor):
        #   read:  robot = (encoder - offset) * sign
        #   write: encoder = robot * sign + offset
        # signs are ±1 (hardware mounting); offsets are the encoder reading
        # observed at robot-frame zero (calibration). Cached/limit-checked
        # values are therefore always in robot frame (positive = closing).
        self.signs = signs
        self.offsets = offsets
        self._bus_period = 1.0 / bus_hz
        self._controller = None
        self._mock = not _HAS_RUSTYPOT

        # Cached state (read by RPC threads, written by bus thread).
        # Tuple reassignment is atomic under the GIL, but we use a lock to
        # keep the intent explicit and to allow replacing the GIL guarantee
        # with a free-threaded build later without silent breakage.
        self._pos_lock = threading.Lock()
        self._cached_positions: tuple[float, float] = (0.0, 0.0)

        # Cached present-load (decoded signed effort), refreshed by the bus
        # thread at a decimated rate. Separate lock so a load read never
        # contends with the higher-rate position read.
        self._load_lock = threading.Lock()
        self._cached_load: tuple[float, float] = (0.0, 0.0)
        self._load_tick = 0

        # Pending write slots (written by RPC threads, consumed by bus thread).
        self._slot_lock = threading.Lock()
        self._pending_goal: tuple[float, float] | None = None
        self._pending_torque: bool | None = None
        # Torque-limit is a per-grasp write: deposit as raw register units,
        # applied only when it changed (dedup) so we never write per-tick.
        self._pending_torque_limit: tuple[int, int] | None = None
        self._last_torque_limit: tuple[int, int] | None = None
        # Servo re-centring request: (done Event, result holder), run by the
        # bus thread because it is the only code allowed on the serial bus.
        self._pending_center: tuple[threading.Event, dict] | None = None
        # Set once a re-centring changed a servo's frame: the software offsets
        # this process runs with no longer match the encoder, so goals and
        # torque-on are refused until the offsets are recalibrated and the
        # service restarted. Reads keep working (calibration needs them).
        self._frame_changed = False

        # Bus thread lifecycle.
        self._running = False
        self._thread: threading.Thread | None = None
        self._read_fail_count = 0

        # Mock state (when rustypot is unavailable).
        self._mock_positions: list[float] = [0.0, 0.0]

    @staticmethod
    def _flush_serial(port: str, baudrate: int) -> None:
        """Flush any stale data left in the serial buffer (e.g. from boot console)."""
        try:
            ser = serial.Serial(port, baudrate, timeout=0.1)
            discarded = ser.read(4096)
            ser.close()
            if discarded:
                logger.info("Flushed %d stale bytes from %s", len(discarded), port)
        except Exception as e:
            logger.warning("Could not flush serial port %s: %s", port, e)

    def start(self) -> None:
        if self._mock:
            logger.warning("rustypot not available — using mock motors")
            return

        # Flush stale data before opening the controller
        self._flush_serial(self.port, self.baudrate)

        for attempt in range(1, MAX_RETRIES + 1):
            self._controller = Sts3215PyController(
                self.port,
                self.baudrate,
                self.timeout,
            )
            try:
                pos = self._controller.sync_read_present_position(self.ids)
                # Encoder -> robot frame: subtract offset, then apply sign.
                # Seed the cache so clients don't see (0, 0) before the
                # first bus-loop tick.
                robot = (
                    (pos[0] - self.offsets[0]) * self.signs[0],
                    (pos[1] - self.offsets[1]) * self.signs[1],
                )
                with self._pos_lock:
                    self._cached_positions = robot
                logger.info(
                    "Motors started on %s: ids=%s, encoder=%s, robot=%s, "
                    "signs=%s, offsets=%s",
                    self.port, self.ids, pos, robot, self.signs, self.offsets,
                )
                break  # success
            except RuntimeError as e:
                logger.warning(
                    "Motor communication attempt %d/%d on %s failed: %s",
                    attempt,
                    MAX_RETRIES,
                    self.port,
                    e,
                )
                self._controller = None
                if attempt < MAX_RETRIES:
                    self._flush_serial(self.port, self.baudrate)
                    time.sleep(RETRY_DELAY)
        else:
            logger.error(
                "Motor communication failed on %s after %d attempts — falling back to mock",
                self.port,
                MAX_RETRIES,
            )
            self._mock = True
            return

        # Launch the bus-owner thread.
        self._running = True
        self._thread = threading.Thread(
            target=self._bus_loop, name="MotorBusLoop", daemon=True
        )
        self._thread.start()
        logger.info("Motor bus thread started at %.1f Hz", 1.0 / self._bus_period)

    def read_positions(self) -> tuple[float, float]:
        """Return the most recent cached positions in radians (ROBOT FRAME:
        0 = open, positive = closing). Non-blocking."""
        if self._mock:
            return (self._mock_positions[0], self._mock_positions[1])
        with self._pos_lock:
            return self._cached_positions

    # Tolerance on the limit check, radians (~0.006 deg — far below the 0.088 deg
    # encoder step, so it cannot mask a real out-of-range command).
    #
    # Needed because goals arrive over gRPC as proto `float` (float32) while the
    # limits are float64. A client legitimately asking for EXACTLY the limit —
    # which is what a full close does — sends a value that quantises to marginally
    # above it and was rejected outright: "goal 1.632 rad outside limits
    # [0.000, 1.632]". Observed on hardware. Values inside the tolerance are
    # clamped to the limit rather than refused.
    _LIMIT_TOL = 1e-4

    def _check_limits(self, pos1: float, pos2: float) -> tuple[float, float]:
        """Validate goals against the configured limits.

        Returns the (possibly clamped) goals. Raises ValueError only for values
        genuinely outside the range, not for float32 rounding at the boundary.
        """
        if self.limits is None:
            return pos1, pos2
        (m1_min, m1_max), (m2_min, m2_max) = self.limits
        if not (m1_min - self._LIMIT_TOL <= pos1 <= m1_max + self._LIMIT_TOL):
            raise ValueError(
                f"Motor 1 goal {pos1:.3f} rad outside limits [{m1_min:.3f}, {m1_max:.3f}]"
            )
        if not (m2_min - self._LIMIT_TOL <= pos2 <= m2_max + self._LIMIT_TOL):
            raise ValueError(
                f"Motor 2 goal {pos2:.3f} rad outside limits [{m2_min:.3f}, {m2_max:.3f}]"
            )
        return (min(max(pos1, m1_min), m1_max), min(max(pos2, m2_min), m2_max))

    def write_goal_positions(self, pos1: float, pos2: float) -> None:
        """Queue a goal write for the bus thread. Non-blocking. Validates limits.

        Arguments are in ROBOT FRAME (0 = open, positive = closing). Limits
        are checked in robot frame, then the goal is converted to the encoder
        frame via `encoder = robot * sign + offset` before being queued for
        the bus thread.

        Last-wins: if a previous goal hasn't been applied yet, it's overwritten.
        """
        self._refuse_if_frame_changed("goal")
        # Clamped, not just checked: a goal exactly AT the limit arrives from the
        # wire as float32 and can land a hair outside it.
        pos1, pos2 = self._check_limits(pos1, pos2)
        if self._mock:
            # Mock keeps robot-frame state to match read_positions() semantics.
            self._mock_positions = [pos1, pos2]
            logger.debug("Mock motors → goals: (%.3f, %.3f)", pos1, pos2)
            return
        # Robot frame -> encoder frame just before crossing the bus boundary.
        encoder_goal = (
            pos1 * self.signs[0] + self.offsets[0],
            pos2 * self.signs[1] + self.offsets[1],
        )
        with self._slot_lock:
            self._pending_goal = encoder_goal

    def set_torque(self, enable: bool) -> None:
        """Queue a torque-enable write for the bus thread. Non-blocking.

        Last-wins: a previous pending torque request is overwritten.
        """
        if enable:
            self._refuse_if_frame_changed("torque on")
        if self._mock:
            logger.debug("Mock motors → torque %s", "on" if enable else "off")
            return
        with self._slot_lock:
            self._pending_torque = enable

    def set_torque_limit(self, values: list[float]) -> None:
        """Queue a RUNNING (RAM) torque-limit write. Non-blocking, last-wins.

        `values` are fractions 0.0–1.0 of the servo's max torque, one per
        motor. The bus thread applies it only when it changed (dedup), so this
        is a per-grasp write, never per-tick. The fraction→register mapping
        lives here so callers never see raw units. Hits the RAM running-torque
        register (safe to rewrite), NOT the EEPROM max-torque register.
        """
        if len(values) != len(self.ids):
            raise ValueError(
                f"expected {len(self.ids)} torque-limit values, got {len(values)}"
            )
        raw = tuple(
            int(round(max(0.0, min(1.0, v)) * TORQUE_LIMIT_MAX)) for v in values
        )
        if self._mock:
            logger.debug("Mock motors → torque_limit %s", raw)
            return
        with self._slot_lock:
            self._pending_torque_limit = raw

    def _refuse_if_frame_changed(self, what: str) -> None:
        if self._frame_changed:
            raise RuntimeError(
                f"{what} refused: the servo zero was re-centred, so the "
                "calibration offsets in use are stale. Finish the calibration "
                "(record the new offsets) and restart the gripette service."
            )

    def center_servo_offsets(self, timeout: float = CENTER_TIMEOUT_S) -> dict:
        """Re-centre each servo's own zero on the CURRENT pose (EEPROM write).

        Meant for zero calibration only, with the gripper held fully open and
        torque off: afterwards that pose reads ~0 rad on the raw encoder, so
        the full closing travel sits away from the 0/4095 seam. The software
        offsets must then be recalibrated (the existing calibration math does
        it: it reads the pose through the old offsets and absorbs the shift),
        and the service restarted; until then goals and torque-on are refused.

        Torque is turned off first. A servo already within
        CENTER_TOLERANCE_STEPS of mid-range is not rewritten.

        Returns {"changed": bool, "old_offsets": [..], "new_offsets": [..],
        "present": [..]} (steps). Raises RuntimeError if the bus fails or the
        servo does not read mid-range after the write (the old offset is
        restored in that case).
        """
        if self._mock:
            return {"changed": False, "old_offsets": [0, 0],
                    "new_offsets": [0, 0], "present": [STS_CENTER_STEPS] * 2}
        if self._thread is None:
            raise RuntimeError("motor bus is not running")
        done, result = threading.Event(), {}
        with self._slot_lock:
            self._pending_center = (done, result)
        if not done.wait(timeout):
            raise RuntimeError("servo re-centring timed out")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["value"]

    def _do_center(self) -> dict:
        """Bus-thread body of center_servo_offsets(). Only the bus thread may
        call this."""
        c = self._controller
        c.sync_write_torque_enable(self.ids, [False] * len(self.ids))
        time.sleep(0.05)
        old, new, present = [], [], []
        for motor_id in self.ids:
            lo, hi = _flat(c.read_raw_data(motor_id, STS_ADDR_OFFSET, 2))
            cur = decode_sts_offset(lo | (hi << 8))
            pos = _single(c.read_raw_present_position(motor_id))
            old.append(cur)
            present.append(pos)
            if abs(pos - STS_CENTER_STEPS) <= CENTER_TOLERANCE_STEPS:
                new.append(cur)
                continue
            wanted = centering_offset(pos, cur)
            self._write_sts_offset(motor_id, wanted)
            time.sleep(0.05)
            after = _single(c.read_raw_present_position(motor_id))
            if abs(after - STS_CENTER_STEPS) > CENTER_TOLERANCE_STEPS:
                self._write_sts_offset(motor_id, cur)  # roll back
                raise RuntimeError(
                    f"servo {motor_id}: offset {cur} -> {wanted} steps should "
                    f"read {STS_CENTER_STEPS}, read {after} (was {pos}); "
                    "offset restored. Was the gripper held still?"
                )
            # From here this servo's frame changed: refuse goals in the old one.
            self._frame_changed = True
            # Park the goal register on the new reading, so a later torque-on
            # cannot jump to a goal expressed in the old frame.
            c.write_raw_goal_position(motor_id, after)
            new.append(wanted)
            present[-1] = after
            logger.warning(
                "Servo %d zero re-centred: offset %d -> %d steps (read %d -> %d)",
                motor_id, cur, wanted, pos, after,
            )
        # Refresh the cache before the caller is released: the calibration
        # samples positions right after this returns, and a pre-shift reading
        # would corrupt its average.
        pos = c.sync_read_present_position(self.ids)
        with self._pos_lock:
            self._cached_positions = (
                (pos[0] - self.offsets[0]) * self.signs[0],
                (pos[1] - self.offsets[1]) * self.signs[1],
            )
        return {"changed": old != new, "old_offsets": old,
                "new_offsets": new, "present": present}

    def _write_sts_offset(self, motor_id: int, steps: int) -> None:
        raw = encode_sts_offset(steps)
        c = self._controller
        c.write_raw_data(motor_id, STS_ADDR_LOCK, [0])
        try:
            c.write_raw_data(motor_id, STS_ADDR_OFFSET, [raw & 0xFF, raw >> 8])
        finally:
            c.write_raw_data(motor_id, STS_ADDR_LOCK, [1])

    @staticmethod
    def _decode_load(raw: int) -> float:
        """Decode Feetech present_load's register layout: bit 10 = direction,
        bits 0–9 = magnitude (0–1000, ~PWM effort, clamped at torque_limit).
        Returns the hardware-frame signed value; robot-frame mapping (sign
        convention) is applied by _load_to_robot."""
        mag = raw & 0x3FF
        return float(-mag if (raw & 0x400) else mag)

    def _load_to_robot(self, raw: int, idx: int) -> float:
        """Decode present_load and map to ROBOT FRAME: positive = closing
        effort (matching the angle convention), left/right-consistent via the
        per-motor `sign`. Sign preserved — a force driving the joint the other
        way flips it. See LOAD_CLOSING_SIGN."""
        return self._decode_load(raw) * self.signs[idx] * LOAD_CLOSING_SIGN

    def get_present_load(self) -> list[float]:
        """Most recent present-load per motor, ROBOT FRAME (positive = closing
        effort, left/right consistent; sign preserved). Non-blocking; refreshed
        by the bus thread at a decimated rate."""
        if self._mock:
            return [0.0] * len(self.ids)
        with self._load_lock:
            return list(self._cached_load)

    def _bus_loop(self) -> None:
        """Single owner of the serial bus. Reads positions, applies pending writes."""
        while self._running:
            tick = time.monotonic()

            # 1. Read positions → convert encoder -> robot frame → update cache.
            #    robot = (encoder - offset) * sign
            try:
                pos = self._controller.sync_read_present_position(self.ids)
                with self._pos_lock:
                    self._cached_positions = (
                        (pos[0] - self.offsets[0]) * self.signs[0],
                        (pos[1] - self.offsets[1]) * self.signs[1],
                    )
                # Reset fail counter on success to reduce log noise.
                self._read_fail_count = 0
            except Exception as e:
                self._read_fail_count += 1
                # Log first failure immediately, then throttle (every 50th).
                if self._read_fail_count == 1 or self._read_fail_count % 50 == 0:
                    logger.warning(
                        "Motor read failed (%d consecutive): %s",
                        self._read_fail_count,
                        e,
                    )

            # 1b. Read present_load at a decimated rate (force telemetry, not
            #     the position control signal). A dropped load read is
            #     non-critical — just skip it.
            self._load_tick += 1
            if self._load_tick % LOAD_READ_DECIMATION == 0:
                try:
                    raw = self._controller.sync_read_present_load(self.ids)
                    with self._load_lock:
                        self._cached_load = (
                            self._load_to_robot(raw[0], 0),
                            self._load_to_robot(raw[1], 1),
                        )
                except Exception:
                    pass

            # 2. Drain pending write slots under the lock, then apply outside it.
            with self._slot_lock:
                goal = self._pending_goal
                self._pending_goal = None
                torque_req = self._pending_torque
                self._pending_torque = None
                tl_req = self._pending_torque_limit
                self._pending_torque_limit = None
                center_req = self._pending_center
                self._pending_center = None

            if center_req is not None:
                # Exclusive: a goal or torque-on queued alongside would be in
                # the frame being replaced. Drop them.
                goal, torque_req = None, None  # _do_center turns torque off
                done, result = center_req
                try:
                    result["value"] = self._do_center()
                except Exception as e:
                    logger.exception("Servo re-centring failed")
                    result["error"] = str(e)
                finally:
                    done.set()

            if goal is not None:
                try:
                    self._controller.sync_write_goal_position(
                        self.ids, [goal[0], goal[1]]
                    )
                except Exception as e:
                    logger.warning("Motor goal write failed: %s", e)

            if torque_req is not None:
                try:
                    self._controller.sync_write_torque_enable(
                        self.ids, [torque_req, torque_req]
                    )
                    logger.info(
                        "Motors torque %s", "enabled" if torque_req else "disabled"
                    )
                except Exception as e:
                    logger.warning("Motor torque write failed: %s", e)

            # Apply torque-limit only when it changed (per-grasp, not per-tick).
            if tl_req is not None and tl_req != self._last_torque_limit:
                try:
                    self._controller.sync_write_torque_limit(
                        self.ids, [tl_req[0], tl_req[1]]
                    )
                    self._last_torque_limit = tl_req
                    logger.info(
                        "Motors torque_limit set to %s /%d", tl_req, TORQUE_LIMIT_MAX
                    )
                except Exception as e:
                    logger.warning("Torque-limit write failed: %s", e)

            # 3. Sleep to the next tick.
            elapsed = time.monotonic() - tick
            sleep_for = self._bus_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._controller is not None:
            self._controller = None
            logger.info("Motors stopped")
