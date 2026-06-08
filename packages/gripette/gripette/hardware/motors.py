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
        bus_hz: float = DEFAULT_BUS_HZ,
    ):
        self.port = port
        self.baudrate = baudrate
        self.ids = [id_1, id_2]
        self.timeout = timeout
        # limits: ((m1_min, m1_max), (m2_min, m2_max)) in radians
        self.limits = limits
        self._bus_period = 1.0 / bus_hz
        self._controller = None
        self._mock = not _HAS_RUSTYPOT

        # Cached state (read by RPC threads, written by bus thread).
        # Tuple reassignment is atomic under the GIL, but we use a lock to
        # keep the intent explicit and to allow replacing the GIL guarantee
        # with a free-threaded build later without silent breakage.
        self._pos_lock = threading.Lock()
        self._cached_positions: tuple[float, float] = (0.0, 0.0)

        # Pending write slots (written by RPC threads, consumed by bus thread).
        self._slot_lock = threading.Lock()
        self._pending_goal: tuple[float, float] | None = None
        self._pending_torque: bool | None = None

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
                # Seed the cache so clients don't see (0, 0) before the first tick.
                with self._pos_lock:
                    self._cached_positions = (pos[0], pos[1])
                logger.info(
                    "Motors started on %s: ids=%s, positions=%s",
                    self.port,
                    self.ids,
                    pos,
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
        """Return the most recent cached positions in radians. Non-blocking."""
        if self._mock:
            return (self._mock_positions[0], self._mock_positions[1])
        with self._pos_lock:
            return self._cached_positions

    def _check_limits(self, pos1: float, pos2: float) -> None:
        """Raise ValueError if positions are outside configured limits."""
        if self.limits is None:
            return
        (m1_min, m1_max), (m2_min, m2_max) = self.limits
        if not (m1_min <= pos1 <= m1_max):
            raise ValueError(
                f"Motor 1 goal {pos1:.3f} rad outside limits [{m1_min:.3f}, {m1_max:.3f}]"
            )
        if not (m2_min <= pos2 <= m2_max):
            raise ValueError(
                f"Motor 2 goal {pos2:.3f} rad outside limits [{m2_min:.3f}, {m2_max:.3f}]"
            )

    def write_goal_positions(self, pos1: float, pos2: float) -> None:
        """Queue a goal write for the bus thread. Non-blocking. Validates limits.

        Last-wins: if a previous goal hasn't been applied yet, it's overwritten.
        """
        self._check_limits(pos1, pos2)
        if self._mock:
            self._mock_positions = [pos1, pos2]
            logger.debug("Mock motors → goals: (%.3f, %.3f)", pos1, pos2)
            return
        with self._slot_lock:
            self._pending_goal = (pos1, pos2)

    def set_torque(self, enable: bool) -> None:
        """Queue a torque-enable write for the bus thread. Non-blocking.

        Last-wins: a previous pending torque request is overwritten.
        """
        if self._mock:
            logger.debug("Mock motors → torque %s", "on" if enable else "off")
            return
        with self._slot_lock:
            self._pending_torque = enable

    def _bus_loop(self) -> None:
        """Single owner of the serial bus. Reads positions, applies pending writes."""
        while self._running:
            tick = time.monotonic()

            # 1. Read positions → update cache.
            try:
                pos = self._controller.sync_read_present_position(self.ids)
                with self._pos_lock:
                    self._cached_positions = (pos[0], pos[1])
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

            # 2. Drain pending write slots under the lock, then apply outside it.
            with self._slot_lock:
                goal = self._pending_goal
                self._pending_goal = None
                torque_req = self._pending_torque
                self._pending_torque = None

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
