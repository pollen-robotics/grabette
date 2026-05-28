"""Real RPi hardware backend.

V2 (rgbd branch): RPi camera + AS5600 angle sensors + OAK-D SR.
The legacy BMI088 IMU was dropped — IMU data now comes from the OAK-D.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from grabette.backend.base import Backend
from grabette.config import settings
from grabette.models import AngleSample, CaptureStatus, IMUSample, SensorState

logger = logging.getLogger(__name__)

FPS = 46

# How long start_capture waits for the OAK-D to produce valid (post-warmup)
# frames before starting the recording clock. Safety fallback only — the OAK-D
# normally becomes ready well within this.
#
# TODO: this is a major contributor to the start-capture latency (cold OAK
# total ~8 s = ~3 s init + up to ~5 s here). On feature-rich scenes depth
# converges to >5% coverage in <500 ms, so 1.5 s would be ample. Lower this
# (and possibly min_depth_coverage in oakd.wait_until_ready) once we have
# empirical data showing it doesn't degrade SLAM quality on cold-boot frames.
# Data quality first, UX second — keep at 5 s until we measure.
OAKD_READY_TIMEOUT_S = 5.0


class RpiBackend(Backend):
    """Backend using real RPi camera + AS5600 angle sensors + OAK-D SR."""

    def __init__(
        self, enable_angle: bool = False, enable_oakd: bool = True,
        oakd_keepalive_s: float = 30.0,
    ) -> None:
        self._running = False
        self._start_time: float | None = None
        self._capturing = False
        self._capture_session_dir: Path | None = None
        self._enable_angle = enable_angle
        self._enable_oakd = enable_oakd
        self._oakd_keepalive_s = oakd_keepalive_s

        self._sync = None
        self._camera = None
        self._angle = None
        self._oakd = None
        # True when the OAK-D is on because a capture auto-enabled it (the
        # daemon owns its power and will auto-power-down when idle). Survives
        # back-to-back captures; cleared on power-down or when the user takes
        # ownership via the UI (set_oakd_enabled).
        self._oakd_auto_enabled = False
        # Pending "power the OAK-D down after the keep-alive window" timer.
        self._oakd_keepalive_task = None
        # Set when stop_capture defers hardware re-init out of the stop path;
        # start_capture then lazily re-inits the camera/angle (overlapped with
        # the OAK-D warmup) so re-init never delays the LED/stop.
        self._needs_reinit = False
        # Teleop mode (mutually exclusive with the recording-mode OakdCapture).
        # When teleop is active, _oakd is shut down and _teleop owns the OAK.
        self._teleop = None
        # Whether deltas should currently be marked send=True on the WS stream.
        # Reset to False whenever start_teleop() runs, so entering teleop
        # never immediately drives the robot.
        self._teleop_sending = False

    async def start(self) -> None:
        from grabette.hardware.sync import SyncManager
        from grabette.hardware.camera import VideoCapture

        self._sync = SyncManager()
        self._camera = VideoCapture(self._sync, fps=FPS)

        logger.info("Initializing camera...")
        self._camera.init_camera()

        if self._enable_angle:
            self._init_angle_sensors()

        if self._enable_oakd:
            self._init_oakd()

        self._running = True
        self._start_time = time.time()
        logger.info("RpiBackend started")

    def _init_oakd(self) -> None:
        """Initialize OAK-D SR (always-on pipeline, used for live view + recording)."""
        try:
            from grabette.hardware.oakd import OakdCapture
            self._oakd = OakdCapture(self._sync)
            self._oakd.init_device()
            logger.info("OAK-D SR initialized")
        except Exception as e:
            logger.warning("OAK-D not available, continuing without it: %s", e)
            self._oakd = None

    def _init_angle_sensors(self) -> None:
        try:
            from grabette.hardware.angle import AngleCapture
            self._angle = AngleCapture(self._sync)
            self._angle.init_sensors()
            logger.info("Angle sensors initialized")
        except Exception:
            logger.warning("Angle sensors not available, continuing without them")
            self._angle = None

    async def stop(self) -> None:
        if self._capturing:
            await self.stop_capture()
        # After any stop_capture (which may re-arm the keep-alive), drop the
        # pending power-down — we're shutting the OAK down directly below.
        self._cancel_oakd_keepalive()
        if self._teleop is not None:
            try:
                self._teleop.shutdown()
            except Exception as e:
                logger.warning("OakdTeleop shutdown error: %s", e)
            self._teleop = None
        if self._oakd:
            try:
                self._oakd.shutdown()
            except Exception as e:
                logger.warning("OAK-D shutdown error: %s", e)
        self._running = False
        self._start_time = None
        logger.info("RpiBackend stopped")

    # ── OAK-D runtime enable/disable (UI-driven, battery saver) ────────────────

    @property
    def is_oakd_enabled(self) -> bool:
        return self._enable_oakd

    @property
    def is_oakd_initialized(self) -> bool:
        return self._oakd is not None and self._oakd.is_initialized

    async def set_oakd_enabled(self, on: bool) -> None:
        if self._capturing:
            raise RuntimeError("cannot toggle OAK-D while a capture is running")
        if self.is_teleop_active:
            raise RuntimeError("cannot toggle OAK-D while teleop is active")

        on = bool(on)

        # An explicit enable/disable cancels any pending auto power-down and
        # hands power ownership to the caller (no auto-shutdown). start_capture
        # re-claims auto-ownership after its own enable call.
        self._cancel_oakd_keepalive()
        self._oakd_auto_enabled = False

        if on == self._enable_oakd and (on == self.is_oakd_initialized):
            return  # already in the requested state

        import asyncio
        loop = asyncio.get_event_loop()

        if on:
            self._enable_oakd = True
            await loop.run_in_executor(None, self._init_oakd)
            logger.info("OAK-D enabled via UI")
        else:
            self._enable_oakd = False
            if self._oakd is not None:
                try:
                    await loop.run_in_executor(None, self._oakd.shutdown)
                except Exception as e:
                    logger.warning("OAK-D shutdown error: %s", e)
                self._oakd = None
            logger.info("OAK-D disabled via UI")

    # ── OAK-D keep-alive (auto-power-down after a grace period) ────────────────

    def _cancel_oakd_keepalive(self) -> None:
        """Cancel a pending auto-power-down, if any."""
        task = self._oakd_keepalive_task
        self._oakd_keepalive_task = None
        if task is not None and not task.done():
            task.cancel()

    def _schedule_oakd_keepalive(self) -> None:
        """Arm the auto-power-down timer (replaces any existing one)."""
        import asyncio
        self._cancel_oakd_keepalive()
        self._oakd_keepalive_task = asyncio.create_task(self._oakd_keepalive_powerdown())

    async def _oakd_keepalive_powerdown(self) -> None:
        """Power the OAK-D down once the keep-alive window elapses, unless a new
        capture/teleop session claimed it or the user took ownership meanwhile."""
        import asyncio
        try:
            await asyncio.sleep(self._oakd_keepalive_s)
        except asyncio.CancelledError:
            return
        # Clear our own ref first so set_oakd_enabled() below is a no-op cancel.
        self._oakd_keepalive_task = None
        if self._capturing or self.is_teleop_active or not self._oakd_auto_enabled:
            return
        logger.info("OAK-D keep-alive expired — powering down")
        await self.set_oakd_enabled(False)

    # ── Teleop mode (mutually exclusive with recording) ───────────────────────

    async def start_teleop(self) -> None:
        if self._capturing:
            raise RuntimeError("cannot enter teleop while a capture is running; stop capture first")
        if self._teleop is not None and self._teleop.is_running:
            logger.info("teleop already running")
            return

        # Teleop takes over the OAK — cancel any pending auto-power-down and
        # drop ownership so the keep-alive timer never fires into teleop.
        self._cancel_oakd_keepalive()
        self._oakd_auto_enabled = False

        # Release the OAK from recording-mode OakdCapture
        if self._oakd is not None:
            try:
                self._oakd.shutdown()
            except Exception as e:
                logger.warning("OakdCapture shutdown before teleop: %s", e)
            self._oakd = None

        from grabette.hardware.oakd_teleop import OakdTeleop
        self._teleop = OakdTeleop()
        # Always start in "not sending" state — user presses button to begin
        # sending, allowing free repositioning without moving the robot.
        self._teleop_sending = False
        # Run blocking OAK build/start in the executor to keep the event loop free
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._teleop.init_device)
        await loop.run_in_executor(None, self._teleop.start)
        logger.info("Teleop mode started (sending=False)")

    async def stop_teleop(self) -> None:
        if self._teleop is None:
            return
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._teleop.shutdown)
        self._teleop = None
        self._teleop_sending = False
        # Re-init the recording-mode OAK pipeline so next capture works
        if self._enable_oakd:
            await loop.run_in_executor(None, self._init_oakd)
        logger.info("Teleop mode stopped")

    @property
    def is_teleop_sending(self) -> bool:
        return self._teleop_sending and self.is_teleop_active

    def set_teleop_send(self, on: bool) -> None:
        if not self.is_teleop_active:
            logger.warning("set_teleop_send ignored — teleop not active")
            return
        self._teleop_sending = bool(on)
        logger.info("Teleop sending = %s", self._teleop_sending)

    @property
    def is_teleop_active(self) -> bool:
        return self._teleop is not None and self._teleop.is_running

    def get_teleop_delta(self) -> dict | None:
        if self._teleop is None:
            return None
        d = self._teleop.latest_delta
        if d is None:
            return None
        return {
            "t_host": d.t_host,
            "dx": d.dx, "dy": d.dy, "dz": d.dz,
            "dqx": d.dqx, "dqy": d.dqy, "dqz": d.dqz, "dqw": d.dqw,
        }

    def get_teleop_pose(self) -> dict | None:
        if self._teleop is None:
            return None
        p = self._teleop.latest_pose
        if p is None:
            return None
        return {
            "t_host": p.t_host,
            "tx": float(p.translation[0]),
            "ty": float(p.translation[1]),
            "tz": float(p.translation[2]),
            "qx": float(p.quaternion[0]),
            "qy": float(p.quaternion[1]),
            "qz": float(p.quaternion[2]),
            "qw": float(p.quaternion[3]),
        }

    def get_teleop_stats(self) -> dict:
        if self._teleop is None:
            return {}
        return self._teleop.stats()

    def get_state(self) -> SensorState:
        angle = None

        if self._capturing:
            # During capture, read from capture buffers (no I2C contention)
            if self._angle and self._angle._samples.samples:
                last = self._angle._samples.samples[-1]
                angle = AngleSample(
                    timestamp_ms=last["cts"],
                    proximal=last["value"][1],
                    distal=last["value"][0],
                )
        else:
            # When idle, read directly from sensors
            if self._angle and self._angle._i2c_1 and self._angle._i2c_2:
                try:
                    raw1 = self._angle._read_angle_raw(self._angle._i2c_1)
                    raw2 = self._angle._read_angle_raw(self._angle._i2c_2)
                    cal1 = self._angle._normalize_angle(raw1 - self._angle._offset_1_deg) * self._angle.DISTAL_SIGN
                    cal2 = self._angle._normalize_angle(raw2 - self._angle._offset_2_deg) * self._angle.PROXIMAL_SIGN
                    angle = AngleSample(
                        timestamp_ms=time.time() * 1000,
                        proximal=math.radians(cal2),
                        distal=math.radians(cal1),
                    )
                except Exception:
                    pass

        imu = None
        if self._oakd is not None and self._oakd.is_initialized:
            raw_imu = self._oakd.get_latest_imu()
            if raw_imu is not None:
                imu = IMUSample(**raw_imu)

        return SensorState(imu=imu, angle=angle, capture=self.get_capture_status())

    async def start_capture(self, session_dir: Path) -> None:
        if self._capturing:
            raise RuntimeError("Already capturing")

        import asyncio
        loop = asyncio.get_event_loop()

        # Per-step timing — collected into a single summary line at the
        # bottom of this function so we can attribute the cost of each
        # phase (OAK cold init dominates by an order of magnitude).
        t_phases: dict[str, float] = {}

        # A new capture cancels any pending OAK-D keep-alive power-down.
        self._cancel_oakd_keepalive()

        # Auto-connect the OAK-D if it's currently off — recording without
        # depth/IMU is rarely what the user wants, and this matches the UI
        # convention that toggling on/off is the "intent" flag. Errors during
        # init are logged inside _init_oakd and leave _oakd=None; the rest
        # of start_capture handles that gracefully. We then own its power and
        # will auto-power-down after the keep-alive window once capture stops.
        _t = time.monotonic()
        if not self.is_oakd_initialized:
            await self.set_oakd_enabled(True)
            self._oakd_auto_enabled = True
        t_phases["oakd_enable"] = (time.monotonic() - _t) * 1000

        # Safety net: the previous stop_capture schedules the camera re-init to
        # run during idle (see stop_capture). If a restart beats it, do it now.
        _t = time.monotonic()
        if self._needs_reinit:
            self._reinit_hardware()
        t_phases["reinit_safety_net"] = (time.monotonic() - _t) * 1000

        # Defer the recording clock until the OAK-D is producing valid frames
        # (autoexposure + depth converged), so t=0 lands on good data instead
        # of cold-boot warmup. No-op/fast if the OAK-D is already warm.
        _t = time.monotonic()
        if self._oakd and self._oakd.is_initialized:
            await loop.run_in_executor(
                None, self._oakd.wait_until_ready, OAKD_READY_TIMEOUT_S,
            )
        t_phases["oakd_wait_ready"] = (time.monotonic() - _t) * 1000

        self._capture_session_dir = session_dir
        # Captured here so metadata.json can record when this episode
        # actually began (vs the sync-scheduled T₀ in `sync.scheduled_start_utc`).
        self._wall_clock_start = datetime.now(timezone.utc).isoformat()

        # Set flag BEFORE starting streams so the daemon poll loop
        # (get_state) reads from capture buffers instead of doing
        # direct I2C reads that would contend with the angle capture thread.
        self._capturing = True

        # Start synchronized capture — all streams share the same
        # SyncManager t=0 reference (time.monotonic based).
        _t = time.monotonic()
        self._sync.start()
        if self._angle:
            self._angle.start_capture()
        if self._oakd and self._oakd.is_initialized:
            self._oakd.start_recording(session_dir)
        self._camera.start_recording(session_dir / "raw_video.mp4")
        t_phases["start_streams"] = (time.monotonic() - _t) * 1000

        total = sum(t_phases.values())
        logger.info(
            "RpiBackend capture started → %s  [timing ms: %s  total=%.0f]",
            session_dir,
            " ".join(f"{k}={v:.0f}" for k, v in t_phases.items()),
            total,
        )

    async def stop_capture(self) -> CaptureStatus:
        if not self._capturing:
            raise RuntimeError("Not capturing")

        # Per-step timing for a one-line summary at the end. Useful for
        # diagnosing where the LED-blinks-too-long-on-stop time goes.
        t_phases: dict[str, float] = {}

        # Keep _capturing = True until ALL streams have stopped, to
        # prevent the daemon poll loop (get_state) from doing direct
        # I2C reads while the angle capture thread is still running.

        # Grab sync-clock duration before stopping streams (monotonic,
        # same clock used by all stream timestamps — no wall-clock drift).
        duration_ms = self._sync.get_timestamp_ms()

        # Stop angle BEFORE camera. camera.stop() runs ffmpeg muxing
        # which takes ~1-2s — if angle capture is still running during
        # muxing, samples extend past the video duration.
        _t = time.monotonic()
        angle_samples = None
        angle_count = 0
        if self._angle:
            angle_data = self._angle.stop()
            angle_count = len(angle_data.samples)
            angle_samples = angle_data.samples if angle_data.samples else None
        t_phases["angle_stop"] = (time.monotonic() - _t) * 1000

        # Finalize OAK and RPi camera concurrently. Both flip their "recording"
        # flag immediately (capture stops at once) and then spend ~1-2s muxing
        # H.264 → mp4. Running the OAK finalize in an executor while the camera
        # finalize runs here overlaps the muxes (the OAK also muxes left/right
        # in parallel internally), cutting the stop/save time to ~the single
        # longest mux. Angle is already stopped above, so the "angle must stop
        # before the camera mux" ordering still holds.
        import asyncio
        loop = asyncio.get_event_loop()
        # Time each branch independently so we can see which mux dominates.
        oakd_t0 = camera_t0 = time.monotonic()
        oakd_fut = None
        oakd_done_cb_t: list[float] = []
        if self._oakd and self._oakd.is_recording:
            def _oakd_done(fut):
                oakd_done_cb_t.append(time.monotonic())
            oakd_fut = loop.run_in_executor(None, self._oakd.stop_recording)
            oakd_fut.add_done_callback(_oakd_done)
        frame_timestamps = self._camera.stop()
        t_phases["camera_stop"] = (time.monotonic() - camera_t0) * 1000
        oakd_stats = await oakd_fut if oakd_fut is not None else None
        # oakd_done_cb_t[0] (if present) is when stop_recording actually
        # finished — gives the OAK's intrinsic mux time, decoupled from
        # the time we spent awaiting it.
        if oakd_done_cb_t:
            t_phases["oakd_stop"] = (oakd_done_cb_t[0] - oakd_t0) * 1000

        # NOW safe to clear flag — all streams stopped, no I2C contention.
        self._capturing = False

        # If the daemon auto-enabled the OAK-D for this capture, keep it warm
        # for the grace period so a back-to-back recording starts instantly,
        # then power it down to save battery. A user-enabled OAK-D (auto flag
        # cleared) is left on for live view.
        if self._oakd_auto_enabled and self.is_oakd_initialized:
            self._schedule_oakd_keepalive()

        duration = round(duration_ms / 1000.0, 2)

        # Compute actual video FPS from frame timestamps
        actual_fps = float(FPS)
        video_span_ms = 0.0
        if len(frame_timestamps) >= 2:
            video_span_ms = frame_timestamps[-1] - frame_timestamps[0]
            if video_span_ms > 0:
                actual_fps = round((len(frame_timestamps) - 1) / (video_span_ms / 1000.0), 3)

        status = CaptureStatus(
            is_capturing=False,
            session_id=self._capture_session_dir.name if self._capture_session_dir else None,
            duration_seconds=duration,
            frame_count=self._camera.frame_count,
            imu_sample_count=oakd_stats.get("imu_samples", 0) if oakd_stats else 0,
            angle_sample_count=angle_count,
        )

        # Write output files
        _t = time.monotonic()
        if self._capture_session_dir:
            # Save per-frame timestamps (sync-clock-relative ms) for frame
            # drop detection and accurate video-trajectory alignment.
            (self._capture_session_dir / "frame_timestamps.json").write_text(
                json.dumps(frame_timestamps)
            )

            # Save angle data on its own (no longer multiplexed into imu_data.json).
            if angle_samples is not None:
                (self._capture_session_dir / "angle_data.json").write_text(
                    json.dumps({"samples": angle_samples})
                )

            meta = {
                "duration_seconds": status.duration_seconds,
                "frame_count": status.frame_count,
                "imu_sample_count": status.imu_sample_count,
                "angle_sample_count": status.angle_sample_count,
                "fps": actual_fps,
                "backend": "rpi",
                "device_id": settings.device_id,
                "wall_clock_start_utc": getattr(self, "_wall_clock_start", None),
            }
            if oakd_stats:
                meta["oakd"] = oakd_stats
            # Attach sync metadata (T₀, peers, sleep skew, start cost) if
            # the EpisodeScheduler populated it. Lets workstation analysis
            # group multi-device episodes without an external manifest.
            sync_meta = self.get_sync_metadata()
            if sync_meta:
                meta["sync"] = sync_meta
            (self._capture_session_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        t_phases["json_writes"] = (time.monotonic() - _t) * 1000

        self._sync.reset()

        # Defer hardware re-init OUT of the stop path: re-creating the picamera2
        # instance (~1-2s) is prep for the NEXT capture, not part of saving this
        # one, so it must not delay the LED/stop. Schedule it to run right after
        # this coroutine returns (LED already off). It still completes during
        # idle — keeping the RPi live preview alive for framing the next shot —
        # and blocks the loop no longer than the old in-stop re-init did.
        self._needs_reinit = True
        loop.call_soon(self._reinit_hardware)

        self._capture_session_dir = None
        # camera_stop and oakd_stop run in PARALLEL (executor + main),
        # so the wall-clock cost of that step is max(the two), not their
        # sum. Surface that in the reported total so the log is honest.
        parallel = max(
            t_phases.get("camera_stop", 0.0),
            t_phases.get("oakd_stop", 0.0),
        )
        serial = (
            t_phases.get("angle_stop", 0.0)
            + t_phases.get("json_writes", 0.0)
        )
        total = serial + parallel
        logger.info(
            "RpiBackend capture stopped  [timing ms: %s  total=%.0f  "
            "(camera_stop‖oakd_stop = max %.0f)]",
            " ".join(f"{k}={v:.0f}" for k, v in t_phases.items()),
            total, parallel,
        )
        return status

    def _reinit_hardware(self) -> None:
        """Re-create the RPi camera (picamera2 needs a fresh instance after a
        stop) and re-init angle sensors, readying them for the next capture.
        Deferred out of stop_capture so it never delays the stop/save; normally
        runs during idle (scheduled via loop.call_soon), with start_capture as a
        fast-restart fallback. Idempotent — the flag guards against double-run.

        Per-step timing logged for diagnosis: this is a SYNC function called
        via loop.call_soon, which means while it runs the event loop is
        blocked — anything else awaiting the loop (e.g. sync/stop's peer
        fan-out continuing after stop_capture returns) waits for this to
        finish. If reinit_hardware turns out to dominate the stop-blink
        budget we'll convert it to asyncio.to_thread.
        """
        if not self._needs_reinit:
            return
        from grabette.hardware.camera import VideoCapture
        t_phases: dict[str, float] = {}
        t0 = time.monotonic()
        _t = time.monotonic()
        self._camera = VideoCapture(self._sync, fps=FPS)
        self._camera.init_camera()
        t_phases["camera_reinit"] = (time.monotonic() - _t) * 1000
        if self._enable_angle:
            _t = time.monotonic()
            self._init_angle_sensors()
            t_phases["angle_reinit"] = (time.monotonic() - _t) * 1000
        self._needs_reinit = False
        total = (time.monotonic() - t0) * 1000
        logger.info(
            "_reinit_hardware completed  [timing ms: %s  total=%.0f] "
            "(NOTE: blocks event loop while running)",
            " ".join(f"{k}={v:.0f}" for k, v in t_phases.items()),
            total,
        )

    def get_capture_status(self) -> CaptureStatus:
        duration = 0.0
        if self._capturing and self._sync and self._sync.is_started:
            duration = self._sync.get_timestamp_ms() / 1000.0

        frame_count = self._camera.frame_count if self._camera else 0
        angle_count = self._angle.sample_count if self._angle else 0
        imu_count = self._oakd.imu_sample_count if (self._oakd and self._oakd.is_recording) else 0

        return CaptureStatus(
            is_capturing=self._capturing,
            session_id=self._capture_session_dir.name if self._capture_session_dir else None,
            duration_seconds=round(duration, 2),
            frame_count=frame_count,
            imu_sample_count=imu_count,
            angle_sample_count=angle_count,
        )

    @property
    def is_capturing(self) -> bool:
        return self._capturing

    def get_frame_jpeg(self) -> bytes | None:
        """Capture a JPEG frame from picamera2.

        Returns None during active capture to avoid competing with the
        H.264 encoder for camera resources (preserves frame timing).
        """
        if self._capturing:
            return None
        if self._camera and self._camera._picam2:
            try:
                import io
                buf = io.BytesIO()
                self._camera._picam2.capture_file(buf, format="jpeg")
                return buf.getvalue()
            except Exception as e:
                logger.debug("Failed to capture JPEG: %s", e)
        return None

    def get_depth_jpeg(self) -> bytes | None:
        """Return latest OAK-D depth frame as a colorized JPEG.

        Available both during capture and at idle (the OAK-D pipeline runs
        continuously after start()). Returns None if OAK-D is not present
        or no depth frame has arrived yet.
        """
        if self._oakd and self._oakd.is_initialized:
            try:
                return self._oakd.get_depth_jpeg()
            except Exception as e:
                logger.debug("Failed to get depth JPEG: %s", e)
        return None
