from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from grabette.models import CaptureStatus, SensorState


class Backend(ABC):
    # The capture-state LED is NOT driven from here: ButtonListener owns it and
    # keeps it in sync with the capture state from its own monitor thread, so a
    # capture started by the button, the dashboard or the fleet all light up the
    # same way. See button_listener._desired_led.

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def get_state(self) -> SensorState: ...

    @abstractmethod
    async def start_capture(self, episode_dir: Path) -> None: ...

    def set_sync_metadata(self, meta: dict) -> None:
        """Attach multi-device sync info to the NEXT episode's metadata.json.

        Set by the CaptureScheduler around a synchronized start (the shared
        scheduled_start_utc, the actual capture-start instant, skew). stop_capture
        folds it into metadata.json so a workstation can (a) pair the per-device
        episodes by the common scheduled_start_utc and (b) convert each stream to
        absolute UTC to align them. Consumed once (cleared after)."""
        self._sync_metadata = dict(meta or {})

    def _take_sync_metadata(self) -> dict:
        """Return and clear the pending sync metadata (so a later solo capture
        never inherits a previous synchronized episode's data)."""
        meta = getattr(self, "_sync_metadata", {})
        self._sync_metadata = {}
        return meta

    async def prepare_capture(self) -> None:
        """Warm the hardware (init + wait until it produces valid frames)
        WITHOUT starting a recording, so a later start_capture can begin the
        recording clock immediately.

        This exists for synchronized multi-device starts: if each device only
        warms up at the shared T0 (inside start_capture), the recording clock
        lands at T0 + a VARIABLE warmup, so devices drift apart (the OAK-D
        cold-boot alone is several seconds). Calling this before T0 removes
        that variance from the start. Idempotent, fast when already warm.
        Default: no-op (backends with no slow init)."""
        return None

    @property
    def hardware_error(self) -> str:
        """Why this device must not record right now ("" = fine).

        A backend sets this when it detects a fault that would make every
        recorded episode unusable downstream (see RpiBackend: no OAK-D offline
        calibration). Capture is refused while it is set, and the button
        listener blinks the error pattern so the fault is visible on the device
        itself. Default: no backend-detectable fault."""
        return ""

    # --- busy gate -----------------------------------------------------------
    # A recording must not start on top of dataset work. On a Pi the upload is
    # not a sleeping socket: hf_xet chunks and hashes in native threads, so it
    # competes for CPU with the H.264 encoders and the OAK-D drainers and the
    # episode comes out with dropped frames and jittered timestamps — an episode
    # that looks recorded and is quietly worse than nothing.
    #
    # The fleet already refuses to start a recording on a device it knows is
    # uploading, but that gate is fleet-side only: the PHYSICAL BUTTON and the
    # local dashboard walk straight past it. So the device enforces it too, for
    # itself, whoever asks.
    #
    # Injected as a probe rather than read directly: knowing about HF uploads is
    # not a backend's business. app/main.py owns that knowledge (it is the same
    # source the fleet heartbeat reports from) and installs it at startup, so
    # there is exactly one answer to "what is this device doing".

    def set_busy_probe(self, probe) -> None:
        """Install the callable answering "why is this device too busy to
        record?" — a reason string, or "" when free."""
        self._busy_probe = probe

    @property
    def busy_reason(self) -> str:
        """Why capture is refused for BUSY reasons ("" = free).

        Never raises: a probe that fails must not take recording down with it,
        so a broken probe reads as free. Distinct from hardware_error — that one
        is a fault needing intervention, this one clears on its own."""
        probe = getattr(self, "_busy_probe", None)
        if probe is None:
            return ""
        try:
            return probe() or ""
        except Exception:  # noqa: BLE001 — a probe must never block recording
            return ""

    def raise_if_capture_blocked(self) -> None:
        """Refuse to start a capture that would produce unusable data.

        Public and called from EVERY start path — the backends' start_capture
        and CaptureScheduler.schedule — because gating the callers one by one is
        how the next start path silently bypasses it.

        RuntimeError so it travels like any other start failure: the REST routes
        turn it into an error response, the relay reports it to the fleet, and
        the button listener logs it (the LED is already showing the state)."""
        if self.hardware_error:
            raise RuntimeError(self.hardware_error)
        busy = self.busy_reason
        if busy:
            raise RuntimeError(busy)

    @abstractmethod
    async def stop_capture(self) -> CaptureStatus: ...

    @abstractmethod
    def get_capture_status(self) -> CaptureStatus: ...

    @property
    @abstractmethod
    def is_capturing(self) -> bool: ...

    @abstractmethod
    def get_frame_jpeg(self) -> bytes | None: ...

    @property
    def is_camera_connected(self) -> bool:
        """True if the RGB camera device is connected/initialized.

        Unlike get_frame_jpeg() this stays True during capture, so it can
        drive a connection indicator. Default False for backends that don't
        track it.
        """
        return False

    def get_depth_jpeg(self) -> bytes | None:
        """Optional: colorized OAK-D depth JPEG for live view. Default: None."""
        return None

    # ── Teleop mode (optional; default = unsupported) ─────────────────────────

    async def start_teleop(self) -> None:
        """Switch into live VIO teleop mode. Mutually exclusive with recording.

        Default raises — backends that don't support teleop are unchanged.
        """
        raise NotImplementedError("teleop mode not supported by this backend")

    async def stop_teleop(self) -> None:
        """Exit teleop mode and return to the idle / recording-ready state."""
        raise NotImplementedError("teleop mode not supported by this backend")

    @property
    def is_teleop_active(self) -> bool:
        """True if teleop mode is currently running."""
        return False

    @property
    def is_teleop_sending(self) -> bool:
        """True if the daemon should emit deltas with send=True.

        Defaults False — `start_teleop()` activates the mode but does NOT
        immediately start sending. The hardware button (or a future UI
        control) toggles this so the user can reposition the grabette
        without driving the robot.
        """
        return False

    def set_teleop_send(self, on: bool) -> None:
        """Turn delta-sending on or off (no-op when teleop is inactive)."""
        pass

    def get_teleop_delta(self) -> dict | None:
        """Most recent camera-local delta. None if no pose yet or teleop is off."""
        return None

    def get_teleop_pose(self) -> dict | None:
        """Most recent absolute pose. None if no pose yet or teleop is off."""
        return None

    def get_teleop_stats(self) -> dict:
        """Live framerate + pose-count stats. Empty when teleop is off."""
        return {}
