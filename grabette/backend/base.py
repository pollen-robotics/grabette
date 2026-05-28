from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from grabette.models import CaptureStatus, SensorState


class Backend(ABC):
    # Optional sync metadata attached by the EpisodeScheduler after a
    # successful start. Backends merge this into metadata.json at
    # stop_capture so each device's episode is self-describing — no
    # external manifest needed. Empty for local-only captures.
    #
    # Subclasses don't need to initialise the attribute explicitly:
    # get_sync_metadata() defaults to an empty dict via getattr.

    def set_sync_metadata(self, meta: dict) -> None:
        self._sync_metadata = dict(meta or {})

    def get_sync_metadata(self) -> dict:
        return getattr(self, "_sync_metadata", {}) or {}

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def get_state(self) -> SensorState: ...

    @abstractmethod
    async def start_capture(self, session_dir: Path) -> None: ...

    @abstractmethod
    async def stop_capture(self) -> CaptureStatus: ...

    @abstractmethod
    def get_capture_status(self) -> CaptureStatus: ...

    @property
    @abstractmethod
    def is_capturing(self) -> bool: ...

    @abstractmethod
    def get_frame_jpeg(self) -> bytes | None: ...

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
