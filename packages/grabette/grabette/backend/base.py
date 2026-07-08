from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from grabette.models import CaptureStatus, SensorState


class Backend(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def get_state(self) -> SensorState: ...

    @abstractmethod
    async def start_capture(self, episode_dir: Path) -> None: ...

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
