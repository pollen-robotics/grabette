from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from casquette.models import CaptureStatus, SensorState


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

    # Runtime camera controls — default to no-op so backends without
    # tunable exposure (e.g. MockBackend) don't have to implement them.
    def get_camera_exposure_us(self) -> int:
        return 0

    def set_camera_exposure_us(self, us: int) -> int:
        raise NotImplementedError(
            "This backend does not support runtime exposure tuning"
        )
