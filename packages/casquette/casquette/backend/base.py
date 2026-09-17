from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from casquette.models import CaptureStatus, SensorState


class Backend(ABC):
    # Pre-T0 warmup hook. A fleet-scheduled capture calls this during the lead
    # window so slow hardware bring-up doesn't sit between T0 and the first
    # frame (that variance is what drifts two devices' recordings apart).
    # Default no-op: casquette keeps its camera warm across captures, so
    # there's nothing to pre-init — backends may override if that changes.
    async def prepare_capture(self) -> None:
        return None

    # Sync metadata attached by the capture scheduler after a synchronized
    # start; backends fold it into metadata.json at stop_capture so a
    # workstation can pair per-device episodes by shared T0. Empty for solo
    # captures. Subclasses need not initialise the attribute (getattr default).
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
