"""The contract a depth camera must satisfy to drive a Grabette recording.

Grabette's depth/IMU sensing was originally an OAK-D SR behind `depthai`, i.e. a
single-source dependency on the critical path. This Protocol is what lets a
second camera exist alongside it: `RpiBackend` talks only to these members, so
swapping the implementation is a one-line change at the construction site rather
than a rewrite of the backend.

Keeping both implementations alive is not optional. The SLAM pipeline has no
ground truth, so validating a new camera means A/B-ing it against OAK-D
recordings of the same motion — which requires the OAK-D path to keep working.

`OakdCapture` already satisfies this Protocol as written; it was extracted from
that class rather than imposed on it, so no OAK-D code had to change.

Lifecycle, in the order `RpiBackend` drives it:

    init_device()            pipeline starts and runs continuously
    wait_until_ready(...)    block until frames are usable (not just arriving)
    start_recording(dir)     writers begin hitting disk
    stop_recording()         writers stop, files are finalised; pipeline stays up
    shutdown()               pipeline stops, threads exit

The pipeline deliberately runs between recordings so live preview keeps working
and back-to-back captures don't pay the cold-boot warmup again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DepthCameraCapture(Protocol):
    """Depth camera driving one episode's depth (+ optionally IMU) streams.

    `runtime_checkable` only verifies that the members exist, not their
    signatures — enough for a smoke test, not a substitute for reading this
    docstring when writing a new implementation.
    """

    # -------------------------------------------------------------- lifecycle

    def init_device(self) -> None:
        """Connect, read calibration, build and START the pipeline.

        Must leave the device streaming continuously. Raising is acceptable and
        expected when no device is attached — `RpiBackend` catches it and
        carries on without a depth camera.
        """

    def shutdown(self) -> None:
        """Stop the pipeline and join any worker threads. Must be idempotent."""

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until the device is producing *usable* frames, or time out.

        "Usable" is stronger than "arriving": the first frames after
        `init_device()` are auto-exposure and stereo warmup, and are unusable
        for SLAM. Callers gate the recording clock on this.

        Returns True if ready within `timeout`, False on timeout. Returning
        False is not fatal — the caller proceeds anyway, on the grounds that a
        late recording beats a hung one.
        """

    # -------------------------------------------------------------- recording

    def start_recording(self, output_dir: Path) -> None:
        """Begin writing streams into `output_dir`.

        Implementations write their own filenames; the postprocess pipeline,
        `checks/*`, and every existing HuggingFace dataset key on the `oakd_*`
        prefix, so a new implementation should keep emitting those names until a
        rename lands with a compatibility shim.
        """

    def stop_recording(self) -> dict:
        """Stop writing, finalise files, and return per-stream counts.

        The pipeline keeps running. Returns a stats dict recorded into
        `metadata.json`; `imu_samples` is read by the capture-status path, so
        include it (0 is fine for a camera with no IMU).
        """

    # ------------------------------------------------------------- live state

    def get_latest_imu(self) -> dict | None:
        """Latest accel + gyro for the live dashboard, or None if unavailable.

        **A camera with no IMU returns None forever, and that is a supported
        case** — the Gemini 305 has no IMU at all. Consumers must treat None as
        "no IMU on this device", not merely "no sample yet".
        """

    def get_depth_jpeg(self) -> bytes | None:
        """Latest depth frame as a colorized JPEG for live preview.

        Returns None before the first depth frame arrives.
        """

    # -------------------------------------------------------------- properties

    @property
    def is_initialized(self) -> bool:
        """True once `init_device()` has succeeded and before `shutdown()`."""

    @property
    def is_recording(self) -> bool:
        """True between `start_recording()` and `stop_recording()`."""

    @property
    def imu_sample_count(self) -> int:
        """IMU samples appended during the current recording; 0 if no IMU."""
