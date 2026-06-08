"""Replay engine — loads a recorded episode and feeds samples into a SampleRing at real-time rate."""

from __future__ import annotations

import asyncio
import json
import logging
from bisect import bisect_left
from pathlib import Path

from grabette.daemon import SampleRing

logger = logging.getLogger(__name__)


class ReplayEngine:
    def __init__(self) -> None:
        self.ring = SampleRing(maxlen=500)
        self._episode_id: str | None = None
        self._duration_ms: float = 0
        self._imu_samples: list[dict] = []   # {"t", "a", "g"}
        self._angle_samples: list[dict] = []  # {"t", "p", "d"}
        self._imu_times: list[float] = []
        self._angle_times: list[float] = []
        self._playback_ms: float = 0
        self._playing: bool = False
        self._active: bool = False
        self._task: asyncio.Task | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def status(self) -> dict:
        return {
            "active": self._active,
            "episode_id": self._episode_id,
            "time_ms": self._playback_ms,
            "duration_ms": self._duration_ms,
            "playing": self._playing,
        }

    def load(self, episode_dir: str, episode_id: str) -> None:
        """Load imu_data.json + metadata.json from an episode directory."""
        path = Path(episode_dir)

        # Load duration from metadata
        meta_path = path / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._duration_ms = meta.get("duration_seconds", 0) * 1000
        else:
            self._duration_ms = 0

        # Load IMU data
        imu_path = path / "imu_data.json"
        if not imu_path.exists():
            raise FileNotFoundError(f"No imu_data.json in {episode_dir}")

        with open(imu_path) as f:
            data = json.load(f)

        streams = data.get("1", {}).get("streams", {})

        accl_samples = streams.get("ACCL", {}).get("samples", [])
        gyro_samples = streams.get("GYRO", {}).get("samples", [])
        angl_samples = streams.get("ANGL", {}).get("samples", [])

        # Zip ACCL + GYRO by index into unified IMU samples
        self._imu_samples = []
        n_imu = min(len(accl_samples), len(gyro_samples))
        for i in range(n_imu):
            a = accl_samples[i]
            g = gyro_samples[i]
            self._imu_samples.append({
                "t": a["cts"],
                "a": a["value"],
                "g": g["value"],
            })
        self._imu_times = [s["t"] for s in self._imu_samples]

        # Parse angle samples — indices swapped to match live convention
        # (in imu_data.json value=[sensor1, sensor2], but live maps
        #  proximal=value[1], distal=value[0])
        self._angle_samples = []
        for s in angl_samples:
            v = s["value"]
            self._angle_samples.append({
                "t": s["cts"],
                "p": v[1],
                "d": v[0],
            })
        self._angle_times = [s["t"] for s in self._angle_samples]

        # Fall back to data duration if metadata missing
        if self._duration_ms == 0 and self._imu_times:
            self._duration_ms = self._imu_times[-1]

        self._episode_id = episode_id
        logger.info(
            "Replay loaded: %s — %d IMU, %d angle samples, %.1fs",
            episode_id, len(self._imu_samples), len(self._angle_samples),
            self._duration_ms / 1000,
        )

    async def start(self) -> None:
        self._playback_ms = 0
        self._playing = True
        self._active = True
        self.ring = SampleRing(maxlen=500)
        self._task = asyncio.create_task(self._feed_loop())

    async def stop(self) -> None:
        self._playing = False
        self._active = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def pause(self) -> None:
        self._playing = False

    def resume(self) -> None:
        self._playing = True

    def seek(self, time_ms: float) -> None:
        """Seek to a position — replace ring and pre-fill ~1s of trailing context."""
        time_ms = max(0, min(time_ms, self._duration_ms))
        self._playback_ms = time_ms
        self.ring = SampleRing(maxlen=500)

        # Pre-fill ~1s of trailing context
        context_start = max(0, time_ms - 1000)
        self._push_window(context_start, time_ms)

    def _push_window(self, from_ms: float, to_ms: float) -> None:
        """Push all samples in [from_ms, to_ms) into the ring."""
        # IMU
        i_start = bisect_left(self._imu_times, from_ms)
        i_end = bisect_left(self._imu_times, to_ms)
        for i in range(i_start, i_end):
            self.ring.push_raw(imu=self._imu_samples[i])

        # Angle
        a_start = bisect_left(self._angle_times, from_ms)
        a_end = bisect_left(self._angle_times, to_ms)
        for i in range(a_start, a_end):
            self.ring.push_raw(angle=self._angle_samples[i])

    async def _feed_loop(self) -> None:
        """Run at 50Hz, advancing playback_ms by 20ms each tick."""
        tick_ms = 20.0
        try:
            while self._active:
                if self._playing:
                    prev = self._playback_ms
                    self._playback_ms += tick_ms
                    if self._playback_ms >= self._duration_ms:
                        self._playback_ms = self._duration_ms
                        self._push_window(prev, self._playback_ms)
                        self._playing = False
                        logger.info("Replay reached end of episode")
                        # Stay active but paused at end
                        continue
                    self._push_window(prev, self._playback_ms)
                await asyncio.sleep(tick_ms / 1000)
        except asyncio.CancelledError:
            pass
