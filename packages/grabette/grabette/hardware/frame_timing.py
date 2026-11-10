"""Associate written encoder frames with their original sensor capture times.

Only public picamera2 API is involved. The camera reports every frame's
`SensorTimestamp` (via `Picamera2.post_callback`) and the output receives each
written picture's PTS (`Output.outputframe`). picamera2 derives that PTS from
the sensor timestamp, but how (origin, rounding) is an internal detail that
varies between versions. So the link is not assumed: the constant offset
between output PTS and sensor time is checked against the first written
pictures. Afterwards each written picture gets the sensor time at PTS + offset.

The offset comes from the encoder when it says where its PTS start
(`pts_origin_us`, picamera2's `encoder.firsttimestamp`), and is used only if
every one of those pictures falls on a camera frame. Otherwise it is found by
matching alone, and must then be unique: on a lightly loaded Pi the frame
intervals can be so regular that offsets shifted by whole frames fit too
(about half of the 2 min takes of 2026-11-10's bench, never in an episode).

If the PTS ever stops following the sensor clock, no offset fits: the take is
marked invalid rather than given guessed times.
"""

import logging
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)

# Written pictures checked together to find the offset. During episodes, frame
# intervals jitter by ~6 us (median) from one frame to the next, and by at
# least 12 us within 10 frames: no offset shifted by whole frames fits them.
# Without the depth camera's load they can stay within 2 us (see above).
LOCK_FRAMES = 10
# PTS and sensor times are compared in microseconds; allow for rounding.
TOLERANCE_US = 2
_DELTAS = sorted(range(-TOLERANCE_US, TOLERANCE_US + 1), key=abs)


class FrameTiming:
    def __init__(self, sync, pts_origin_us=None):
        """`pts_origin_us`: optional callable returning the sensor time (us)
        the encoder's PTS count from, or None when unknown."""
        self.sync = sync
        self._pts_origin_us = pts_origin_us
        self.timestamps = []
        self.frames = []
        self.errors = []
        self.camera_frame_count = 0
        self.written_count = 0
        self.unwritten_camera_frame_count = 0
        self.pts_offset_us = None
        self.pts_offset_source = None
        self.pts_offset_candidates = None
        self._camera = OrderedDict()  # sensor time (us) -> sensor time (ns), in capture order
        self._last_camera_ns = None
        self._held = []  # (index, pts_us) written before the offset is known
        self._matching_failed = False
        self._last_output_pts = None
        self._lock = Lock()

    def _error(self, message):
        if len(self.errors) < 10:
            self.errors.append(message)

    def record_error(self, message):
        with self._lock:
            self._error(message)

    def camera_frame(self, sensor_ns):
        # Called on picamera2's request thread for every camera frame, before
        # the encoder sees it: keep it short.
        with self._lock:
            self.camera_frame_count += 1
            if self._matching_failed:
                return
            if sensor_ns is None:
                self._error("Missing sensor timestamp on a camera frame")
                return
            if self._last_camera_ns is not None and sensor_ns <= self._last_camera_ns:
                self._error(f"Non-increasing camera sensor timestamp: {sensor_ns}")
                return
            self._last_camera_ns = sensor_ns
            self._camera[sensor_ns // 1000] = sensor_ns

    def written(self, pts_us):
        # The H.264 output invokes this only after a picture has been written/flushed.
        # It also runs while stop_encoder drains its outstanding output.
        with self._lock:
            index = self.written_count
            self.written_count += 1
            if pts_us is None:
                self._error("Missing output PTS")
                return
            # The output writes pictures in PTS order (camera.REORDER_WINDOW,
            # which also rejects B-frames): anything else here is a picture
            # that came too late to be put back, and the take is invalid.
            if self._last_output_pts is not None and pts_us <= self._last_output_pts:
                self._error(f"Non-increasing encoder output PTS: {pts_us}")
                return
            self._last_output_pts = pts_us
            if self.pts_offset_us is None:
                self._held.append((index, pts_us))
                if len(self._held) >= LOCK_FRAMES:
                    self._match_held()
                return
            self._map(index, pts_us)

    def finish(self):
        """Map pictures still waiting for the offset (takes shorter than
        LOCK_FRAMES). Call once the encoder has drained."""
        with self._lock:
            if self._held:
                self._match_held()

    def _find(self, sensor_us):
        for delta in _DELTAS:
            if sensor_us + delta in self._camera:
                return sensor_us + delta
        return None

    def _encoder_origin(self):
        if self._pts_origin_us is None:
            return None
        try:
            origin = self._pts_origin_us()
        except Exception:
            return None
        return origin if isinstance(origin, int) and not isinstance(origin, bool) else None

    def _match_held(self):
        held, self._held = self._held, []
        if self._matching_failed:
            return
        first_pts = held[0][1]

        def fits(offset):
            return all(self._find(pts + offset) is not None for _, pts in held)

        offsets = [key - first_pts for key in self._camera if fits(key - first_pts)]
        # Kept to see how close matching alone comes to being ambiguous.
        self.pts_offset_candidates = len(offsets)
        origin = self._encoder_origin()
        if origin is not None and fits(origin):
            self.pts_offset_us, self.pts_offset_source = origin, "encoder_first_timestamp"
        elif len(offsets) == 1:
            if origin is not None:
                logger.warning("Encoder PTS origin %d us does not fit the camera frames; "
                               "using the matched offset %d us", origin, offsets[0])
            self.pts_offset_us, self.pts_offset_source = offsets[0], "matched"
        else:
            # Not unique: picking one could shift every timestamp by whole frames.
            self._error(f"Cannot match encoder PTS to camera frames: {len(offsets)} "
                        f"offsets fit the first {len(held)} written pictures")
            self._matching_failed = True
            self._camera.clear()
            return
        # Camera frames before the first written picture never reached the
        # file (encoder not started yet, or no keyframe yet): not a loss.
        first_key = self._find(first_pts + self.pts_offset_us)
        while next(iter(self._camera)) < first_key:
            self._camera.popitem(last=False)
        for index, pts in held:
            self._map(index, pts)

    def _map(self, index, pts_us):
        if self._matching_failed:
            return
        key = self._find(pts_us + self.pts_offset_us)
        if key is None:
            self._error(f"No camera frame for output PTS: {pts_us}")
            return
        # Earlier camera frames were never written: the encoder skipped them.
        while next(iter(self._camera)) < key:
            self._camera.popitem(last=False)
            self.unwritten_camera_frame_count += 1
        ns = self._camera.pop(key)
        try:
            ts = self.sync.boottime_ns_to_ms(ns)
        except Exception as exc:
            self._error(f"Cannot map sensor timestamp: {exc}")
            return
        if self.timestamps and ts <= self.timestamps[-1]:
            self._error(f"Non-increasing sensor timestamp at frame {index}")
        self.timestamps.append(ts)
        self.frames.append(dict(frame_index=index, sensor_timestamp_ns=ns,
                                encoder_pts_us=pts_us, timestamp_ms=ts))

    def result(self, video_frame_count):
        # Caller must stop the encoder and call finish() before this snapshot.
        with self._lock:
            errors = list(self.errors)
            if not self.timestamps:
                errors.append("Video contains no mapped frames")
            if video_frame_count != len(self.timestamps) or self.written_count != len(self.timestamps):
                errors.append(f"Frame counts differ: video={video_frame_count}, "
                              f"written={self.written_count}, timestamps={len(self.timestamps)}")
            return dict(valid=not errors, source="encoded_output_pts_matched_sensor_timestamp_v2",
                        video_frame_count=video_frame_count,
                        timestamp_count=len(self.timestamps),
                        written_count=self.written_count,
                        camera_frame_count=self.camera_frame_count,
                        unwritten_camera_frame_count=self.unwritten_camera_frame_count,
                        pts_offset_us=self.pts_offset_us,
                        pts_offset_source=self.pts_offset_source,
                        pts_offset_candidates=self.pts_offset_candidates, errors=errors)
