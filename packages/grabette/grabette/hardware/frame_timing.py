"""Associate written encoder frames with their original sensor capture times."""

from threading import Lock


class FrameTiming:
    def __init__(self, sync):
        self.sync = sync
        self.timestamps = []
        self.frames = []
        self.errors = []
        self.submitted_count = 0
        self.written_count = 0
        self._pending = {}
        self._last_input_pts = None
        self._last_output_pts = None
        self._lock = Lock()

    def _error(self, message):
        if len(self.errors) < 10:
            self.errors.append(message)

    def record_error(self, message):
        with self._lock:
            self._error(message)

    def submitted(self, pts_us, sensor_ns):
        # Called before the request is queued to the hardware encoder.
        with self._lock:
            self.submitted_count += 1
            if self._last_input_pts is not None and pts_us <= self._last_input_pts:
                self._error(f"Non-increasing encoder input PTS: {pts_us}")
                return
            self._last_input_pts = pts_us
            if sensor_ns is None:
                self._error(f"Missing sensor timestamp for input PTS: {pts_us}")
                return
            self._pending[pts_us] = sensor_ns

    def written(self, pts_us):
        # FileOutput invokes this only after a picture has been written/flushed.
        # It also runs while stop_encoder drains its outstanding output.
        with self._lock:
            index = self.written_count
            self.written_count += 1
            if pts_us is None:
                self._error("Missing output PTS")
                return
            if self._last_output_pts is not None and pts_us <= self._last_output_pts:
                self._error(f"Non-increasing encoder output PTS: {pts_us}")
            self._last_output_pts = pts_us
            ns = self._pending.pop(pts_us, None)
            if ns is None:
                self._error(f"No sensor timestamp for output PTS: {pts_us}")
                return
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
        # Caller must join/stop the encoder before taking this snapshot.
        with self._lock:
            errors = list(self.errors)
            if not self.timestamps:
                errors.append("Video contains no mapped frames")
            if video_frame_count != len(self.timestamps) or self.written_count != len(self.timestamps):
                errors.append(f"Frame counts differ: video={video_frame_count}, "
                              f"written={self.written_count}, timestamps={len(self.timestamps)}")
            return dict(valid=not errors, source="encoded_output_sensor_timestamp_v1",
                        video_frame_count=video_frame_count,
                        timestamp_count=len(self.timestamps),
                        written_count=self.written_count,
                        submitted_count=self.submitted_count,
                        unwritten_input_count=len(self._pending), errors=errors)
