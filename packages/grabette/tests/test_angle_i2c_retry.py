"""A broken I2C transaction costs a retry, not a 1 s hole in angle_data."""
import errno
import logging
import os
import threading
import time

from grabette.hardware import angle as angle_mod


class FakeSync:
    is_started = True

    def get_timestamp_ms(self):
        return time.monotonic() * 1000


class ScriptedI2C:
    """Fails with the given errnos on its first reads, then answers."""

    def __init__(self, *errnos, fail_s=0.0):
        self.errnos = list(errnos)
        self.fail_s = fail_s
        self.calls = 0

    def writeto_then_readfrom(self, addr, out, result):
        self.calls += 1
        if self.errnos:
            code = self.errnos.pop(0)
            time.sleep(self.fail_s)
            raise OSError(code, os.strerror(code))
        result[0], result[1] = 0x01, 0x00


def _run(cap, seconds, caplog):
    with caplog.at_level(logging.INFO, logger=angle_mod.logger.name):
        cap._running = True
        t = threading.Thread(target=cap._capture_loop)
        t.start()
        time.sleep(seconds)
        cap._running = False
        t.join()
    return [r.getMessage() for r in caplog.records]


def test_the_eio_then_timeout_sequence_keeps_every_sample(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(angle_mod, "CALIBRATION_FILE", tmp_path / "none.json")
    cap = angle_mod.AngleCapture(FakeSync())
    # What the Pi does: EIO, then the next read on that bus times out.
    distal = ScriptedI2C(errno.EIO, errno.ETIMEDOUT, fail_s=0.02)
    proximal = ScriptedI2C()
    cap._i2c_1, cap._i2c_2 = distal, proximal
    messages = _run(cap, 0.3, caplog)

    # Two extra reads of the pair, and no iteration without a sample.
    assert proximal.calls == distal.calls
    assert len(cap._samples.samples) == distal.calls - 2
    failed = [m for m in messages if "read failed" in m]
    assert len(failed) == 2
    assert "attempt 1" in failed[0] and "attempt 2" in failed[1]
    summary = next(m for m in messages if m.startswith("Angle: ") and "reads" in m)
    assert "2 errors" in summary and "1 samples saved by a retry" in summary
    assert "'EIO': 1" in summary and "'ETIMEDOUT': 1" in summary
    # A retried pair is ~40 ms late, under the stall threshold.
    assert not [m for m in messages if "Angle loop stalled" in m]


def test_the_retried_sample_takes_the_time_of_the_good_read(monkeypatch, tmp_path):
    monkeypatch.setattr(angle_mod, "CALIBRATION_FILE", tmp_path / "none.json")
    cap = angle_mod.AngleCapture(FakeSync())
    cap._i2c_1, cap._i2c_2 = ScriptedI2C(errno.EIO, fail_s=0.05), ScriptedI2C()
    cap._running = True
    t = threading.Thread(target=cap._capture_loop)
    t0 = time.monotonic() * 1000
    t.start()
    time.sleep(0.15)
    cap._running = False
    t.join()
    # The first sample is stamped after the 50 ms failure, not before it.
    assert cap._samples.samples[0]["cts"] - t0 >= 45


def test_a_sensor_that_keeps_failing_is_tried_max_attempts_per_iteration(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(angle_mod, "CALIBRATION_FILE", tmp_path / "none.json")
    cap = angle_mod.AngleCapture(FakeSync())
    gone = ScriptedI2C(*[errno.EREMOTEIO] * 10_000)
    cap._i2c_1, cap._i2c_2 = ScriptedI2C(), gone
    messages = _run(cap, 0.2, caplog)
    summary = next(m for m in messages if m.startswith("Angle: ") and "reads" in m)
    iterations = int(summary.split("Angle: ")[1].split(" reads")[0])
    assert gone.calls == iterations * angle_mod.MAX_READ_ATTEMPTS
    assert cap._samples.samples == []
    assert "0 samples saved by a retry" in summary


def test_set_i2c_timeout_sets_the_adapter_timeout_in_10_ms_units(monkeypatch):
    calls = []
    monkeypatch.setattr(angle_mod.os, "open", lambda path, flags: calls.append(path) or 7)
    monkeypatch.setattr(angle_mod.os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(angle_mod.fcntl, "ioctl",
                        lambda fd, req, arg: calls.append((fd, req, arg)))
    assert angle_mod.set_i2c_timeout(3) is True
    assert calls == ["/dev/i2c-3", (7, 0x0702, 2), ("close", 7)]


def test_set_i2c_timeout_without_the_bus_only_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=angle_mod.logger.name):
        assert angle_mod.set_i2c_timeout(987) is False
    assert "I2C timeout not set on /dev/i2c-987" in caplog.text


def test_set_i2c_timeout_refused_by_the_kernel_closes_the_bus(monkeypatch, caplog):
    closed = []
    monkeypatch.setattr(angle_mod.os, "open", lambda path, flags: 7)
    monkeypatch.setattr(angle_mod.os, "close", closed.append)

    def refuse(fd, req, arg):
        raise OSError(errno.ENOTTY, os.strerror(errno.ENOTTY))

    monkeypatch.setattr(angle_mod.fcntl, "ioctl", refuse)
    with caplog.at_level(logging.WARNING, logger=angle_mod.logger.name):
        assert angle_mod.set_i2c_timeout(4) is False
    assert closed == [7]
