"""A stalled angle iteration is logged with where its time went."""
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


class FakeI2C:
    def __init__(self, stall_s=0.0, fail_errno=None):
        self.stall_s = stall_s
        self.fail_errno = fail_errno
        self.calls = 0

    def writeto_then_readfrom(self, addr, out, result):
        self.calls += 1
        if self.stall_s:
            stall, self.stall_s = self.stall_s, 0.0
            time.sleep(stall)  # one slow transfer, like the 593 ms hole
            if self.fail_errno is not None:
                # The driver gives up after its timeout, like the ~1 s holes.
                code, self.fail_errno = self.fail_errno, None
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


def test_a_failed_read_is_logged_with_its_errno_and_bus(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(angle_mod, "CALIBRATION_FILE", tmp_path / "none.json")
    cap = angle_mod.AngleCapture(FakeSync())
    distal = FakeI2C(stall_s=0.25, fail_errno=errno.ETIMEDOUT)
    proximal = FakeI2C()
    cap._i2c_1, cap._i2c_2 = distal, proximal
    messages = _run(cap, 0.45, caplog)

    failed = [m for m in messages if "read failed" in m]
    assert len(failed) == 1
    assert "distal" in failed[0] and "/dev/i2c-3" in failed[0] and "TimeoutError" in failed[0]
    stall = next(m for m in messages if "Angle loop stalled" in m)
    # The proximal sensor was still read in the failing iteration.
    assert "distal read 25" in stall and "ETIMEDOUT" in stall
    assert "proximal read -" not in stall
    assert proximal.calls == distal.calls
    summary = next(m for m in messages if m.startswith("Angle: ") and "reads" in m)
    assert "1 errors" in summary and "{'ETIMEDOUT': 1}" in summary
    # The failed iteration leaves no sample: it would have no distal angle.
    assert len(cap._samples.samples) == distal.calls - 1


def test_failed_reads_stop_being_logged_one_by_one(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(angle_mod, "CALIBRATION_FILE", tmp_path / "none.json")

    class Unplugged:
        def writeto_then_readfrom(self, addr, out, result):
            raise OSError(errno.EREMOTEIO, os.strerror(errno.EREMOTEIO))

    cap = angle_mod.AngleCapture(FakeSync())
    cap._i2c_1, cap._i2c_2 = FakeI2C(), Unplugged()
    messages = _run(cap, 0.4, caplog)
    assert len([m for m in messages if "read failed" in m]) == angle_mod.MAX_LOGGED_READ_ERRORS
    summary = next(m for m in messages if m.startswith("Angle: ") and "reads" in m)
    assert "proximal {'EREMOTEIO':" in summary and "distal none" in summary
    assert cap._samples.samples == []


def test_a_stalled_i2c_read_is_logged_against_its_sensor(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(angle_mod, "CALIBRATION_FILE", tmp_path / "none.json")
    cap = angle_mod.AngleCapture(FakeSync())
    cap._i2c_1, cap._i2c_2 = FakeI2C(), FakeI2C(stall_s=0.25)
    with caplog.at_level(logging.WARNING, logger=angle_mod.logger.name):
        cap._running = True
        t = threading.Thread(target=cap._capture_loop)
        t.start()
        time.sleep(0.45)
        cap._running = False
        t.join()
    stalls = [r.getMessage() for r in caplog.records if "Angle loop stalled" in r.getMessage()]
    assert len(stalls) == 1
    # The proximal read (bus 2) took the time, not the distal one.
    proximal_ms = int(stalls[0].split("proximal read ")[1].split(" ms")[0])
    distal_ms = int(stalls[0].split("distal read ")[1].split(" ms")[0])
    assert proximal_ms >= 240 and distal_ms < 50


def test_normal_iterations_log_nothing(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(angle_mod, "CALIBRATION_FILE", tmp_path / "none.json")
    cap = angle_mod.AngleCapture(FakeSync())
    cap._i2c_1, cap._i2c_2 = FakeI2C(), FakeI2C()
    with caplog.at_level(logging.WARNING, logger=angle_mod.logger.name):
        cap._running = True
        t = threading.Thread(target=cap._capture_loop)
        t.start()
        time.sleep(0.2)
        cap._running = False
        t.join()
    assert not [r for r in caplog.records if "Angle loop stalled" in r.getMessage()]
