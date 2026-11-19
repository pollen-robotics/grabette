"""A stalled angle iteration is logged with where its time went."""
import logging
import threading
import time

from grabette.hardware import angle as angle_mod


class FakeSync:
    is_started = True

    def get_timestamp_ms(self):
        return time.monotonic() * 1000


class FakeI2C:
    def __init__(self, stall_s=0.0):
        self.stall_s = stall_s

    def writeto_then_readfrom(self, addr, out, result):
        if self.stall_s:
            stall, self.stall_s = self.stall_s, 0.0
            time.sleep(stall)  # one slow transfer, like the 593 ms hole
        result[0], result[1] = 0x01, 0x00


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
