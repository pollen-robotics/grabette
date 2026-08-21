"""Tactile sensor capture for DFRobot SEN0704/SEN0705 arrays over Modbus RTU UART.

Each sensor is a Modbus-RTU slave; the Pi is the master. Multiple sensors share
ONE UART bus (multi-drop) and are addressed individually by their Modbus device
address, so a single serial port polls them sequentially. Sensors on the same bus
may have different shapes (e.g. a 6x6 SEN0704 next to a 4x8 SEN0705); each
sensor's shape (rows, cols) is supplied explicitly in the config.

A frame is one READ_INPUT_REGISTERS (function 0x04) call at register 0x0007 with
length = rows * cols; each cell is a 12-bit ADC value (0-4095) encoded big-endian.
The flat register block is reshaped row-major directly into a rows x cols grid.
Framing/decoding mirror the reference visualizer (tactile/src/sensor_visualizer.py).
"""

import logging
import threading
import time
from dataclasses import dataclass, field

from .sync import SyncManager

logger = logging.getLogger(__name__)

_FUNC_READ_INPUT_REGISTERS = 0x04
_START_REGISTER = 0x0007
# Small settle before flushing the bus after a failed/partial transaction.
_RECOVER_SLEEP_S = 0.0002


def _modbus_crc(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def _build_request(address: int, count: int) -> bytes:
    packet = bytes([
        address,
        _FUNC_READ_INPUT_REGISTERS,
        (_START_REGISTER >> 8) & 0xFF,
        _START_REGISTER & 0xFF,
        (count >> 8) & 0xFF,
        count & 0xFF,
    ])
    crc = _modbus_crc(packet)
    return packet + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


@dataclass
class TactileSamples:
    """Collected tactile samples from a capture session, keyed by device address.

    Each sensor carries its own shape (rows, cols) so mixed arrays on one bus are
    stored unambiguously. Sample ``value`` is a row-major rows x cols grid.
    """
    shapes: dict[int, tuple[int, int]] = field(default_factory=dict)  # addr -> (rows, cols)
    sensors: dict[int, list[dict]] = field(default_factory=dict)  # addr -> [{cts, value}]

    @property
    def count(self) -> int:
        return sum(len(s) for s in self.sensors.values())

    def effective_hz(self, addr: int) -> float:
        """Measured per-sensor rate from the first/last sample timestamps (ms)."""
        s = self.sensors.get(addr, [])
        if len(s) < 2:
            return 0.0
        span_ms = s[-1]["cts"] - s[0]["cts"]
        return (len(s) - 1) / (span_ms / 1000.0) if span_ms > 0 else 0.0


class TactileCapture:
    """Captures pressure data from one or more DFRobot SEN0704 tactile sensors.

    All sensors share a single UART bus; each is polled by its Modbus address.
    ``sensors`` maps device address -> (rows, cols). The bus is polled as fast as
    possible (no pacing), so the effective per-sensor rate is ~bus_throughput / N.
    """

    def __init__(
        self,
        sync_manager: SyncManager,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 115200,
        sensors: dict[int, tuple[int, int]] | None = None,
        timeout: float = 0.02,
    ):
        self.sync = sync_manager
        self.port = port
        self.baudrate = baudrate
        self.shapes: dict[int, tuple[int, int]] = dict(sensors or {1: (6, 6)})
        self.timeout = timeout

        self._counts = {a: r * c for a, (r, c) in self.shapes.items()}
        self._requests = {a: _build_request(a, self._counts[a]) for a in self.shapes}
        self._response_sizes = {a: 3 + self._counts[a] * 2 + 2 for a in self.shapes}

        self._samples = TactileSamples(shapes=dict(self.shapes))
        self._running = False
        self._thread: threading.Thread | None = None
        self._serial = None

    @property
    def addresses(self) -> list[int]:
        return list(self.shapes.keys())

    def init_sensors(self) -> None:
        import serial

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        logger.info(
            "Tactile sensors initialized (port=%s, shapes=%s)",
            self.port, self.shapes,
        )

    def _read_exactly(self, size: int) -> bytearray | None:
        data = bytearray(size)
        view = memoryview(data)
        pos = 0
        while pos < size:
            n = self._serial.readinto(view[pos:])
            if n == 0:  # serial timeout
                return None
            pos += n
        return data

    def _recover_bus(self) -> None:
        time.sleep(_RECOVER_SLEEP_S)
        if self._serial is not None:
            self._serial.reset_input_buffer()

    def _read_frame(self, addr: int) -> list[list[int]] | None:
        """One Modbus transaction -> row-major rows x cols grid, or None on failure."""
        count = self._counts[addr]
        self._serial.write(self._requests[addr])
        response = self._read_exactly(self._response_sizes[addr])
        if response is None:
            return None
        if (
            response[0] != addr
            or response[1] != _FUNC_READ_INPUT_REGISTERS
            or response[2] != count * 2
        ):
            return None
        recv_crc = response[-2] | (response[-1] << 8)
        if _modbus_crc(response[:-2]) != recv_crc:
            return None

        data = response[3:3 + count * 2]
        flat = [(data[2 * i] << 8) | data[2 * i + 1] for i in range(count)]
        rows, cols = self.shapes[addr]
        return [flat[r * cols:(r + 1) * cols] for r in range(rows)]

    def read_latest(self) -> dict[int, list[list[int]]]:
        """Read one frame per sensor directly — for idle live view."""
        out: dict[int, list[list[int]]] = {}
        if self._serial is None:
            return out
        for addr in self.addresses:
            try:
                grid = self._read_frame(addr)
                if grid is not None:
                    out[addr] = grid
                else:
                    self._recover_bus()
            except Exception:
                pass
        return out

    def _capture_loop(self) -> None:
        # Poll continuously with no pacing (matches the reference visualizer):
        # the bus runs at max throughput, giving each sensor ~throughput / N Hz.
        error_count = 0
        read_count = 0

        while self._running:
            read_count += 1
            for addr in self.addresses:
                if not self._running:
                    break
                try:
                    ts = self.sync.get_timestamp_ms()
                    cells = self._read_frame(addr)
                    if cells is None:
                        error_count += 1
                        self._recover_bus()
                        continue
                    self._samples.sensors[addr].append({"cts": ts, "value": cells})
                except Exception:
                    error_count += 1

        logger.info("Tactile: %d loops, %d errors", read_count, error_count)

    def start_capture(self) -> None:
        if self._running:
            raise RuntimeError("Tactile capture already running")
        if self._serial is None:
            raise RuntimeError("Sensors not initialized. Call init_sensors() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before tactile capture")

        self._samples = TactileSamples(shapes=dict(self.shapes))
        for addr in self.addresses:
            self._samples.sensors[addr] = []
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> TactileSamples:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

        return self._samples

    @property
    def sample_count(self) -> int:
        return self._samples.count
