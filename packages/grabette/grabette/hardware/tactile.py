"""Tactile sensor capture for DFRobot SEN0704/SEN0705 arrays over Modbus RTU UART.

Each sensor is a Modbus-RTU slave; the Pi is the master. Multiple sensors share
ONE UART bus (multi-drop) and are addressed individually by their Modbus device
address, so a single serial port + RtuMaster polls them sequentially. Sensors on
the same bus may have different shapes (e.g. a 6x6 SEN0704 next to a 4x8 SEN0705);
each sensor's shape is auto-detected from its model register.

A frame is one READ_INPUT_REGISTERS call at register 0x0007 (length = array size);
each cell is a 12-bit ADC value (0-4095). The flat register block is row-major
with rows in reverse order relative to the physical grid (see DFRobot get_datas),
so we reshape into a canonical rows x cols grid (row-major, top-to-bottom).
Protocol reference: DFRobot_Tactile_Sensor.py (github.com/DFRobot/DFRobot_TactileSensor).
"""

import logging
import threading
import time
from dataclasses import dataclass, field

from .sync import SyncManager

logger = logging.getLogger(__name__)

# Input registers
_INPUTREG_VERSION = 0x0005
_INPUTREG_GETDATAS = 0x0007
_INPUTREG_MODEL = 0x002B
# Holding registers
_HOLDINGREG_THLD = 0x0006
_HOLDINGREG_SAMPLE_RATE = 0x0008

_SAMPLE_RATE_20HZ = 0
_SAMPLE_RATE_50HZ = 1

# array cell-count -> (rows, cols). Mirrors the DFRobot driver's x/y mapping
# (36 -> 6x6, 32 -> 4x8). Extend here if new array variants are added.
_ARRAY_SHAPES = {32: (4, 8), 36: (6, 6)}
# model register value -> array cell count.
_MODEL_TO_ARRAY = {0: 32, 1: 36}


@dataclass
class TactileSamples:
    """Collected tactile samples from a capture session, keyed by device address.

    Each sensor carries its own shape (rows, cols) so mixed arrays on one bus are
    stored unambiguously. Sample ``value`` is a canonical row-major rows x cols grid.
    """
    sample_rate_hz: int = 50
    shapes: dict[int, tuple[int, int]] = field(default_factory=dict)  # addr -> (rows, cols)
    arrays: dict[int, int] = field(default_factory=dict)  # addr -> cell count
    sensors: dict[int, list[dict]] = field(default_factory=dict)  # addr -> [{cts, value}]

    @property
    def count(self) -> int:
        return sum(len(s) for s in self.sensors.values())


class TactileCapture:
    """Captures pressure data from one or more DFRobot SEN0704 tactile sensors.

    All sensors share a single UART bus; each is polled by its Modbus address.
    """

    def __init__(
        self,
        sync_manager: SyncManager,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 115200,
        addresses: list[int] | None = None,
        array: int = 36,
        sample_rate_hz: int = 50,
        threshold: int = 50,
    ):
        self.sync = sync_manager
        self.port = port
        self.baudrate = baudrate
        self.addresses = addresses or [1]
        self.array = array  # fallback cell count if model auto-detect fails
        self.sample_rate_hz = sample_rate_hz
        self.threshold = threshold

        # Per-sensor shape, auto-detected from the model register in init_sensors.
        self._arrays: dict[int, int] = {}
        self._shapes: dict[int, tuple[int, int]] = {}

        self._samples = TactileSamples(sample_rate_hz=sample_rate_hz)
        self._running = False
        self._thread: threading.Thread | None = None
        self._serial = None
        self._master = None

    def init_sensors(self) -> None:
        import serial
        import modbus_tk.defines as cst  # noqa: F401  (imported lazily for parity)
        from modbus_tk import modbus_rtu

        self._serial = serial.Serial(
            port=self.port, baudrate=self.baudrate,
            bytesize=8, parity="N", stopbits=1,
        )
        self._master = modbus_rtu.RtuMaster(self._serial)
        self._master.set_timeout(1.0 / self.sample_rate_hz)

        rate = _SAMPLE_RATE_50HZ if self.sample_rate_hz >= 50 else _SAMPLE_RATE_20HZ
        for addr in self.addresses:
            array = self.array
            try:
                model = self._read_reg(addr, _INPUTREG_MODEL, 1)[0]
                array = _MODEL_TO_ARRAY.get(model, self.array)
                if model not in _MODEL_TO_ARRAY:
                    logger.warning(
                        "Tactile addr %d: unknown model %d, falling back to array=%d",
                        addr, model, self.array,
                    )
                self._write_reg(addr, _HOLDINGREG_THLD, self.threshold)
                self._write_reg(addr, _HOLDINGREG_SAMPLE_RATE, rate)
            except Exception as e:
                logger.warning("Tactile addr %d init failed: %s", addr, e)
            self._arrays[addr] = array
            self._shapes[addr] = _ARRAY_SHAPES.get(array, (1, array))
        logger.info(
            "Tactile sensors initialized at %d Hz (port=%s, shapes=%s)",
            self.sample_rate_hz, self.port,
            {a: self._shapes[a] for a in self.addresses},
        )

    def _read_reg(self, addr: int, reg: int, length: int) -> list[int]:
        import modbus_tk.defines as cst
        return list(self._master.execute(addr, cst.READ_INPUT_REGISTERS, reg, length))

    def _write_reg(self, addr: int, reg: int, value: int) -> None:
        import modbus_tk.defines as cst
        # SEN0704 expects the 16-bit payload byte-swapped (see DFRobot driver).
        payload = [((value >> 8) & 0xFF) | ((value & 0xFF) << 8)]
        self._master.execute(addr, cst.WRITE_MULTIPLE_REGISTERS, reg, output_value=payload)

    def _read_frame(self, addr: int) -> list[list[int]]:
        array = self._arrays.get(addr, self.array)
        flat = self._read_reg(addr, _INPUTREG_GETDATAS, array)
        return self._reshape(addr, flat)

    def _reshape(self, addr: int, flat: list[int]) -> list[list[int]]:
        """Map the flat register block to a canonical row-major rows x cols grid.

        The DFRobot register order is row-major but with rows reversed relative to
        the physical grid, so grid[r][c] = flat[(rows - 1 - r) * cols + c].
        """
        rows, cols = self._shapes.get(addr, (1, len(flat)))
        return [
            [flat[(rows - 1 - r) * cols + c] for c in range(cols)]
            for r in range(rows)
        ]

    def read_latest(self) -> dict[int, list[list[int]]]:
        """Read one frame per sensor directly — for idle live view."""
        out: dict[int, list[list[int]]] = {}
        if self._master is None:
            return out
        for addr in self.addresses:
            try:
                out[addr] = self._read_frame(addr)
            except Exception:
                pass
        return out

    def _capture_loop(self) -> None:
        error_count = 0
        read_count = 0
        sample_interval = 1.0 / self.sample_rate_hz

        while self._running:
            loop_start = time.monotonic()
            read_count += 1
            for addr in self.addresses:
                try:
                    ts = self.sync.get_timestamp_ms()
                    cells = self._read_frame(addr)
                    self._samples.sensors[addr].append({"cts": ts, "value": cells})
                except Exception:
                    error_count += 1

            elapsed = time.monotonic() - loop_start
            sleep_time = sample_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Tactile: %d loops, %d errors", read_count, error_count)

    def start_capture(self) -> None:
        if self._running:
            raise RuntimeError("Tactile capture already running")
        if self._master is None:
            raise RuntimeError("Sensors not initialized. Call init_sensors() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before tactile capture")

        self._samples = TactileSamples(
            sample_rate_hz=self.sample_rate_hz,
            shapes=dict(self._shapes),
            arrays=dict(self._arrays),
        )
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

        if self._master is not None:
            try:
                self._master.close()
            except Exception:
                pass
            self._master = None
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
