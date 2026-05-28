"""LED + button (switch) controller using gpiod v2.

V1 hardware (Grove HAT, legacy): Grove LED Button on D22 connector
    - GPIO22 = LED (active HIGH)
    - GPIO23 = Button (active LOW with internal pull-up)

V2 hardware (custom HAT, rgbd branch):
    - GPIO11 = LED (active LOW — wire LOW lights the LED)
    - GPIO10 = Switch (active LOW with internal pull-up)

The LED polarity differs between V1 and V2. We pass `active_low=True` to
gpiod for V2 so the API stays the same (`led_on()` lights the LED on both
boards). Defaults below match V2.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import gpiod
from gpiod.line import Bias, Direction, Value

logger = logging.getLogger(__name__)


class LedButton:
    """LED + button/switch controller using gpiod v2."""

    LED_PIN = 11
    BUTTON_PIN = 10
    # Pi 4 and earlier use gpiochip0, Pi 5 uses gpiochip4
    CHIP_PATHS = ["/dev/gpiochip0", "/dev/gpiochip4"]

    def __init__(
        self,
        led_pin: int = LED_PIN,
        button_pin: int = BUTTON_PIN,
        led_active_low: bool = True,  # V2 wiring; set False for V1 Grove HAT
    ) -> None:
        self._led_pin = led_pin
        self._button_pin = button_pin

        chip_path = self._find_chip()

        self._led_request = gpiod.request_lines(
            chip_path,
            consumer="grabette-led",
            config={led_pin: gpiod.LineSettings(
                direction=Direction.OUTPUT,
                active_low=led_active_low,
            )},
        )
        self._button_request = gpiod.request_lines(
            chip_path,
            consumer="grabette-button",
            config={
                button_pin: gpiod.LineSettings(
                    direction=Direction.INPUT,
                    bias=Bias.PULL_UP,
                )
            },
        )

        self._blink_thread: threading.Thread | None = None
        self._blink_stop = threading.Event()

    @classmethod
    def _find_chip(cls) -> str:
        for path in cls.CHIP_PATHS:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"No GPIO chip found. Tried: {', '.join(cls.CHIP_PATHS)}"
        )

    def led_on(self) -> None:
        self._blink_stop.set()
        self._led_request.set_value(self._led_pin, Value.ACTIVE)

    def led_off(self) -> None:
        self._blink_stop.set()
        self._led_request.set_value(self._led_pin, Value.INACTIVE)

    def led_blink(self, interval: float = 0.3) -> None:
        """Start the LED blinking at the given interval (seconds per state).

        Diagnostic: the thread records each tick's wall time and logs a
        summary on exit (called when led_on / led_off sets _blink_stop).
        Lets us spot stalls — if a single state is held much longer than
        `interval`, the thread was preempted or wedged.

        Known issue: during a sync_start that involves OAK-D cold init,
        depthai's C extension holds the Python GIL across long stretches
        (~2-3 s) while uploading the pipeline blob over USB. The blink
        thread's time.sleep returns on schedule but it can't re-acquire
        the GIL until depthai releases it, so the LED appears to freeze
        on its last state for the duration of the stall (visible as
        "blink, then ~3 s of off or on, then resume").
        Confirmed by the diagnostic log: tick count ≈ 26 of expected ~34
        in 10 s, longest_gap ≈ 2.8 s vs the 300 ms interval — i.e. one
        single tick was preempted by 2.5 s.
        The actual recording is unaffected; it's a purely cosmetic LED
        artifact. Fixing it would require moving the LED out of Python
        (hardware PWM via /sys/class/pwm, or a separate process); we
        accepted the cosmetic issue and live with the stall for now.
        """
        # If a previous blink thread is still alive, stop it first.
        # Otherwise we'd have two threads racing on the same GPIO.
        if self._blink_thread is not None and self._blink_thread.is_alive():
            self._blink_stop.set()
            self._blink_thread.join(timeout=interval * 2)
        self._blink_stop.clear()

        log = logger  # capture in closure

        def _blink() -> None:
            state = Value.INACTIVE
            ticks = 0
            t_start = time.monotonic()
            t_prev = t_start
            longest_gap = 0.0
            while not self._blink_stop.is_set():
                self._led_request.set_value(self._led_pin, state)
                state = Value.ACTIVE if state == Value.INACTIVE else Value.INACTIVE
                time.sleep(interval)
                now = time.monotonic()
                gap = now - t_prev
                if gap > longest_gap:
                    longest_gap = gap
                t_prev = now
                ticks += 1
            elapsed_ms = (time.monotonic() - t_start) * 1000
            expected_ticks = max(1, int(elapsed_ms / (interval * 1000)))
            log.info(
                "led_blink: %d ticks in %.0fms (expected ~%d, "
                "longest_gap=%.0fms, interval=%.0fms)",
                ticks, elapsed_ms, expected_ticks,
                longest_gap * 1000, interval * 1000,
            )

        self._blink_thread = threading.Thread(target=_blink, daemon=True)
        self._blink_thread.start()

    def is_pressed(self) -> bool:
        """Button is active-low: pressed = LOW = INACTIVE."""
        return self._button_request.get_value(self._button_pin) == Value.INACTIVE

    def wait_for_press(self, debounce_ms: int = 50) -> None:
        """Block until button is pressed and then released."""
        self.wait_for_press_down()
        self.wait_for_release(debounce_ms)

    def wait_for_press_down(self) -> None:
        """Block until button transitions from unpressed to pressed."""
        # If already pressed, wait for release first
        while self._button_request.get_value(self._button_pin) == Value.INACTIVE:
            time.sleep(0.01)
        # Wait for press
        while self._button_request.get_value(self._button_pin) == Value.ACTIVE:
            time.sleep(0.01)

    def wait_for_release(self, debounce_ms: int = 50) -> None:
        """Wait for button release with debounce delay."""
        while self._button_request.get_value(self._button_pin) == Value.INACTIVE:
            time.sleep(0.01)
        time.sleep(debounce_ms / 1000)

    def cleanup(self) -> None:
        self._blink_stop.set()
        self._led_request.set_value(self._led_pin, Value.INACTIVE)
        self._led_request.release()
        self._button_request.release()
