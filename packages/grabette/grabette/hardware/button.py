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

import os
import threading
import time

import gpiod
from gpiod.line import Bias, Direction, Value


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
        def _blink(stop: threading.Event) -> None:
            state = Value.INACTIVE
            while not stop.is_set():
                self._led_request.set_value(self._led_pin, state)
                state = Value.ACTIVE if state == Value.INACTIVE else Value.INACTIVE
                time.sleep(interval)

        self._start_pattern(_blink)

    def led_pulses(self, count: int = 3, on: float = 0.1, off: float = 0.1,
                   gap: float = 1.0) -> None:
        """Repeat a burst of `count` short pulses, then hold dark for `gap`.

        Distinguishable at a glance from every other pattern (which are all
        even-duty blinks): a burst-then-pause reads as "this device is faulty",
        not as "this device is busy". Used for the hardware-error state, where
        the LED is the only feedback an operator has while the grabette sits on
        a bench with no screen.

        The gap is slept in short slices so a state change (led_on/off/blink)
        takes effect promptly instead of waiting out a full dark period.
        """
        def _pulse(stop: threading.Event) -> None:
            while not stop.is_set():
                for _ in range(max(1, count)):
                    if stop.is_set():
                        break
                    self._led_request.set_value(self._led_pin, Value.ACTIVE)
                    time.sleep(on)
                    self._led_request.set_value(self._led_pin, Value.INACTIVE)
                    time.sleep(off)
                waited = 0.0
                while waited < gap and not stop.is_set():
                    time.sleep(min(0.05, gap - waited))
                    waited += 0.05

        self._start_pattern(_pulse)

    def _start_pattern(self, target) -> None:
        """Run target(stop_event) as THE blink thread, replacing any previous one.

        Each thread gets its OWN stop Event, captured as an argument rather than
        read off self: a pattern switch installs a fresh event, and a thread that
        read the shared attribute would then poll the NEW (unset) one and never
        exit — two threads would drive the same line and the pattern would be
        neither. Joining the outgoing thread before starting the next one keeps
        exactly one writer on the LED; the join is bounded so a wedged GPIO write
        can't block the caller."""
        prev, prev_stop = self._blink_thread, self._blink_stop
        prev_stop.set()
        if prev is not None and prev.is_alive():
            prev.join(timeout=1.0)
        stop = threading.Event()
        self._blink_stop = stop
        self._blink_thread = threading.Thread(target=target, args=(stop,), daemon=True)
        self._blink_thread.start()

    def is_pressed(self) -> bool:
        """Button is active-low: pressed = LOW = INACTIVE."""
        return self._button_request.get_value(self._button_pin) == Value.INACTIVE

    def cleanup(self) -> None:
        self._blink_stop.set()
        self._led_request.set_value(self._led_pin, Value.INACTIVE)
        self._led_request.release()
        self._button_request.release()
