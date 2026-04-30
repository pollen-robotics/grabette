"""LED + button (switch) controller using gpiod v2.

V1 hardware (Grove HAT, legacy): Grove LED Button on D22 connector
    - GPIO22 = LED (active HIGH)
    - GPIO23 = Button (active LOW with internal pull-up)

V2 hardware (custom HAT, rgbd branch):
    - GPIO11 = LED (active HIGH)
    - GPIO10 = Switch (active LOW with internal pull-up)

Same protocol on both — only pin numbers differ. Defaults below match V2.
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
        self, led_pin: int = LED_PIN, button_pin: int = BUTTON_PIN,
    ) -> None:
        self._led_pin = led_pin
        self._button_pin = button_pin

        chip_path = self._find_chip()

        self._led_request = gpiod.request_lines(
            chip_path,
            consumer="grabette-led",
            config={led_pin: gpiod.LineSettings(direction=Direction.OUTPUT)},
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
        self._blink_stop.clear()

        def _blink() -> None:
            state = Value.INACTIVE
            while not self._blink_stop.is_set():
                self._led_request.set_value(self._led_pin, state)
                state = Value.ACTIVE if state == Value.INACTIVE else Value.INACTIVE
                time.sleep(interval)

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
