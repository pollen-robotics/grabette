"""Background button listener for physical start/stop capture control.

Runs in a daemon thread, polls the Grove LED Button, and triggers
capture start/stop through the same code path as the REST API.

The LED hardware is owned here but registered with the backend
(set_led_controller), which drives the capture-state feedback so every
trigger source — button, dashboard, fleet relay — behaves identically:
  - Blink: warming up / saving
  - Solid: recording in progress
  - Off:   idle, ready for capture
This listener only drives the LED directly for the teleop send toggle.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from grabette.session import UNASSIGNED_ID

logger = logging.getLogger(__name__)


class ButtonListener:
    """Watches the physical button and drives capture start/stop."""

    def __init__(self, backend, session_manager) -> None:
        self._backend = backend
        self._session_manager = session_manager
        self._button = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start listening. Must be called from the async event loop thread."""
        self._loop = loop
        try:
            from grabette.hardware.button import LedButton
            self._button = LedButton()
        except Exception as e:
            logger.info("Button not available: %s", e)
            return

        # Hand the LED to the backend so it drives recording feedback on the
        # shared capture path — a capture started from the dashboard (or fleet
        # relay) then lights the LED exactly like a button press. We still own
        # the hardware (lifecycle + cleanup) and keep driving the LED for the
        # teleop send toggle, which is button-only.
        self._backend.set_led_controller(self._button)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="button-listener",
        )
        self._thread.start()
        logger.info("Button listener started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._button is not None:
            # Unregister before cleanup: the backend must not drive the LED
            # (e.g. a stop_capture during daemon shutdown) once its gpiod lines
            # are released below.
            self._backend.set_led_controller(None)
            self._button.led_off()
            self._button.cleanup()
            self._button = None
        logger.info("Button listener stopped")

    def _run(self) -> None:
        """Main loop: each button press dispatches by current daemon mode.

        - Teleop active : press toggles backend.is_teleop_sending
        - Capturing     : press stops the capture
        - Idle          : press starts a capture
        """
        btn = self._button
        try:
            btn.led_off()
            while not self._stop_event.is_set():
                self._wait_for_press()
                if self._stop_event.is_set():
                    break
                self._on_press()
        except Exception:
            logger.exception("Button listener error")
        finally:
            if btn is not None:
                btn.led_off()

    # -- Blocking wait (runs in the button thread) --

    def _wait_for_press(self) -> None:
        """Wait for one button press (press → release + debounce)."""
        btn = self._button
        while not self._stop_event.is_set():
            if btn.is_pressed():
                # Wait for release with debounce
                while btn.is_pressed() and not self._stop_event.is_set():
                    self._stop_event.wait(0.01)
                self._stop_event.wait(0.05)
                return
            self._stop_event.wait(0.01)

    # -- Press dispatch --

    def _on_press(self) -> None:
        """Decide what a button press means given the current daemon mode."""
        if self._backend.is_teleop_active:
            self._toggle_teleop_send()
        elif self._backend.is_capturing:
            self._do_stop_capture()
        else:
            self._do_start_capture()

    def _toggle_teleop_send(self) -> None:
        new_state = not self._backend.is_teleop_sending
        self._backend.set_teleop_send(new_state)
        if new_state:
            self._button.led_on()
            logger.info("Button — teleop sending ON")
        else:
            self._button.led_off()
            logger.info("Button — teleop sending OFF (reposition)")

    # -- Capture actions (scheduled on the async event loop) --

    def _do_start_capture(self) -> None:
        # LED feedback (blink during warmup → solid when live → off on error) is
        # driven by backend.start_capture so every trigger source behaves alike.
        episode_id = self._session_manager.create_episode(session_id=UNASSIGNED_ID)
        episode_dir = self._session_manager.episode_dir(episode_id)

        future = asyncio.run_coroutine_threadsafe(
            self._backend.start_capture(episode_dir), self._loop,
        )
        try:
            future.result(timeout=20.0)
            logger.info("Button capture started: %s", episode_id)
        except Exception:
            logger.exception("Button start_capture failed")
            self._session_manager.discard_pending_episode()

    def _do_stop_capture(self) -> None:
        if not self._backend.is_capturing:
            logger.warning("Button stop ignored — not capturing")
            return

        # LED feedback (blink while muxing → off when saved/on error) is driven
        # by backend.stop_capture so every trigger source behaves alike.
        future = asyncio.run_coroutine_threadsafe(
            self._backend.stop_capture(), self._loop,
        )
        try:
            status = future.result(timeout=30.0)
            self._session_manager.register_episode(
                getattr(status, "session_id", None)
            )
            logger.info(
                "Button capture stopped: %.1fs, %d frames",
                status.duration_seconds, status.frame_count,
            )
        except Exception:
            logger.exception("Button stop_capture failed")
