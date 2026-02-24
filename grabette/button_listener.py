"""Background button listener for physical start/stop capture control.

Runs in a daemon thread, polls the Grove LED Button, and triggers
capture start/stop through the same code path as the REST API.

LED feedback:
  - Blink: daemon starting / sensors initializing
  - Off:   idle, ready for capture
  - Solid: recording in progress
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

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
            self._button.led_off()
            self._button.cleanup()
            self._button = None
        logger.info("Button listener stopped")

    def _run(self) -> None:
        """Main loop: wait for press → start capture → wait for press → stop."""
        btn = self._button
        try:
            btn.led_off()
            while not self._stop_event.is_set():
                self._wait_for_start()
                if self._stop_event.is_set():
                    break

                self._do_start_capture()
                if self._stop_event.is_set():
                    break

                self._wait_for_stop()
                if self._stop_event.is_set():
                    break

                self._do_stop_capture()
        except Exception:
            logger.exception("Button listener error")
        finally:
            if btn is not None:
                btn.led_off()

    # -- Blocking waits (run in the button thread) --

    def _wait_for_start(self) -> None:
        """Wait for button press to start capture."""
        btn = self._button
        # Poll with stop_event check so we can exit cleanly
        # Wait for press-down
        while not self._stop_event.is_set():
            if btn.is_pressed():
                btn.led_on()
                logger.info("Button pressed — starting capture")
                # Wait for release with debounce
                while btn.is_pressed() and not self._stop_event.is_set():
                    self._stop_event.wait(0.01)
                self._stop_event.wait(0.05)  # debounce
                return
            self._stop_event.wait(0.01)

    def _wait_for_stop(self) -> None:
        """Wait for button press to stop capture."""
        btn = self._button
        while not self._stop_event.is_set():
            if btn.is_pressed():
                btn.led_off()
                logger.info("Button pressed — stopping capture")
                while btn.is_pressed() and not self._stop_event.is_set():
                    self._stop_event.wait(0.01)
                self._stop_event.wait(0.05)
                return
            self._stop_event.wait(0.01)

    # -- Capture actions (scheduled on the async event loop) --

    def _do_start_capture(self) -> None:
        if self._backend.is_capturing:
            logger.warning("Button start ignored — already capturing")
            return

        session_id = self._session_manager.create_session()
        session_dir = self._session_manager._session_dir(session_id)

        future = asyncio.run_coroutine_threadsafe(
            self._backend.start_capture(session_dir), self._loop,
        )
        try:
            future.result(timeout=10.0)
            self._button.led_on()
            logger.info("Button capture started: %s", session_id)
        except Exception:
            logger.exception("Button start_capture failed")
            self._button.led_off()

    def _do_stop_capture(self) -> None:
        if not self._backend.is_capturing:
            logger.warning("Button stop ignored — not capturing")
            return

        future = asyncio.run_coroutine_threadsafe(
            self._backend.stop_capture(), self._loop,
        )
        try:
            status = future.result(timeout=30.0)
            self._button.led_off()
            logger.info(
                "Button capture stopped: %.1fs, %d frames",
                status.duration_seconds, status.frame_count,
            )
        except Exception:
            logger.exception("Button stop_capture failed")
            self._button.led_off()
