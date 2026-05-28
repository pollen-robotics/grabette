"""Background button listener for physical start/stop capture control.

Runs in a daemon thread, polls the Grove LED Button, and dispatches
multi-device sync start/stop by HTTP-looping back through the local
/api/sync endpoints. Going through the sync orchestrator means a single
button press fans out to all configured peers with NTP-precision T0
scheduling, and degrades to local-only capture when no peers are
configured — same code path either way.

LED feedback:
  - Blink: daemon starting / sensors initializing / recording-being-saved
  - Off:   idle, ready for capture
  - Solid: recording in progress
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import httpx

logger = logging.getLogger(__name__)


# How long to ignore subsequent button presses after a press is
# dispatched. Two purposes: (1) bouncy button — the in-loop debounce
# already covers ms-scale bounce, but slower repeated taps could still
# fire twice. (2) Cross-device race — when two grabettes' buttons are
# pressed within ~1 s, both fan out sync/start concurrently. The
# server-side 409 guard catches the conflict, but the local deadtime
# prevents this device from also re-triggering during the race window.
PRESS_DEADTIME_S = 1.0

# Timeouts for the loopback HTTP calls. Start fans out to peers and
# does preflight (timedatectl on each peer ~500 ms) — give it room.
# Stop on grabette can include OAK-D + ffmpeg teardown (5-10 s).
SYNC_START_TIMEOUT_S = 20.0
SYNC_STOP_TIMEOUT_S = 30.0


class ButtonListener:
    """Watches the physical button and drives sync-mode capture start/stop."""

    def __init__(
        self, backend, session_manager,
        daemon_port: int = 8000,
    ) -> None:
        self._backend = backend
        self._session_manager = session_manager
        self._daemon_port = daemon_port
        self._button = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Monotonic timestamp of the most recent dispatched press
        # (regardless of success/failure). PRESS_DEADTIME_S after this
        # we accept the next press.
        self._last_press_t: float = 0.0

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
        # Deadtime gate: ignore presses too soon after the last one.
        now = time.monotonic()
        if now - self._last_press_t < PRESS_DEADTIME_S:
            logger.debug(
                "Button press ignored (within %.1f s deadtime)",
                PRESS_DEADTIME_S,
            )
            return
        self._last_press_t = now

        if self._backend.is_teleop_active:
            self._toggle_teleop_send()
        elif self._backend.is_capturing:
            self._do_sync_stop()
        else:
            self._do_sync_start()

    def _toggle_teleop_send(self) -> None:
        new_state = not self._backend.is_teleop_sending
        self._backend.set_teleop_send(new_state)
        if new_state:
            self._button.led_on()
            logger.info("Button — teleop sending ON")
        else:
            self._button.led_off()
            logger.info("Button — teleop sending OFF (reposition)")

    # -- Capture actions (HTTP loopback to local /api/sync/*) --

    def _sync_url(self, path: str) -> str:
        return f"http://localhost:{self._daemon_port}{path}"

    def _do_sync_start(self) -> None:
        """Start a multi-device sync episode via the local /api/sync/start.

        Going through the orchestrator means the same scheduling +
        preflight + fan-out + rollback logic the REST API uses. With no
        peers configured, the endpoint degrades to a single-device
        scheduled start (still goes through T₀, but no fan-out).
        """
        # Blink while the request runs — preflight + fan-out + slow
        # hardware init (OAK-D ~5-8 s) all happen between dispatch and
        # actual recording start. Go solid only on success.
        self._button.led_blink()
        try:
            with httpx.Client(timeout=SYNC_START_TIMEOUT_S) as c:
                r = c.post(self._sync_url("/api/sync/start"))
            if r.status_code == 200:
                self._button.led_on()
                data = r.json()
                peers = data.get("peers", []) or []
                logger.info(
                    "Button sync start: local=%s, %d peer(s)",
                    data.get("local_episode_id"), len(peers),
                )
            else:
                logger.warning(
                    "Button sync start failed: HTTP %d — %s",
                    r.status_code, r.text[:300],
                )
                self._button.led_off()
        except Exception:
            logger.exception("Button sync start raised")
            self._button.led_off()

    def _do_sync_stop(self) -> None:
        """Stop the multi-device sync episode via /api/sync/stop.

        Mirrors _do_sync_start. The orchestrator will stop the local
        scheduler AND fan out /api/episodes/stop to all configured
        peers, even if our own state is somehow already idle.
        """
        # Blink to show "saving" — stop_capture spends seconds muxing
        # the mp4s and tearing down OAK-D.
        self._button.led_blink()
        try:
            with httpx.Client(timeout=SYNC_STOP_TIMEOUT_S) as c:
                r = c.post(self._sync_url("/api/sync/stop"))
            self._button.led_off()
            if r.status_code == 200:
                data = r.json()
                local = data.get("local") or {}
                peers = data.get("peers", []) or []
                logger.info(
                    "Button sync stop: local=%s, %d peer(s)",
                    local.get("status") or
                    f"{local.get('duration_seconds')}s {local.get('frame_count')}fr",
                    len(peers),
                )
            else:
                logger.warning(
                    "Button sync stop failed: HTTP %d — %s",
                    r.status_code, r.text[:300],
                )
        except Exception:
            logger.exception("Button sync stop raised")
            self._button.led_off()
