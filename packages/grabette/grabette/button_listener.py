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
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Upper bound on how long a button press waits for the recording to actually
# become live before giving up on the LED feedback. Must cover the fleet
# group-sync lead time (GROUP_START_LEAD_S in grabette-fleet, currently 6s)
# plus worst-case hardware init (OAK-D cold boot ~5-8s).
RECORDING_WAIT_TIMEOUT_S = 20.0


class ButtonListener:
    """Watches the physical button and drives capture start/stop."""

    def __init__(self, backend, task_manager) -> None:
        self._backend = backend
        self._task_manager = task_manager
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
        from grabette.capture_scheduler import get_capture_scheduler

        if self._backend.is_teleop_active:
            self._toggle_teleop_send()
        elif self._backend.is_capturing or get_capture_scheduler().is_scheduled():
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
        # Blink while the start coroutine runs — it may spend several seconds
        # waiting for a fleet group-sync T0 and/or warming up the OAK-D. Go
        # solid only once the recording is genuinely live.
        self._button.led_blink()
        future = asyncio.run_coroutine_threadsafe(self._start_capture_coro(), self._loop)
        try:
            future.result(timeout=RECORDING_WAIT_TIMEOUT_S)
            self._button.led_on()
            logger.info("Button capture started")
        except Exception:
            logger.exception("Button start_capture failed")
            self._button.led_off()

    async def _start_capture_coro(self) -> None:
        """Runs on the event loop: request group sync, then start (scheduled
        or immediate), and block here until the recording is actually live so
        the caller's LED feedback reflects reality."""
        from grabette.capture_scheduler import get_capture_scheduler
        from grabette.fleet_sync import request_group_start
        from grabette.task import episode_id_for

        sm = self._task_manager
        # When this device is grouped, a button press must behave exactly like
        # the fleet "start group recording" button: the GROUP's task (assigned
        # on the fleet) wins and the start is synchronized at the shared T0. So
        # we don't impose our local active task — fleet returns the group's
        # task in the sync response.
        sync = await request_group_start("")
        status = sync.get("status")
        if status == "scheduled":
            gname = sync.get("task_name") or ""
            task_id = sm.get_or_create_task(gname) if gname else sm.active_task_id
            target = datetime.fromisoformat(sync["scheduled_start_utc"])
        elif status == "refused":
            # Fleet says we're in a group session but can't start it now (e.g.
            # a peer is offline). Do NOT silently record a half-rig solo
            # episode — abort so the operator retries once the group is whole.
            raise RuntimeError(f"group start refused by fleet: {sync.get('detail', '')}")
        else:
            # "solo" (not in a session) or "unreachable" (standalone) → local
            # active task, immediate.
            task_id = sm.active_task_id
            target = None

        # Derive the episode id from the shared T0 (not local creation time)
        # so this device's episode folder matches its peers' exactly.
        episode_id = sm.create_episode(task_id, episode_id=episode_id_for(target) if target else None)
        episode_dir = sm.episode_dir(episode_id)

        if target is not None:
            scheduler = get_capture_scheduler()
            await scheduler.schedule(self._backend, sm, episode_dir, target)
            deadline = time.monotonic() + RECORDING_WAIT_TIMEOUT_S
            while not self._backend.is_capturing:
                if time.monotonic() > deadline:
                    raise TimeoutError("scheduled group start did not fire in time")
                await asyncio.sleep(0.1)
            return

        try:
            await self._backend.start_capture(episode_dir)
        except Exception:
            sm.discard_pending_episode()
            raise

    def _do_stop_capture(self) -> None:
        # Acknowledge the press immediately: capture stops at once, but
        # stop_capture then spends a few seconds muxing the mp4s. Blink to
        # show "saving" instead of leaving the LED solid (looks like it's
        # still recording), then go off when the save completes.
        self._button.led_blink()
        future = asyncio.run_coroutine_threadsafe(self._stop_capture_coro(), self._loop)
        try:
            future.result(timeout=30.0)
        except Exception:
            logger.exception("Button stop_capture failed")
        finally:
            self._button.led_off()

    async def _stop_capture_coro(self) -> None:
        from grabette.capture_scheduler import get_capture_scheduler
        from grabette.fleet_sync import notify_group_stop

        scheduler = get_capture_scheduler()
        sm = self._task_manager
        try:
            outcome = await scheduler.cancel_or_wait(self._backend)
        except RuntimeError:
            logger.exception("Button stop: refusing to interrupt in-flight start")
            return
        if outcome == "cancelled":
            sm.discard_pending_episode()
            logger.info("Button stop: cancelled a pending scheduled start")
            return
        if not self._backend.is_capturing:
            logger.warning("Button stop ignored — not capturing")
            return
        status = await self._backend.stop_capture()
        sm.register_episode(getattr(status, "episode_id", None))
        logger.info(
            "Button capture stopped: %.1fs, %d frames",
            status.duration_seconds, status.frame_count,
        )
        asyncio.create_task(notify_group_stop())  # best-effort; don't block on it
