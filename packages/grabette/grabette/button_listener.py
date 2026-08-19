"""Background button listener for physical start/stop capture control.

Runs in a daemon thread, polls the Grove LED Button, and triggers
capture start/stop through the same code path as the REST API.

LED feedback is driven by the device's ACTUAL capture state (a separate monitor
thread), not by which button was pressed — so in a group/bimanual recording
BOTH grabettes reflect their own status, even the peer started via the fleet:
  - Off:        idle
  - Blink:      initializing (warming up / waiting for the shared T0)
  - Solid:      recording in progress
  - Fast blink: stopping (from the stop until the capture is fully torn down)
  - 1 pulse, pause, repeat:  busy with dataset work (upload / conversion) —
    recording is refused until it finishes (see Backend.busy_reason)
  - 3 pulses, pause, repeat: hardware fault — this device REFUSES to record
    (see Backend.hardware_error; today: no usable OAK-D offline calibration)
(Teleop mode reuses the LED: solid = sending, off = repositioning.)
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

# How long the pressed device blinks right after a STOP press, as an
# acknowledgment that the command registered (the state monitor would otherwise
# just switch it off). Local to the pressed device — the peer isn't affected.
STOP_ACK_S = 1.2

# Debounce for the LED going 'off': a cold start briefly reads all-idle between
# internal states, which must NOT flash the LED off mid-initialization. An 'off'
# has to persist this long (while blinking/solid) before it's applied.
_OFF_DEBOUNCE_S = 1.0

# Hardware-fault pattern: N short pulses, then a dark gap, repeating. Read off
# the LED alone, it must not be confusable with the busy patterns — hence a burst
# rather than yet another duty cycle.
_ERROR_PULSES = 3
_ERROR_PULSE_ON_S = 0.1
_ERROR_PULSE_OFF_S = 0.1
_ERROR_GAP_S = 1.0

# Busy pattern: the SAME burst shape with a single pulse and a longer pause, so
# the two read as one family — count the blips: one = busy, three = broken. A
# busy device must not look idle (dark), or an operator picks it up and presses.
_BUSY_PULSES = 1
_BUSY_GAP_S = 1.5


class ButtonListener:
    """Watches the physical button and drives capture start/stop."""

    def __init__(self, backend, task_manager) -> None:
        self._backend = backend
        self._task_manager = task_manager
        self._button = None
        self._thread: threading.Thread | None = None
        self._led_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_ack_until = 0.0  # monotonic deadline: blink to ack a stop press

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start listening. Must be called from the async event loop thread."""
        self._loop = loop
        try:
            from grabette.hardware.button import LedButton
            self._button = LedButton()
        except Exception as e:
            logger.info("Button not available: %s", e)
            return

        # The LED is driven ONLY from here — the _led_monitor thread below reads
        # the backend's capture state and applies the matching pattern. We do not
        # register the LED with the backend (set_led_controller): letting it also
        # drive the LED imperatively would fight the monitor, e.g. plain blink vs
        # our fast blink during a stop.
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="button-listener",
        )
        self._thread.start()
        # Separate thread that keeps the LED in sync with the capture state,
        # regardless of how the recording was triggered (button, dashboard, or
        # group fan-out) — so a peer grabette lights up too.
        self._led_thread = threading.Thread(
            target=self._led_monitor, daemon=True, name="button-led",
        )
        self._led_thread.start()
        logger.info("Button listener started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._led_thread is not None:
            self._led_thread.join(timeout=2.0)
            self._led_thread = None
        if self._button is not None:
            # The monitor thread is already joined above, so nothing can touch
            # the LED between here and the gpiod lines being released.
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
        # This thread does NOT touch the LED: _led_monitor is the single writer
        # (see start()). It used to switch the LED off on entry and on exit, which
        # was harmless while every run began and ended idle — but a device that
        # comes up in a hardware fault starts on the error pattern, and a stray
        # led_off() here would race the monitor, win, and leave the LED dark with
        # the monitor believing it had already applied the pattern. Teardown is
        # covered by stop(), which switches the LED off after joining both threads.
        try:
            while not self._stop_event.is_set():
                self._wait_for_press()
                if self._stop_event.is_set():
                    break
                self._on_press()
        except Exception:
            logger.exception("Button listener error")

    # -- LED state monitor (runs in its own thread) --

    def _desired_led(self) -> str:
        """The LED state that reflects what this device is doing right now:
        'error' | 'on' | 'blink' | 'blink_fast' | 'off'. State-driven (not
        press-driven) so every group member shows its own status:
          off        → idle
          blink      → initializing (warming up / waiting for the shared T0)
          on         → recording (solid)
          blink_fast → stopping (from stop until the capture is fully torn down)
          busy       → dataset work in progress, capture refused (1 pulse / pause)
          error      → hardware fault, capture refused (3 pulses / pause)."""
        from grabette.capture_scheduler import get_capture_scheduler
        b = self._backend
        # Highest priority, above teleop and above any capture state: the device
        # is refusing to record, and the LED is the only place an operator holding
        # a screenless grabette can find that out. A fault that is outranked by a
        # busy pattern is a fault nobody sees.
        if getattr(b, "hardware_error", ""):
            return "error"
        if b.is_teleop_active:
            return "on" if b.is_teleop_sending else "off"
        # Stopping wins over is_capturing: the backend keeps is_capturing True
        # through the mux teardown, so check the stopping flag (and the local
        # press ack) first → fast blink until fully off.
        if getattr(b, "is_stopping", False) or time.monotonic() < self._stop_ack_until:
            return "blink_fast"
        # A start in progress keeps blinking CONTINUOUSLY until the recording is
        # fully live. Checked BEFORE is_capturing on purpose: the backend flips
        # is_capturing True partway through start_capture (before the streams are
        # started and while the scheduled-start task is still finishing), so
        # checking it first would flash solid mid-init, then blink again. Since
        # is_scheduled stays True until start_capture fully returns, this holds a
        # steady blink through the whole warm-up → solid only once truly recording.
        if getattr(b, "is_starting", False) or get_capture_scheduler().is_scheduled():
            return "blink"       # initializing: warming up / waiting for T0
        if b.is_capturing:
            return "on"          # recording (solid)
        # Below the capture states on purpose: if a recording IS somehow running,
        # showing it wins. Above idle, because a device busy uploading is not
        # available, and dark is the signal for available.
        if getattr(b, "busy_reason", ""):
            return "busy"
        return "off"             # idle

    def _led_monitor(self) -> None:
        """Drive the LED from _desired_led(), but LATCH the transitions so a slow
        cold start — where the backend flags briefly read all-false (→ off) or
        flip between states, differently on each device — can't make the LED
        flicker. The intent the user wants is dead simple: blink from the click
        until the recording is truly live, then a steady solid. So:
          • a momentary 'off' during a start/recording is debounced away;
          • once solid (recording), a transient 'blink' reading never regresses
            it back to blinking — only a stop/idle leaves 'on'.
        Teleop is exempt (immediate feedback, no latching). Applying only on
        change avoids restarting led_blink()'s phase and needless GPIO writes."""
        btn = self._button
        applied = None          # LED state currently on the hardware
        off_since = None        # monotonic when 'off' was first seen (debounce)
        while not self._stop_event.is_set():
            try:
                raw = self._desired_led()
                if raw == "error":
                    # Never debounced and never latched away: while the fault
                    # stands the LED shows nothing else.
                    want = raw
                    off_since = None
                elif self._backend.is_teleop_active:
                    want = raw  # teleop: snappy, no latching
                    off_since = None
                elif raw == "off":
                    # Swallow a brief all-idle gap mid start/recording; a real
                    # stop reaches 'off' via 'blink_fast', which is applied at once.
                    if applied in ("blink", "on"):
                        off_since = off_since or time.monotonic()
                        want = "off" if time.monotonic() - off_since >= _OFF_DEBOUNCE_S else applied
                    else:
                        want = "off"
                else:
                    off_since = None
                    # Don't drop a live 'solid' back to 'blink' on a transient
                    # scheduled/starting reading — hold solid until stop/idle.
                    want = "on" if (raw == "blink" and applied == "on") else raw
                if want != applied:
                    if want == "error":
                        btn.led_pulses(_ERROR_PULSES, _ERROR_PULSE_ON_S,
                                       _ERROR_PULSE_OFF_S, _ERROR_GAP_S)
                    elif want == "busy":
                        btn.led_pulses(_BUSY_PULSES, _ERROR_PULSE_ON_S,
                                       _ERROR_PULSE_OFF_S, _BUSY_GAP_S)
                    elif want == "on":
                        btn.led_on()
                    elif want == "blink":
                        btn.led_blink(0.3)        # initializing: steady blink
                    elif want == "blink_fast":
                        btn.led_blink(0.1)        # stopping: rapid blink
                    else:
                        btn.led_off()
                    applied = want
            except Exception:
                logger.debug("LED monitor tick failed", exc_info=True)
            self._stop_event.wait(0.2)

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
        # LED follows is_teleop_sending via the monitor.
        new_state = not self._backend.is_teleop_sending
        self._backend.set_teleop_send(new_state)
        logger.info("Button — teleop sending %s", "ON" if new_state else "OFF (reposition)")

    # -- Capture actions (scheduled on the async event loop) --

    def _do_start_capture(self) -> None:
        # LED (blink → solid) is driven by the state monitor; here we just run
        # the start and log the outcome.
        future = asyncio.run_coroutine_threadsafe(self._start_capture_coro(), self._loop)
        try:
            future.result(timeout=RECORDING_WAIT_TIMEOUT_S)
            logger.info("Button capture started")
        except Exception:
            logger.exception("Button start_capture failed")

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
        members = sync.get("members")
        signature = sync.get("signature")
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
            # active task, immediate. With no session running and no task
            # explicitly selected on the local UI, that resolves to Unassigned —
            # a press outside a session records a visibly untriaged solo episode
            # rather than hiding one inside the last task recorded to.
            task_id = sm.active_task_id
            target = None

        # Derive the episode id from the shared T0 (not local creation time)
        # so this device's episode folder matches its peers' exactly.
        episode_id = sm.create_episode(
            task_id,
            episode_id=episode_id_for(target) if target else None,
            members=members,
            signature=signature,
        )
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
        # Blink briefly right here to acknowledge the press on THIS device (the
        # monitor would otherwise just switch off); then it follows state again.
        self._stop_ack_until = time.monotonic() + STOP_ACK_S
        future = asyncio.run_coroutine_threadsafe(self._stop_capture_coro(), self._loop)
        try:
            future.result(timeout=30.0)
        except Exception:
            logger.exception("Button stop_capture failed")

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

        # Tell the fleet to stop the group's peers FIRST, then stop locally.
        # Order is critical: backend.stop_capture() muxes the mp4 synchronously,
        # blocking the event loop for ~1-2s, so anything we send AFTER it only
        # leaves the box once the mux is done — the peers would then keep
        # recording for our whole mux (observed as ~4s longer peer episodes).
        # Notifying first (the fleet is warm during a session → sub-second) lets
        # the peers stop within ~1 poll interval of us, keeping the group's stop
        # spread small. notify_group_stop never raises and returns fast when
        # solo/unreachable, so this doesn't meaningfully delay a standalone stop.
        # A transient failure is retried in the background (see notify_group_stop);
        # by the time a retry runs, our stop_capture below has completed, so a
        # device that is capturing again is necessarily a NEW episode — which the
        # retry must not stop.
        await notify_group_stop(
            should_abort=lambda: self._backend.is_capturing or scheduler.is_scheduled(),
        )
        status = await self._backend.stop_capture()
        sm.register_episode(getattr(status, "episode_id", None))
        logger.info(
            "Button capture stopped: %.1fs, %d frames",
            status.duration_seconds, status.frame_count,
        )
