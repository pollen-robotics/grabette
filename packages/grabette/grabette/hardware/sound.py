"""Audible feedback through the TLV320AIC3104 codec on the V2 HAT.

Hardware: the codec hangs off the Pi's I2S bus (control over i2c-1 @ 0x18,
12 MHz fixed MCLK) and drives a small speaker from its line outputs. See
`config/overlays/tlv320aic3104-overlay.dts` + `scripts/aic3104-init.sh` for the
devicetree/mixer side, and `make install-audio`.

Used to mark both ends of a take, so an operator holding the grabette knows
where the recording actually starts and stops without looking at the LED:
  • an ASCENDING beep at the instant the recording goes live (after the OAK-D
    warm-up) — a whole group of grabettes hits that instant at the shared T0,
    so the rig audibly starts in unison;
  • a DESCENDING beep the moment the streams stop saving frames — which is the
    START of the ~1-2s mux, not its end, so it is heard while the daemon is
    still busy writing the episode;
  • a SHORT, HIGH blip once the episode is fully written to disk — the mp4 mux
    and the JSON sidecars are done, so the device can be moved or powered off.
    That lands ~1-2s after the stop cue, which fires at the START of the mux;
  • a REPEATED, three-tone buzz when a capture command fails — the case that
    actually needs sound, since the failure modes are exactly the ones with no
    screen in front of them: you press the button, the LED goes back to idle,
    and nothing tells you whether the take started.

Playback rules that matter here:
  • The speaker is OPTIONAL hardware. A grabette built without one is a
    supported configuration, not a broken install: no codec (or no alsa-utils,
    or GRABETTE_SOUND_ENABLED=false) means prepare() logs one line, disables
    itself, and every play_*() becomes a no-op. Nothing downstream branches on
    it — the backend calls play_*() unconditionally.
  • It must NEVER delay or break a capture. Every failure path is swallowed and
    logged; the actual play happens on a throwaway thread so the caller (the
    start_capture hot path) is not even charged the fork/exec of `aplay`.
  • 48 kHz stereo S16_LE. The overlay clocks the codec from a FIXED 12 MHz
    oscillator with mclk-fs = 250, so 48 kHz is the only natively supported
    rate; we render at exactly that and go through `plughw` for safety.
  • The card is addressed BY NAME (`plughw:CARD=aic3104`), never by index: on a
    Pi 4 the vc4-hdmi cards are also registered, so the codec's card number is
    not stable across boots/HDMI state.
"""

from __future__ import annotations

import array
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# ALSA card id created by the overlay (simple-audio-card,name = "aic3104").
CARD_NAME = "aic3104"

# The codec's only natively clocked rate — see the module docstring.
SAMPLE_RATE = 48000

# The cues, as (frequency Hz, duration s) sequences. Kept brief so they can't
# bleed into more than the first/last few frames of a take. Start ASCENDS and
# stop DESCENDS — the pair has to be told apart by ear alone, and rising vs
# falling is unmistakable in a way that two different pitches are not.
START_TONES: tuple[tuple[float, float], ...] = ((880.0, 0.09), (1320.0, 0.11))
STOP_TONES: tuple[tuple[float, float], ...] = ((1320.0, 0.09), (880.0, 0.11))
# Error: three buzzes at one pitch. It has to be recognisable as "something went
# wrong" by someone looking at the workspace, not at the device — and what makes
# it so is the SHAPE, not the pitch: a repeated triplet with silent gaps, ~0.66s
# against the start pair's 0.20s, where every other cue is a two-tone glide or a
# single blip.
#
# It was designed as the LOWEST of the four (220 Hz, an octave and a half under
# the others) and that turned out to be untenable on the hardware: the HAT's
# speaker is a small unenclosed driver, it rolls off steeply below ~1 kHz, and
# the cue was flatly inaudible on-device — the worst possible failure for the one
# cue whose whole job is to report faults nobody is watching for. Measured on a
# V2 HAT at the mixer levels of scripts/aic3104-init.sh: 220 Hz silent, 440 Hz
# barely there, 660 Hz audible but weak even at full scale, 1100 Hz clean. So it
# now sits INSIDE the speaker's usable band, on a frequency no other cue uses
# (880/1320 are start+stop, 1760 is saved), and leans on rhythm and length alone
# to stand apart.
#
# Do not take it back down towards the bottom end without re-testing on a real
# speaker at the shipped mixer levels — the constraint here is the driver's
# response, not taste. See also CUE_GAINS: it is rendered hotter than the rest.
# A 0 Hz "tone" renders as silence (sin(0) = 0), which is how the gaps are made.
ERROR_TONES: tuple[tuple[float, float], ...] = (
    (1100.0, 0.15), (0.0, 0.07), (1100.0, 0.15), (0.0, 0.07), (1100.0, 0.22),
)
# Saved: one short high blip. Deliberately the slightest of the four — it is a
# confirmation arriving a second or two after the stop cue, on every single
# take, so it has to register without competing with the cues that mark the
# recording itself. Single and very short is what sets it apart from the pair
# and the triplet above.
SAVED_TONES: tuple[tuple[float, float], ...] = ((1760.0, 0.06),)

CUE_START = "capture_start"
CUE_STOP = "capture_stop"
CUE_SAVED = "capture_saved"
CUE_ERROR = "capture_error"
CUES: dict[str, tuple[tuple[float, float], ...]] = {
    CUE_START: START_TONES,
    CUE_STOP: STOP_TONES,
    CUE_SAVED: SAVED_TONES,
    CUE_ERROR: ERROR_TONES,
}

# Per-cue multiplier on the configured volume (GRABETTE_SOUND_VOLUME). The error
# buzz is not a routine confirmation like the other three — it is the one cue
# that must never be missed — so it renders at the full digital scale
# (_render_wav clamps there) while the rest keep the shared trim, which exists
# to stop the routine cues being obnoxious. A multiplier rather than a fixed
# amplitude, so lowering GRABETTE_SOUND_VOLUME keeps the buzz proportionally
# louder instead of quietly flattening the difference. It buys ~4 dB: real, but
# a safety margin on top of a cue placed where the speaker can reproduce it
# (see ERROR_TONES) — never a substitute for that.
CUE_GAINS: dict[str, float] = {CUE_ERROR: 1.7}

# A failure is typically noticed by several layers at once (the backend raises,
# the scheduler logs it and discards the episode, the button listener reports it
# to the operator). Every one of them is a legitimate place to ask for the error
# cue, and none of them can know whether another already did — so instead of
# coordinating ownership, the same cue simply won't replay within this window.
CUE_DEBOUNCE_S = 1.5

# Upper bound on one cue playback before we assume aplay is wedged (the cue
# itself is a fraction of a second).
PLAY_TIMEOUT_S = 5.0

# Linear fade applied to each tone's edges. Without it the abrupt start/stop of
# the waveform is a step on the DAC output and the speaker clicks audibly.
_FADE_S = 0.006


def _render_wav(path: Path, tones, volume: float, rate: int = SAMPLE_RATE) -> None:
    """Render `tones` to a 16-bit stereo WAV at `rate`, in place of shipping a
    binary asset (keeps the cue tweakable from config)."""
    amplitude = max(0.0, min(1.0, volume)) * 32767.0
    samples = array.array("h")
    for freq, duration in tones:
        n = int(rate * duration)
        fade = max(1, int(rate * _FADE_S))
        for i in range(n):
            # Triangular fade in/out, clamped to 1.0 in the middle.
            env = min(1.0, i / fade, (n - i) / fade)
            v = int(amplitude * env * math.sin(2.0 * math.pi * freq * i / rate))
            samples.append(v)  # left
            samples.append(v)  # right
    if sys.byteorder == "big":
        samples.byteswap()  # WAV frames are little-endian
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def autodetect_device() -> str | None:
    """ALSA device for the HAT codec, or None if the card isn't registered.

    /proc/asound/<id> is the per-card symlink ALSA creates from the card id, so
    its presence is both the existence check and the name-based address.
    """
    if Path(f"/proc/asound/{CARD_NAME}").exists():
        return f"plughw:CARD={CARD_NAME},DEV=0"
    return None


class Speaker:
    """Fire-and-forget cue player. Safe to construct with no audio hardware."""

    def __init__(
        self, device: str = "", volume: float = 0.6, enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._device = device or ""
        self._volume = volume
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._cues: dict[str, Path] = {}
        self._aplay = shutil.which("aplay")
        self._procs: list[subprocess.Popen] = []
        self._lock = threading.Lock()
        # cue name -> monotonic time it last played (see CUE_DEBOUNCE_S).
        self._last_played: dict[str, float] = {}

    def prepare(self) -> None:
        """Resolve the device and pre-render the cue. Called once at backend
        start so the play path is only a fork/exec. Never raises."""
        if not self._enabled:
            logger.info("Speaker cues disabled by config (GRABETTE_SOUND_ENABLED)")
            return
        if self._aplay is None:
            logger.info(
                "No aplay binary — running without the speaker cues "
                "(apt install alsa-utils)",
            )
            self._enabled = False
            return
        if not self._device:
            device = autodetect_device()
            if device is None:
                # Not a warning: this is the normal state of a grabette with no
                # speaker fitted, and it must not read as a broken install.
                logger.info(
                    "No '%s' sound card — running without the speaker cues. "
                    "Expected if this grabette has no speaker; if it has one, "
                    "see 'make install-audio'.", CARD_NAME,
                )
                self._enabled = False
                return
            self._device = device
        try:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="grabette-sound-")
            for name, tones in CUES.items():
                path = Path(self._tmpdir.name) / f"{name}.wav"
                gain = CUE_GAINS.get(name, 1.0)
                _render_wav(path, tones, self._volume * gain)
                self._cues[name] = path
        except Exception:
            logger.warning("Speaker cue rendering failed; sound disabled", exc_info=True)
            self._cues = {}
            self._enabled = False
            return
        logger.info("Speaker ready on '%s'", self._device)

    @property
    def is_available(self) -> bool:
        return self._enabled and bool(self._cues)

    def play_start(self) -> None:
        """Beep 'recording is live' (ascending). Returns at once; never raises."""
        self._play(CUE_START)

    def play_stop(self) -> None:
        """Beep 'recording over' (descending). Returns at once; never raises."""
        self._play(CUE_STOP)

    def play_saved(self) -> None:
        """Blip 'episode written to disk' (short, high). Returns at once; never
        raises."""
        self._play(CUE_SAVED)

    def play_error(self) -> None:
        """Buzz 'that command failed' (repeated triplet). Safe to call from every
        layer that notices the same failure — see CUE_DEBOUNCE_S. Returns at
        once; never raises."""
        self._play(CUE_ERROR)

    def _play(self, name: str) -> None:
        wav = self._cues.get(name) if self._enabled else None
        if wav is None:
            return
        now = time.monotonic()
        with self._lock:
            if now - self._last_played.get(name, 0.0) < CUE_DEBOUNCE_S:
                return
            self._last_played[name] = now
        # Off-thread so the caller (start_capture, right after the streams go
        # live) never pays the fork/exec — cheap, and a stuck aplay can't stall
        # a recording.
        threading.Thread(
            target=self._spawn, args=(wav,), daemon=True, name="speaker-cue",
        ).start()

    def _spawn(self, wav: Path) -> bool:
        """Run aplay to completion on this (throwaway) thread, and REPORT what
        it said. Discarding aplay's stderr makes a silent speaker impossible to
        diagnose from the journal — a muted mixer, a busy card and a missing
        /dev/snd permission all look identical (nothing at all). So: capture
        stderr, check the exit status, log a warning on failure. Still never
        raises, still off the caller's thread. Returns True if aplay exited 0
        (which means it PLAYED, not that anything was audible — a muted mixer
        exits 0 too)."""
        try:
            proc = subprocess.Popen(
                [self._aplay, "-q", "-D", self._device, str(wav)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except Exception:
            logger.warning("Speaker: cannot run %s", self._aplay, exc_info=True)
            return False
        with self._lock:
            # Keep a handle so close() can cut a hung player short, and prune
            # finished ones so repeated cues don't pile up.
            self._procs = [p for p in self._procs if p.poll() is None]
            self._procs.append(proc)
        try:
            _, err = proc.communicate(timeout=PLAY_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.warning(
                "Speaker: aplay still running after %.0fs on %s, killed",
                PLAY_TIMEOUT_S, self._device,
            )
            return False
        except Exception:
            logger.debug("Speaker: aplay wait failed", exc_info=True)
            return False
        if proc.returncode != 0:
            logger.warning(
                "Speaker: aplay failed on %s (exit %s): %s",
                self._device, proc.returncode,
                (err or b"").decode("utf-8", "replace").strip() or "(no message)",
            )
            return False
        return True

    def close(self) -> None:
        with self._lock:
            procs, self._procs = self._procs, []
        for p in procs:
            if p.poll() is None:
                p.terminate()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        self._cues = {}


_speaker: Speaker | None = None


def cue_error() -> None:
    """Buzz that a capture command failed, callable from anywhere in the daemon.

    For the failures that never reach RpiBackend.start_capture (a fleet refusal,
    a scheduled start that doesn't fire, a stop that can't proceed) and that the
    operator would otherwise learn about only from the journal. Reaches the
    process-wide speaker directly, so a caller doesn't need a backend handle;
    no-op when no speaker was ever set up (mock backend, no codec fitted), and
    never raises. Overlapping callers are handled by the cue debounce, not by
    coordinating who owns the beep."""
    try:
        get_speaker().play_error()
    except Exception:
        logger.debug("error cue failed", exc_info=True)


def get_speaker() -> Speaker:
    """Process-wide speaker, configured from settings. Not prepared yet —
    the owner (RpiBackend.start) calls prepare()."""
    global _speaker
    if _speaker is None:
        from grabette.config import settings
        _speaker = Speaker(
            device=settings.sound_device,
            volume=settings.sound_volume,
            enabled=settings.sound_enabled,
        )
    return _speaker
