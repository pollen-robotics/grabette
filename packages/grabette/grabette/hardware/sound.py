"""Audible feedback through the TLV320AIC3104 codec on the V2 HAT.

Hardware: the codec hangs off the Pi's I2S bus (control over i2c-1 @ 0x18,
12 MHz fixed MCLK) and drives a small speaker from its line outputs. See
`config/overlays/tlv320aic3104-overlay.dts` + `scripts/aic3104-init.sh` for the
devicetree/mixer side, and `make install-audio`.

Used for one thing today: a beep at the instant a recording actually goes live
(after the OAK-D warm-up), so an operator holding the grabette knows the take
has started without looking at the LED — and so a whole group of grabettes
audibly starts in unison at the shared T0.

Playback rules that matter here:
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
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# ALSA card id created by the overlay (simple-audio-card,name = "aic3104").
CARD_NAME = "aic3104"

# The codec's only natively clocked rate — see the module docstring.
SAMPLE_RATE = 48000

# The "recording is live" cue: two short ascending tones. Kept brief so it
# can't bleed into more than the first few frames of the take.
# (frequency Hz, duration s)
START_TONES: tuple[tuple[float, float], ...] = ((880.0, 0.09), (1320.0, 0.11))

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
        self._start_wav: Path | None = None
        self._aplay = shutil.which("aplay")
        self._procs: list[subprocess.Popen] = []
        self._lock = threading.Lock()

    def prepare(self) -> None:
        """Resolve the device and pre-render the cue. Called once at backend
        start so the play path is only a fork/exec. Never raises."""
        if not self._enabled:
            logger.info("Speaker disabled by config")
            return
        if self._aplay is None:
            logger.info("Speaker unavailable: aplay not found (apt install alsa-utils)")
            self._enabled = False
            return
        if not self._device:
            device = autodetect_device()
            if device is None:
                logger.info(
                    "Speaker unavailable: no ALSA card '%s' "
                    "(codec overlay not loaded? see 'make install-audio')", CARD_NAME,
                )
                self._enabled = False
                return
            self._device = device
        try:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="grabette-sound-")
            self._start_wav = Path(self._tmpdir.name) / "capture_start.wav"
            _render_wav(self._start_wav, START_TONES, self._volume)
        except Exception:
            logger.warning("Speaker cue rendering failed; sound disabled", exc_info=True)
            self._enabled = False
            return
        logger.info("Speaker ready on '%s'", self._device)

    @property
    def is_available(self) -> bool:
        return self._enabled and self._start_wav is not None

    def play_start(self) -> None:
        """Beep 'recording is live'. Returns immediately; never raises."""
        self._play(self._start_wav)

    def _play(self, wav: Path | None) -> None:
        if not self._enabled or wav is None:
            return
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
        self._start_wav = None


_speaker: Speaker | None = None


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
