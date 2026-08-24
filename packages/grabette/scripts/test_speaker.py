"""Speaker bring-up test — plays each of the daemon's cues (start, stop, error)
in turn, through the same code path it uses.

Run on the Pi 4, with the SYSTEM python — no venv needed:
    python3 scripts/test_speaker.py

(grabette.hardware.sound is deliberately stdlib-only, so this diagnostic works
before/independently of `make install-rpi`. It reads the GRABETTE_SOUND_* env
vars directly rather than going through grabette.config, which would drag in
pydantic and only run inside the venv.)

If it stays silent, in order:
    aplay -l | grep aic3104               # card registered? (else: overlay/config.txt + reboot)
    sudo /usr/local/bin/aic3104-init.sh    # mixer un-muted? (installed by make install-audio)
    amixer -c aic3104 contents | less     # inspect 'Line Playback Switch' / volumes
    groups                                # the daemon's user needs the 'audio' group
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grabette.hardware.sound import (  # noqa: E402
    CARD_NAME,
    CUE_ERROR,
    CUE_SAVED,
    CUE_START,
    CUE_STOP,
    CUES,
    MIN_CUE_S,
    PLAY_TIMEOUT_S,
    Speaker,
    autodetect_device,
)


def _run(label, *cmd) -> None:
    """Best-effort diagnostic command: show its output, never blow up."""
    print(f"       $ {' '.join(cmd)}   ({label})")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        out = (r.stdout + r.stderr).strip() or "(no output)"
    except FileNotFoundError:
        out = f"(not installed — apt install {cmd[0]})"
    except Exception as e:
        out = f"({e})"
    for line in out.splitlines()[:12]:
        print(f"         {line}")


_PCM_STATUS = Path(f"/proc/asound/{CARD_NAME}/pcm0p/sub0/status")


def _pcm_field(status: str, key: str) -> str:
    for line in status.splitlines():
        if line.strip().startswith(key):
            return line.split(":", 1)[-1].strip()
    return ""


def _probe_dma(wav: Path, device: str) -> None:
    """THE discriminating test. Start a playback and watch the kernel's own PCM
    pointer while it runs. aplay blocking tells us nothing on its own; hw_ptr
    says which half of the stack is stuck:

      state RUNNING + hw_ptr advancing → the DMA runs and audio is flowing; a
          hang would then be impossible, so look at what aplay is waiting on
      state RUNNING + hw_ptr FROZEN    → the transfer started and stalled. The
          Pi is bitclock master here, so its I2S clock is the suspect, not the
          codec
      state PREPARED                   → the transfer never started at all —
          a trigger/start-threshold problem, not a clock one
      "closed" / missing               → aplay never opened the device
    """
    print()
    print("       --- is the DMA advancing? ---")
    if not _PCM_STATUS.exists():
        print(f"       {_PCM_STATUS} is missing — no playback substream on this card.")
        return
    aplay = shutil.which("aplay") or "aplay"
    proc = subprocess.Popen(
        [aplay, "-q", "-D", device, str(wav)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        seen = []
        for i in range(3):
            time.sleep(0.7)
            status = _PCM_STATUS.read_text()
            state = _pcm_field(status, "state") or status.strip().splitlines()[0]
            hw = _pcm_field(status, "hw_ptr")
            seen.append((state, hw))
            print(f"       t+{0.7 * (i + 1):.1f}s  state={state or '?':10s} hw_ptr={hw or '-'}")
    finally:
        proc.kill()
        try:
            proc.communicate(timeout=2)
        except Exception:
            pass
    ptrs = {hw for _, hw in seen if hw}
    states = {st for st, _ in seen}
    print()
    if not ptrs:
        print("       VERDICT: the substream never opened (state above). aplay is")
        print("                not reaching the hardware at all — check that no")
        print("                other process holds the card: sudo fuser -v /dev/snd/*")
    elif len(ptrs) == 1 and states <= {"RUNNING", "XRUN"}:
        print("       VERDICT: the transfer STARTED and is FROZEN — hw_ptr never")
        print("                moved. The Pi drives the bit clock on this link, so")
        print("                suspect the I2S clock / DMA, not the codec. Try a")
        print("                full power cycle (rail actually down), and check")
        print("                dmesg for bcm2835-i2s / dma errors.")
    elif "PREPARED" in states or "SETUP" in states:
        print("       VERDICT: the stream is set up but NEVER TRIGGERED. That's a")
        print("                start-threshold / trigger problem rather than a")
        print("                clock one — report this output, it changes the fix.")
    else:
        print("       VERDICT: hw_ptr IS advancing — the DMA runs and audio is")
        print("                flowing. Then the hang is not in the transfer;")
        print("                report this output.")


def _diagnose_wedged_codec(wav: Path, device: str) -> None:
    """A hang means the PCM opened but never progressed. Walk the stack from the
    kernel's own view of the transfer outwards, rather than guessing."""
    _probe_dma(wav, device)
    print()
    print("       --- codec control path ---")
    _run("0x18: 'UU' = held by the driver (good), '--' = gone from the bus",
         "i2cdetect", "-y", "1")
    _run("codec / I2S / DMA errors", "sh", "-c",
         "dmesg 2>/dev/null | grep -iE 'tlv320|aic3104|i2c|i2s|dma' | tail -15")
    _run("who holds the card", "sh", "-c", "fuser -v /dev/snd/* 2>&1 | head -8")
    print()
    print("       Worth trying, in order:")
    print("         sudo systemctl stop grabette       # rule out our own daemon")
    print("         sudo /usr/local/bin/aic3104-init.sh --reset")
    print("             ^ alsa-restore replays /var/lib/alsa/asound.state at every")
    print("               boot, so a bad mixer state saved during a misbehaving")
    print("               session comes back after a reboot. --reset clears it.")
    print("         power the Pi down and unplug it for ~30s (a reboot leaves the")
    print("             3.3V rail up, so it clears nothing on the HAT)")


def main() -> int:
    device = os.environ.get("GRABETTE_SOUND_DEVICE") or autodetect_device()
    if device is None:
        print(f"[FAIL] no ALSA card '{CARD_NAME}' — the codec overlay isn't live.")
        print("       Check: dtoverlay=tlv320aic3104 in /boot/firmware/config.txt,")
        print("              /boot/firmware/overlays/tlv320aic3104.dtbo exists,")
        print("              then reboot.  (make install-audio)")
        print("       Cards ALSA currently sees:")
        listing = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True,
        )
        for line in (listing.stdout + listing.stderr).splitlines() or ["(none)"]:
            print(f"         {line}")
        return 1
    print(f"[OK]   using device {device}")

    speaker = Speaker(
        device=device,
        volume=float(os.environ.get("GRABETTE_SOUND_VOLUME", "0.6")),
    )
    speaker.prepare()
    if not speaker.is_available:
        print("[FAIL] speaker unavailable — see the log line above "
              "(aplay missing? apt install alsa-utils)")
        return 1

    # _spawn (rather than the play_* methods) so we run aplay on THIS thread —
    # any error it reports lands on screen instead of in a background thread,
    # and the cue debounce doesn't apply.
    played = True
    elapsed = 0.0
    for cue, label in (
        (CUE_START, "START  (ascending — recording is live)"),
        (CUE_STOP, "STOP   (descending — frames no longer saved)"),
        (CUE_SAVED, "SAVED  (short blip — episode written, mux done)"),
        (CUE_ERROR, "ERROR  (low, repeated — a capture command failed)"),
    ):
        if not played:
            break
        print(f"[..]   playing {label}")
        # Time each one: a cue that takes far longer than its own length is the
        # signature of a clip too short for the device buffer, where aplay
        # blocks in drain instead of returning (see MIN_CUE_S).
        t0 = time.monotonic()
        played = speaker._spawn(speaker._cues[cue])
        elapsed = time.monotonic() - t0
        tone_ms = sum(d for _, d in CUES[cue]) * 1000
        print(f"       aplay returned in {elapsed * 1000:.0f} ms "
              f"(tones {tone_ms:.0f} ms, padded to {MIN_CUE_S * 1000:.0f} ms)")
        time.sleep(0.2)

    try:
        if played:
            print("[OK]   aplay played every cue without error. If you heard nothing,")
            print("       the mixer is almost certainly still muted — the codec boots")
            print("       with its line outputs off:  sudo /usr/local/bin/aic3104-init.sh")
            return 0
        # Which failure it was matters, and elapsed time separates them: a
        # permission or busy-device refusal exits IMMEDIATELY, while a device
        # that opens and then stalls burns the whole watchdog.
        if elapsed >= PLAY_TIMEOUT_S:
            print(f"[FAIL] aplay opened the card but blocked for {elapsed:.0f}s.")
            print("       This is NOT permissions and NOT the card being missing:")
            print("       it is registered and the device opened. What follows asks")
            print("       the kernel where the transfer actually stopped.")
            # Before close(), which deletes the rendered cues.
            _diagnose_wedged_codec(speaker._cues[CUE_START], device)
        else:
            print("[FAIL] aplay refused the device (see its message above).")
            print("       Common causes: the invoking user is not in the 'audio'")
            print("       group (check with: groups), or another process holds the")
            print("       card (systemctl stop grabette, then retry).")
        return 1
    finally:
        speaker.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
