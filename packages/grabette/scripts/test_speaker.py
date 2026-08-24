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
    Speaker,
    autodetect_device,
)


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
    speaker.close()
    if not played:
        print("[FAIL] aplay reported an error (above) — the card is there but "
              "playback failed.")
        print("       Common cause: the invoking user is not in the 'audio' "
              "group (check with: groups).")
        return 1
    print("[OK]   aplay played every cue without error. If you heard nothing, the")
    print("       mixer is almost certainly still muted — the codec boots with its")
    print("       line outputs off:  sudo /usr/local/bin/aic3104-init.sh")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
