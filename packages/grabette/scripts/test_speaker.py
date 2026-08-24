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
    CUE_START,
    CUE_STOP,
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
        (CUE_ERROR, "ERROR  (low, repeated — a capture command failed)"),
    ):
        if not played:
            break
        print(f"[..]   playing {label}")
        played = speaker._spawn(speaker._cues[cue])
        time.sleep(0.5)
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
