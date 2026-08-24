"""Speaker bring-up test — plays the exact cue the daemon plays when a
recording goes live, through the same code path.

Run on the Pi 4:
    python scripts/test_speaker.py

If it stays silent, in order:
    aplay -l | grep aic3104            # card registered? (else: overlay/config.txt + reboot)
    sudo /usr/local/bin/aic3104-init.sh   # mixer un-muted? (installed by make install-audio)
    amixer -c aic3104 contents | less  # inspect 'Line Playback Switch' / volumes
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grabette.hardware.sound import CARD_NAME, autodetect_device, get_speaker  # noqa: E402


def main() -> int:
    device = autodetect_device()
    if device is None:
        print(f"[FAIL] no ALSA card '{CARD_NAME}' — the codec overlay isn't live.")
        print("       Check: dtoverlay=tlv320aic3104 in /boot/firmware/config.txt,")
        print("              /boot/firmware/overlays/tlv320aic3104.dtbo exists,")
        print("              then reboot.  (make install-audio)")
        return 1
    print(f"[OK]   card '{CARD_NAME}' found → {device}")

    speaker = get_speaker()
    speaker.prepare()
    if not speaker.is_available:
        print("[FAIL] speaker unavailable — see the log lines above "
              "(GRABETTE_SOUND_ENABLED=false? aplay missing?)")
        return 1

    print("[..]   playing the capture-start cue")
    speaker.play_start()
    time.sleep(1.5)  # let the detached aplay finish before we tear the temp wav down
    speaker.close()
    print("[OK]   done — if you heard nothing, the mixer is probably still muted:")
    print("       sudo /usr/local/bin/aic3104-init.sh")
    return 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
