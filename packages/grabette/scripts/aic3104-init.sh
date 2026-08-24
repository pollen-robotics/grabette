#!/bin/bash
# TLV320AIC3104 mixer setup for the GRABETTE V2 HAT speaker.
#
# Ported from github.com/apirrone/microduck_runtime (rpi_setup/aic3104-init.sh):
# same codec, same HAT — that one runs on a Pi Zero 2W (and a Radxa), this one
# on a Raspberry Pi 4. Differences from the original:
#   • Playback only. The HAT's onboard microphone routing is dropped — grabette
#     records no audio, the speaker is purely operator feedback.
#   • Shorter card wait: on Pi OS the codec driver (snd_soc_tlv320aic3x) is
#     in-tree and autoloads from the devicetree I2C match, so there is no DKMS
#     build to wait out like on the Radxa. We still retry, because this unit can
#     be ordered ahead of the card actually registering.
#   • Explicit modprobe fallback for the same reason.
#
# The kernel codec driver handles PLL / DAC / line-output configuration from the
# devicetree overlay; this script only applies the mixer levels and the output
# routing on top. Idempotent — safe to re-run at any time to un-mute a device.

set -u

CARD=aic3104

# Best-effort: normally the I2C devicetree match autoloads this.
modprobe snd-soc-tlv320aic3x-i2c 2>/dev/null || \
  modprobe snd-soc-tlv320aic3x 2>/dev/null || true

# The ALSA card may not be registered the instant this unit runs; retry briefly.
for _ in $(seq 1 15); do
    if amixer -c "$CARD" info >/dev/null 2>&1; then
        # Speaker path: the speaker amp hangs off the codec's line outputs
        # (LEFT_LOP/RIGHT_LOP).
        #   'PCM Playback Volume'      digital DAC gain
        #   'Line DAC Playback Volume' DAC -> LOP routing level
        #   'Line Playback Switch'     the LOP output-stage MUTE. It is NOT a
        #                              line-in bypass: switching it off silences
        #                              the speaker entirely.
        #   'Line Playback Volume'     LOP output-stage gain, 0..9 dB
        amixer -c "$CARD" cset name='PCM Playback Volume'      127,127 >/dev/null 2>&1
        amixer -c "$CARD" cset name='Line DAC Playback Volume' 118,118 >/dev/null 2>&1
        amixer -c "$CARD" cset name='Line Playback Switch'     on,on   >/dev/null 2>&1
        amixer -c "$CARD" cset name='Line Playback Volume'     9,9     >/dev/null 2>&1

        echo "TLV320AIC3104 mixer levels set on card '$CARD'"
        exit 0
    fi
    sleep 1
done

echo "TLV320AIC3104: ALSA card '$CARD' never appeared." >&2
echo "Check: dtoverlay=tlv320aic3104 in /boot/firmware/config.txt," >&2
echo "       /boot/firmware/overlays/tlv320aic3104.dtbo exists (make install-audio)," >&2
echo "       i2cdetect -y 1 shows a device at 0x18." >&2
exit 1
