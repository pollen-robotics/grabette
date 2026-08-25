#!/bin/bash
# TLV320AIC3104 mixer setup for the GRABETTE V2 HAT speaker.
#
# Ported from github.com/apirrone/microduck_runtime (rpi_setup/aic3104-init.sh):
# same codec, same HAT — that one runs on a Pi Zero 2W (and a Radxa), this one
# on a Raspberry Pi 4. Differences from the original:
#   • Playback only. The HAT's onboard microphone routing is dropped — grabette
#     records no audio, the speaker is purely operator feedback.
#   • No DKMS wait: on Pi OS the codec driver (snd_soc_tlv320aic3x) is in-tree
#     and autoloads from the devicetree I2C match, so there is no module build
#     to sit out like on the Radxa.
#   • THE SPEAKER IS OPTIONAL — a grabette built without one is a supported
#     configuration, so this script never fails and never stalls because of it.
#     That matters concretely: it is a Type=oneshot ordered Before=
#     grabette.service, so whatever it waits for is added to the DAEMON'S boot
#     latency, and a non-zero exit would leave a permanently failed unit (and
#     `systemctl is-system-running` = degraded) on a device working exactly as
#     intended. Hence the three-way structure below: configure if the card is
#     up, wait only if the devicetree says a codec should appear, and otherwise
#     return immediately.
#
# The kernel codec driver handles PLL / DAC / line-output configuration from the
# devicetree overlay; this script only applies the mixer levels and the output
# routing on top. Idempotent — safe to re-run at any time to un-mute a device.

set -u

CARD=aic3104

# Property that exists ONLY because our overlay was applied (see
# config/overlays/tlv320aic3104-overlay.dts: simple-audio-card,name = "aic3104").
# Its absence means no codec is configured on this device.
DT_CARD_NAME=/proc/device-tree/sound/simple-audio-card,name

# How long to wait for a codec the devicetree promises, in 0.5s steps. The card
# is created during kernel boot, so a couple of seconds is plenty.
TRIES=10

apply_mixer_levels() {
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
}

card_is_up() {
    amixer -c "$CARD" info >/dev/null 2>&1
}

# 1. Card already registered — the normal case on a device with a speaker.
if card_is_up; then
    apply_mixer_levels
    exit 0
fi

# 2. Not registered. Before waiting, ask the devicetree whether a codec is even
#    supposed to be here: on a speaker-less grabette there is nothing to wait
#    for, and every second spent here delays the daemon's start. Note this only
#    gates the WAIT, never the configuring above — so a mis-detection here can
#    never leave a working speaker muted.
if ! grep -qa "$CARD" "$DT_CARD_NAME" 2>/dev/null; then
    echo "No '$CARD' codec in the devicetree — this grabette has no speaker."
    echo "If it should have one: dtoverlay=tlv320aic3104 in /boot/firmware/config.txt"
    echo "+ /boot/firmware/overlays/tlv320aic3104.dtbo ('make install-audio'), then reboot."
    exit 0
fi

# 3. The devicetree declares the codec but the card hasn't appeared yet — give
#    the probe a moment.
for _ in $(seq 1 $TRIES); do
    sleep 0.5
    if card_is_up; then
        apply_mixer_levels
        exit 0
    fi
done

# Declared but never registered: the overlay is loaded and the chip isn't
# answering (not fitted, or an I2C fault). Worth saying out loud — but still
# exit 0, since the speaker is optional and this unit must not fail the boot.
echo "'$CARD' is declared in the devicetree but never registered as a card." >&2
echo "The codec is not answering on I2C. Check 'i2cdetect -y 1' for 0x18," >&2
echo "and dmesg for tlv320aic3x probe errors. Recording is unaffected." >&2
exit 0
