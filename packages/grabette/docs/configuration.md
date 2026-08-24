# Grabette — Configuration

See the [README](../README.md) for install and usage.

## Robot-frame convention

Finger angles published in `AngleSample.proximal` / `AngleSample.distal` (and in the data this daemon writes) are in **robot frame**, matching the gripette runtime:

- `0 rad` — fingers fully **open**
- positive — **closing**

The two AS5600L magnets rotate in opposite directions when the fingers close, and a right-hand grabette is the mirror of a left-hand one — so the per-sensor sign that bridges raw rotation → robot frame depends on the `hand` setting. Defaults: `right → distal=+1, proximal=-1`; `left → distal=-1, proximal=+1`. Override individual signs via `GRABETTE_DISTAL_SIGN` / `GRABETTE_PROXIMAL_SIGN` only for an asymmetric hardware revision.

## Environment variables

All settings via environment variables with `GRABETTE_` prefix. Persistent per-device config lives in `/etc/grabette/env`, sourced by `grabette.service`.

| Variable | Default | Description |
|---|---|---|
| `GRABETTE_HOST` | `0.0.0.0` | Server bind address |
| `GRABETTE_PORT` | `8000` | Server port |
| `GRABETTE_BACKEND` | `auto` | `auto`, `mock`, or `rpi` |
| `GRABETTE_DATA_DIR` | `~/grabette-data` | Data storage directory |
| `GRABETTE_CAMERA_FPS` | `46` | Camera frame rate |
| `GRABETTE_IMU_HZ` | `200` | IMU sample rate |
| `GRABETTE_ANGLE_SENSORS` | `true` | Enable AS5600 angle sensors |
| `GRABETTE_HAND` | `right` | `left` or `right` — determines default `*_sign`. Written by `make install-rpi HAND=…` |
| `GRABETTE_DISTAL_SIGN` | (from `hand`) | Override the hand-derived distal sensor sign. ±1 |
| `GRABETTE_PROXIMAL_SIGN` | (from `hand`) | Override the hand-derived proximal sensor sign. ±1 |
| `GRABETTE_UI_ENABLED` | `true` | Enable Gradio dashboard |
| `GRABETTE_BUTTON_ENABLED` | `true` | Enable hardware button |
| `GRABETTE_SOUND_ENABLED` | `true` | Cues on the HAT speaker: recording start, recording stop, failed command |
| `GRABETTE_SOUND_DEVICE` | (auto) | ALSA device. Empty = auto-detect the codec by card name (`plughw:CARD=aic3104`) |
| `GRABETTE_SOUND_VOLUME` | `0.6` | Amplitude of the generated cue, `0`..`1` (absolute loudness is the codec mixer's job) |
| `GRABETTE_LOG_LEVEL` | `INFO` | Logging level |

## Audible recording cue

`GRABETTE_SOUND_*` drives the TLV320AIC3104 codec on the V2 HAT. Two cues, both
placed at the real boundaries of the take rather than at the button press:

- **ascending**, from `RpiBackend.start_capture`, at the point where the
  recording is genuinely rolling — OAK-D warmed up, sync clock started, all
  streams recording. On a synchronized group start every device reaches that
  point at the shared T0 and they beep together.
- **descending**, from the top of `RpiBackend.stop_capture`, where the streams
  stop saving frames — i.e. *before* the ~1-2s mux, which it then plays over
  (the cue is a detached subprocess, the mux blocks the event loop).
- **low and repeated**, when a capture command fails: from `start_capture`
  (any trigger), from `CaptureScheduler` (a group start/stop failing around it,
  possibly on a peer nobody is watching), and from `ButtonListener` (failures
  that never reach the backend — a fleet refusal, a start that never fired, a
  refused stop). `sound.cue_error()` is the entry point for callers with no
  backend handle. Overlapping reports of one failure collapse into a single
  buzz via a per-cue debounce (`CUE_DEBOUNCE_S`), so no layer has to know
  whether another already cued it.

The card is always addressed **by name**, never by index: on a Pi 4 the
`vc4-hdmi` cards are registered too, so the codec's number isn't stable. Setting
`GRABETTE_SOUND_DEVICE` overrides the auto-detection with any ALSA device string.

The speaker is **optional hardware**, and sound is cosmetic: a missing codec, a
missing `aplay`, or a playback error logs one line and is otherwise ignored —
`play_start()`/`play_stop()` become no-ops and nothing in the recording path
branches on it. Setup, the speaker-less case, and troubleshooting:
[README → Speaker](../README.md#speaker-audible-recording-cue-make-install-audio).
