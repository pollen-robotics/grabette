# Grabette Teleoperation

How to drive a Gripette-equipped robot live with a Grabette V2 (Pi 4 + OAK-D SR).

This document covers the **implemented** teleoperation system.

## Overview

A Grabette acts as a 6-DoF motion-capture device. The user holds the
Grabette and moves their hand; the Pi runs visual-inertial SLAM
(BasaltVIO on host) to track the camera pose; per-step camera-local
deltas plus the current angle-sensor readings are streamed to a
"bridge" process which forwards them to the robot's arm and gripper
gRPC servers.

```
┌─────────────────────────────────────────┐    WebSocket (30 Hz)
│ Grabette V2 (Pi 4 + OAK-D SR)           │  /api/teleop/stream
│  ┌─────────────┐  ┌──────────────────┐  │  {dx,dy,dz, dqx,dqy,dqz,dqw,
│  │ Stereo + IMU│→ │ BasaltVIO        │  │     send, lost, t}
│  └─────────────┘  │ (host-side CPU)  │  │ ──────────────►
│                   └────────┬─────────┘  │
│                            ↓            │  HTTP (30 Hz polling)
│                   ┌──────────────────┐  │  /api/state.angle
│  ┌─────────────┐  │ Camera-local     │  │ ──────────────►
│  │ Angle ×2    │  │ delta + angle    │  │
│  └─────────────┘→ │ broadcast        │  │
│                   └──────────────────┘  │
└─────────────────────────────────────────┘
                                                ▼
                           ┌───────────────────────────────────┐
                           │ teleop_bridge.py (workstation)    │
                           │  • WS consumer + dedup            │
                           │  • IMU motion gate + deadband     │
                           │  • Pos/rot scale + safety cap     │
                           │  • Send-toggle gate (hw button)   │
                           └────────────┬──────────────────────┘
                                        │
                       gRPC (CartesianDelta + MotorCommand)
                                        ▼
                  ┌─────────────────────────────────────┐
                  │ Robot (sim or real grpc server)     │
                  │  ArmService    @ :50052             │
                  │  GripperService @ :50051            │
                  └─────────────────────────────────────┘
```

Camera-local delta convention follows LeRobot §10.3 (the same one used
in the offline dataset format), so policies trained on grabette data
share representation with what the bridge sends live.

## Quickstart

### Prerequisites

- Grabette V2 powered on, `grabette.service` running on the Pi.
- A robot or sim exposing the `openarm` gRPC interfaces — see
  [openarm_gripette_simu](../../openarm_gripette_simu/) for the proto.
- Network reachability: the workstation running the bridge must be able
  to reach both the Pi and the robot's gRPC endpoints.
- Bridge dependencies installed: `uv sync --extra teleop_bridge` from
  the grabette repo on the workstation.

### Operator workflow

```
1. Open the Grabette UI in a browser:   http://rgrabette2:8000/
2. Click "Enter Teleop Mode"            (pauses live-view timers,
                                         starts BasaltVIO on the Pi)
3. Start the bridge:                    uv run python scripts/teleop_bridge.py [args]
4. Hold the Grabette in your hand, oriented as it will be relative to
   the robot's end-effector. This is the initial frame — all motion
   you make is relative to this starting pose.
5. Press the hardware button: robot arm & gripper start mirroring your
   motion (button pressed = send ON).
6. Release the button to "reposition" without moving the robot. SLAM
   keeps tracking; the integrator on the robot just stops advancing.
7. Press again to resume.
8. Click "Exit Teleop Mode" when done.
```

The button is a toggle, not a momentary — press once for ON, again for
OFF. Status feedback is visible in the UI's teleop status line.

### Example commands

**Simulation** (sim arm + sim gripper, both on localhost):

```bash
uv run python scripts/teleop_bridge.py
```

**Real robot** (real gripper on a separate host, conservative scale):

```bash
uv run python scripts/teleop_bridge.py \
    --gripper 192.168.1.36:50051 \
    --max-delta-m 0.02 \
    --pos-scale 0.3 --rot-scale 0.3
```

**Dry-run on the real robot** (validate plumbing, no motion):

```bash
uv run python scripts/teleop_bridge.py \
    --gripper 192.168.1.36:50051 \
    --dry-run --debug
```

## Server-side components (the Pi)

| Path | Role |
|---|---|
| `grabette/hardware/oakd_teleop.py` | OAK pipeline build + BasaltVIO + drain thread that produces camera-local deltas. Camera resolution is 320×200 @ 30 fps (lowered from 640×400 to keep Basalt real-time on Pi). |
| `grabette/backend/rpi.py` | `start_teleop()`/`stop_teleop()`, exposes `is_teleop_active`, `is_teleop_sending`, `get_teleop_delta()`. Mutex with the recording pipeline — the OAK device cannot be in both modes at once. |
| `grabette/app/routers/teleop.py` | REST + WS endpoints: `/api/teleop/{start,stop,status,send,stream}`. |
| `grabette/button_listener.py` | Single-press dispatch by mode: teleop active → toggle send; capturing → stop; idle → start capture. |
| `grabette/ui/app.py` | Web UI toggle + status feedback; pauses live-view timers and chart iframes during teleop. |

### WS payload schema

`/api/teleop/stream` sends one JSON message per WS tick (30 Hz):

```json
{
  "t":   12.345,        // seconds since teleop start (server monotonic)
  "send": true,         // hardware-button state
  "lost": false,        // true if SLAM has no current pose
  "dx":  0.001,         // m   (camera-local position delta, LeRobot §10.3)
  "dy":  0.000,
  "dz":  0.000,
  "dqx": 0.0,           // unit quaternion XYZW of the rotation delta
  "dqy": 0.0,
  "dqz": 0.0,
  "dqw": 1.0
}
```

The WS rate is 30 Hz but Basalt's pose rate is ~22-30 Hz, so successive
messages can carry the same `t` (i.e., the same delta). The bridge
dedups on `t`.

## Bridge: `scripts/teleop_bridge.py`

Single-file async script. Three concurrent tasks share a `SharedState`:

- `arm_loop` — WS consumer. Per message: dedup → send-toggle gate →
  IMU-static gate → deadband → scale → safety cap → `SendCartesianDelta`.
  Also writes `shared.send_enabled` so the gripper task can mirror the
  hardware button.
- `state_loop` — 30 Hz HTTP poll of `/api/state`. Updates the IMU
  motion-gate state and (when `send_enabled`) forwards angle-sensor
  positions as `MotorCommand` to the gripper.
- `stats_printer` — every 2 s, summary line: state, recent gRPC
  latency, per-counter deltas.

### Delta-pipeline order

```
WS msg → dedup(t) → lost? → send? → static? → deadband → scale → cap → gRPC
```

Each step has its own counter in the stats line so you can see exactly
where messages are being dropped.

### Conventions

- Position units: **meters** (matches the proto).
- Rotation: input quaternion XYZW → `Rotation.from_quat` →
  `rotation_matrix_to_6d` → 6-float `dr6d` (first two rows of R,
  flattened). The 6D rotation tool is imported directly from
  `openarm_gripette_simu/rotation.py` so client and server share the
  exact same conversion code.

## Tuning

All knobs are CLI flags. Defaults are tuned for **sim** + a real
OAK-D SR Grabette at 320×200 BasaltVIO.

| Flag | Default | Purpose |
|---|---|---|
| `--ws` | `ws://rgrabette2:8000/...` | Pi WS endpoint |
| `--api` | `http://rgrabette2:8000` | Pi REST endpoint (state polling) |
| `--arm` | `localhost:50052` | Arm gRPC address |
| `--gripper` | `localhost:50051` | Gripper gRPC address |
| `--max-delta-m` | `0.05` | Hard cap on per-step translation magnitude (safety) |
| `--pos-scale` | `1.0` | Linear multiplier on (dx,dy,dz). <1 to dampen, >1 to amplify |
| `--rot-scale` | `1.0` | Multiplier on rotation angle (axis preserved) |
| `--pos-deadband-m` | `0.0005` | Zero translation when raw \|delta\| below this. 0.5 mm catches SLAM noise floor |
| `--rot-deadband-rad` | `0.005` | Zero rotation when raw angle below this. ~0.3° |
| `--imu-gyro-thresh` | `0.05` rad/s | Threshold for IMU "motion" detection (gyro) |
| `--imu-accel-thresh` | `0.6` m/s² | Threshold for IMU "motion" detection (\|accel\|-g) |
| `--static-window-s` | `0.15` | How long IMU must read still before declaring static |
| `--no-imu-gate` | off | Disable the IMU motion gate (deltas always sent when `send=True`) |
| `--no-gripper` | off | Skip the gripper bridge (arm only) |
| `--invert-proximal` / `--invert-distal` | off | Flip sign of angle sensor → gripper motor mapping |
| `--state-rate-hz` | `30.0` | HTTP poll rate of `/api/state` (drives gripper update + IMU gate) |
| `--dry-run` | off | Ping gRPC servers but do NOT send commands. Logs would-be calls |
| `--debug` | off | Per-tick IMU values + per-message decisions |

### Picking scales

Start conservative. For a real robot whose motion seems amplified
relative to your hand:

1. Begin with `--pos-scale 0.3 --rot-scale 0.3`.
2. Move the Grabette ~10 cm in one direction (button pressed).
3. Measure robot end-effector displacement.
4. Adjust scale up/down until the ratio matches what you want
   (typically 1:1).
5. The value that gives 1:1 tells you the calibration discrepancy
   between Grabette frame and robot workspace.

### Safety

Three layers, applied in order on every delta:

1. **Send-toggle gate** (hardware button): nothing leaves the bridge
   unless `send=True`. Releasing the button halts the arm AND the
   gripper instantly — by far the most important safety control.
2. **IMU motion gate**: when the Grabette's IMU reads still for
   `--static-window-s`, all deltas are dropped regardless of SLAM
   noise. Prevents SLAM-drift-induced robot motion when you put the
   Grabette down.
3. **Magnitude cap** (`--max-delta-m`): clamps any single per-step
   translation magnitude. SLAM re-acquisition spikes can produce >1 m
   "deltas"; this caps them.

Cap is applied **after** scale, so the absolute output is bounded
regardless of what `--pos-scale` is set to.

## Troubleshooting

### Symptom → likely cause

| Symptom | Most likely cause |
|---|---|
| `ConnectionRefusedError` at startup | Pi daemon down, or wrong `--ws` / `--api` URL, or gRPC server not running |
| Bridge starts but `ws_recv` stays at 0 | Teleop not started on the Pi. Click "Enter Teleop Mode" in the UI. |
| `ws_recv` climbing but `arm_sent` near zero | `send=False` (button not pressed) OR IMU gate stuck on STATIC. Check stats line for `STATIC` vs `moving` and `arm_skipped_no_send` / `arm_skipped_static`. |
| Robot motion lags 1-2 s behind hand | BasaltVIO can't keep up. Confirm camera resolution is 320×200 (not 640×400) in `oakd_teleop.py`. Check Pi load (`top` should show ~150% on the daemon, not 200%+). |
| Robot motion much bigger than hand motion | Real server workspace/units differ from sim. Use `--pos-scale` and `--rot-scale` to dial back, start with 0.3. |
| Arm drifts when Grabette is held still | IMU gate not triggering. Check the `--debug` `imu:` line — does `static=True` ever appear when you hold still? If not, raise `--imu-accel-thresh` or lower `--imu-gyro-thresh`. |
| Per-frame motion is jerky / stuttery | Deltas getting dropped intermittently. Watch `arm_skipped_static` — if it flickers between 0 and non-zero per window, the gate is borderline. Loosen `--imu-gyro-thresh` / `--imu-accel-thresh`. |
| Gripper moves but arm doesn't | Bridge processes are alive but `send_enabled` is False. The arm + gripper share the toggle, so this should be impossible unless the bridge has gone stale — restart it. |
| Bursty WS data (websocat) | Live-view polling on the UI competing for uvicorn event loop. Click "Enter Teleop Mode" (not just navigate to the page) so the UI pauses its timers and chart iframes. |

### Useful diagnostic commands

```bash
# Quick smoke test: are deltas being produced?
curl http://rgrabette2:8000/api/teleop/status | jq

# Watch the WS stream directly (skip the bridge):
websocat ws://rgrabette2:8000/api/teleop/stream

# CPU load on the Pi during teleop:
ssh rasp@rgrabette2 'top -bn 1 -p $(pgrep -f "python -m grabette") | tail -3'

# Bridge in maximum-visibility mode:
uv run python scripts/teleop_bridge.py --dry-run --debug
```

## Known limitations

- **No drift correction**: BasaltVIO has small but non-zero pose drift
  even when held still. The deadband + IMU gate mask it in steady
  state, but cumulative drift over a long teleop session will manifest
  as a slowly accumulating offset on the robot. Re-enter teleop to
  reset.
- **No collision avoidance**: the robot's gRPC server is responsible
  for its own joint-limit / workspace-limit / collision handling. The
  bridge has no map of the robot's surroundings.
- **Single-consumer WS**: `/api/teleop/stream` is intended for one
  bridge. Multiple WS subscribers all get the stream but each issues
  its own gRPC sends — don't run two bridges against the same robot.
- **Real-robot calibration scale**: as of this writing, the real
  `grpc_server_real` motion is over-amplified relative to the sim by
  some empirical factor (~3×). Use `--pos-scale` / `--rot-scale` to
  compensate; the root cause is in the real server's interpretation of
  CartesianDelta and is tracked separately.

## See also

- `openarm_gripette_simu/proto/{arm,gripper}.proto` — gRPC interface
  definitions.
- `openarm_gripette_simu/rotation.py` — 6D rotation encoding used by
  both bridge and server.
- `openarm_gripette_simu/arm_servicer.py` — server-side delta
  integrator + IK (reference implementation for the sim; real-robot
  server should follow the same convention).
