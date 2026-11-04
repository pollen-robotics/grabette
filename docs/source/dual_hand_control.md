# Control one or two recorders from a terminal

Use the standalone `packages/grabette/scripts/grabette_control.py` script to
start either hand, send concurrent start requests to both hands, and monitor
recording and saved-file status without keeping the browser or an editor open.
It runs on a workstation with Python 3.11+ and uses only the standard library.
Interactive operation supports macOS and Linux terminals; `--once` provides a
read-only snapshot for other environments.

## Configure the recorders

Both devices need the local-control API included with this script. Each must
have the correct `GRABETTE_HAND` setting (`left` or `right`) and a distinct
device ID. Their HTTP services must be reachable from the workstation over a
trusted LAN. The script does not need an SSH connection, a Hugging Face login,
or a fleet session.

From the repository root:

```bash
cp packages/grabette/scripts/devices.example.json packages/grabette/scripts/devices.local.json
```

Edit `devices.local.json` and replace the example host names with each device's
actual reachable address. This local configuration file is Git-ignored. Remove
either top-level hand object when using just one recorder. If both are
configured but one is offline, the other can still be controlled individually;
a request to start both first checks that both are ready.

For additional protection against an address being reused, add
`"expected_device_id": "<the-device-id>"` to each hand object. Read the identity
without starting a recording:

```bash
curl http://grabette-left.local:8000/api/capture/control-info
```

The client checks the reported hand and stable device ID before every control
operation. A changed identity requires reconnecting with `r`; an explicitly
configured `expected_device_id` must still match. Redirects and credentials
embedded in addresses are refused. These identity checks prevent accidental
misaddressing; they are not authentication for an untrusted network.

## Run and control

```bash
python3 packages/grabette/scripts/grabette_control.py \
  --config packages/grabette/scripts/devices.local.json
```

- `1`, `2`, `3`: start the left hand, right hand, or all configured hands.
- `4`, `5`, `6` (or `s`): stop the left hand, right hand, or all configured hands.
- `7`, `8`, `9`: select the latest saved episode on the left, right, or both.
  The display shows the exact episode IDs. Press `y` to permanently delete
  those IDs, or `n` to cancel. Confirmation expires after 30 seconds. A newer
  episode appearing during confirmation does not change the selected target.
- `r`: reconnect and check device identities again.
- `q`: exit the monitor. **Exiting does not stop an active recording.**

A stop can be requested while start is in progress. It waits for initialization
to resolve and sends an episode-specific stop; it does not interrupt camera
initialization mid-flight. Other duplicate or conflicting requests are refused.
Backends that support saving in the background can accept a new recording while
older episodes save, provided their own `blocked_reason` permits it. Deletion
requires both recording and saving to be idle.

The display includes camera/IMU/angle/head counters when available, recording
duration, queued saves, buffer use, battery/charging state, CPU temperature,
disk space, available memory, and the active task session. Older backends may
not expose every metric; missing values appear as `?`, rather than as zero.
Warnings flag stale data, stalled selected counters, hardware/saving faults,
low battery, low disk/memory, high temperature, and buffer pressure. Slow-save
time counts only observed idle time; active recording and disconnected intervals
do not falsely accumulate as saving stalls. The monitor does not automatically
stop, restart, or delete data in response to a warning.

For a read-only snapshot:

```bash
python3 packages/grabette/scripts/grabette_control.py \
  --config packages/grabette/scripts/devices.local.json --once --json
```

`--logs DIRECTORY` changes the location of per-run JSONL control logs. The
default is `~/.local/state/grabette-control`. Logs contain command outcomes,
episode IDs, and observation times; they are local and are not uploaded. A
logging failure is displayed but does not prevent a stop request.

## Understand what is confirmed

An acknowledged start is not proof that usable data has arrived. The display
separately waits for fresh, increasing counters within the same episode. The
device advertises `required_capture_counters`: the main camera, enabled angle
sensors, and IMU when using an OAK-D. A Gemini 305 rig without IMU is therefore
not held waiting for nonexistent IMU samples.

For custom capture streams, add their counter names to a hand's
`"required_counters"` list. Accepted names are `frame_count`, `imu_sample_count`,
`angle_sample_count`, and `head_frame_count`; the main camera is always required.
For a compatible older local-control API that does not advertise its counters,
only the main camera plus any explicitly configured counters are confirmed.
The display deliberately calls these the **selected data counters**. Their
growth does not establish that every optional stream is present or that files
decode correctly. Saved-file publication is shown separately. A legacy API
reporting files as present is not labeled as independently validated.

## Software concurrency and failure handling

The `3` command sends separate local HTTP requests concurrently. Each device
warms up independently. This is **not hardware synchronization or a guarantee
of simultaneous exposure**, and it does not create a shared episode timestamp.
Keep the original per-device timestamps for downstream alignment. Use the
existing fleet scheduling path when a shared scheduled start is required.

If one concurrent start succeeds and another fails, the client attempts to
stop only the positively acknowledged episode, using its exact ID. An unknown
result, such as a lost HTTP reply, is not automatically retried and is not
treated as a definite failure. Check each device's fresh status before acting.
Stops also specify the expected episode ID, so a delayed request cannot stop a
different take. Two-device deletion is not a transaction: a connection failure
can leave one selected episode deleted and the other intact; inspect the result
shown for each hand before doing anything further.

## Local-control API compatibility

This change adds `GET /api/capture/control-info`, which advertises the hand,
device ID, counter requirements, and control protocol versions. Explicit
`local_only: true` on `POST /api/episodes/start` and `/api/episodes/stop` bypasses
fleet forwarding for the addressed device. `expected_episode_id` on stop adds
the mismatch protection described above. Existing requests without these
fields retain their fleet behavior. The client refuses control if a server
does not advertise support; it never sends flags that an older server could
silently ignore. Read-only status can still be inspected when control is
unavailable, and the interface identifies the device as unverified.
