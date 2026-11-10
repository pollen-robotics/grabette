# Benchmarks: does a branch regress against develop?

Tools to run on one or more grabettes before merging a PR that touches capture
(camera, depth camera, angles, start/stop, sync, daemon load). They measure
everything the November 2026 sync review checked, on the baseline (`develop`)
and on the branch, and compare the two.

| Tool | Runs on | Python | Does |
|---|---|---|---|
| `run_episodes.py` | Pi (or workstation with `--api`) | system `python3` | Records a session through the daemon's API: solo or fleet group, mixed lengths, stops before the SD card fills up, keeps the daemon and kernel logs |
| `episode_report.py` | Pi or workstation | system `python3` | Measures every episode (RGB, depth, angles, start, sync) and sums up per device |
| `compare_reports.py` | anywhere | system `python3` | Baseline vs candidate report: Markdown table for the PR, exit code 1 on regression |
| `measure_chrono_sync.py` | workstation | `uv run` (installs RapidOCR in a throw-away env) | Real sync, from a chronometer filmed by every camera |
| `bench_i2c_angles.py` | Pi, daemon stopped | grabette venv | Angle-sensor I2C reads under a chosen load (none, RPi camera, RPi recording, depth camera, depth recording) |
| `monitor_capture.py` | Pi, during a session | system `python3` | Samples CPU, I/O, page cache, blocked threads, throttling |
| `analyze_frame_gaps.py` | Pi or workstation | system `python3` | Explains each hole: which streams stopped, what the Pi was doing (`--monitor`, `--journal`) |

Paths below are relative to `packages/grabette`.

**What it compares against**: there is no stored reference. The reviewer
records the baseline themselves, on `develop`, on the same grabette(s), just
before or after the branch. Grabettes differ (one had angle holes every 14 min,
another every 37 min, on the same code), and so do the place, the scene in
front of the depth camera and the SD card's fill level: only a baseline taken
in the same conditions tells a regression from a difference of setup.

## What is checked

| Check | Measured by | Judged as |
|---|---|---|
| RGB timestamps ↔ frames (PR 239): alignment valid, no unwritten camera frame, as many timestamps as frames | `episode_report` from `metadata.json` / `video_timestamp_alignment` | invariant (must stay 0) |
| RGB writer queue never full (300) | same | invariant |
| PTS offset (since `ad4eb7b`): found from the encoder's origin or by matching alone; episodes where matching alone was ambiguous | `video_timestamp_alignment` | info |
| Videos decodable: frames ffprobe decodes in `raw_video.mp4`, `dcam_left.mp4`, `dcam_depth.mkv` vs their timestamps (`--decode`) | ffprobe `-count_frames` | invariant |
| RGB holes: > 250 ms (daemon flag), > 150 ms, > 30 ms after the first 1.5 s | `frame_timestamps.json` | rate per hour, Poisson test |
| RGB start-up burst (holes in the first 1.5 s), first frame after t=0, frame rate | same | value with tolerance |
| Pictures put back in order by the writer (`reordered_output_count`) | `video_timestamp_alignment` | info |
| Depth: holes > 100 ms, frames the camera numbered but the host never received, encoder-queue drops, IR = depth counts, first frame vs t=0 | `dcam_left_timestamps.json`, `metadata.dcam` | rate / invariant / value |
| Angles: loop rate (Hz), stalls > 100 ms, I2C timeouts > 500 ms | `angle_data.json` | value / rate |
| Start: `start_capture_ms`, starts later than 1 s, `capture_started_utc − scheduled_start_utc` | `metadata.sync` | value / rate |
| End: spread of the streams' last samples, a stream stopping > 0.5 s before the others | each stream's last timestamp | value / rate |
| Stop: how long the stop call takes; the daemon's stop phases (`camera_stop`, muxes...) | `run_episodes` (`run.json`, `daemon.log`) | value / reported |
| Clock: NTP synchronised and offset, at the start and end of a run | `run_episodes` (`timedatectl` or `chronyc`) | warning in the comparison |
| Group: spread of `capture_started_utc` between devices of one episode | `metadata.sync` of each device | reported |
| Real sync: RPi − Orbbec offset and its drift within an episode, real t=0 gap left/right, error of `capture_started_utc` | `measure_chrono_sync.py` → `episode_report --chrono` | value with tolerance and limit |
| Daemon / kernel log lines: `Angle loop stalled`, I2C read failures, flagged frame holes, depth camera not ready, alignment errors, tracebacks, I2C / USB / mmc / under-voltage kernel messages | `run_episodes` logs → `episode_report --run` | reported |

Rare events need time: angle I2C timeouts came about once per 15 to 40 min of
recording, long RGB holes about as rarely. `compare_reports.py` says "worse,
not significant" rather than "regression" when the counts are too small to
tell; record about an hour per side when the PR could affect them.

## Protocol for a PR

Same grabette(s), same place and scene for both sides (the depth camera needs
a scene with valid depth, or every start waits up to 10 s for it). Before each
side: dashboard closed (its camera preview loads the Pi), same Bluetooth state,
clock synchronised (`timedatectl timesync-status`), enough free space
(`df -h ~/grabette-data`), a throwaway task selected (see `run_episodes.py
--help`), and run inside `tmux` (a dropped SSH session kills the script before
it writes its logs).

### 1. Baseline (develop)

```bash
# on the Pi
cd <repo> && git checkout develop && git pull && sudo systemctl restart grabette
cd packages/grabette
python3 benchmarks/run_episodes.py --label "develop @ $(git rev-parse --short HEAD)"
# ~30 episodes of 5 / 30 / 120 s, about 26 min; --total 3600 --durations 120 for an hour
python3 benchmarks/episode_report.py ~/grabette-data/episodes --run ~/benchmarks/<run folder> \
    --label "develop @ <sha>" --out ~/benchmarks/report-develop.json
```

Add `--decode` to also count the frames each video really decodes into. It
takes about 1 min per 2 min episode on a Pi: run it on a workstation on the
downloaded episodes if time matters, on both sides or neither.

### 2. Candidate (the PR branch)

Same commands after `git checkout <branch>` and a daemon restart, with
`--out ~/benchmarks/report-<branch>.json`.

### 3. Sync, when the PR touches start, timestamps or the cameras

Two grabettes in a fleet session, both filming the same on-screen chronometer
(`HH:MM:SS.mmm`). On one of them:

```bash
python3 benchmarks/run_episodes.py --group --durations 10,30,120 --count 12 --label "<branch> chrono"
```

The 120 s takes show a drift, or a one-frame step, within an episode. Check
`timedatectl timesync-status` on every member before starting: the script only
records the clock of the Pi it runs on.

Download the session from the dashboard (multi-device raw download), then on a
workstation:

```bash
uv run benchmarks/measure_chrono_sync.py ~/Downloads/<session>-rawdl-<...>/
python3 benchmarks/episode_report.py ~/Downloads/<session>-rawdl-<...>/ \
    --chrono ~/Downloads/<session>-rawdl-<...>/chrono_sync/results.json \
    --label "<branch> chrono" --out report-<branch>-chrono.json
```

Do the same on develop and compare the two chrono reports.

### 4. Compare

```bash
python3 benchmarks/compare_reports.py report-develop.json report-<branch>.json --out compare.md
```

Paste `compare.md` in the PR. `--device <id>` compares one grabette only.

## Not covered

- **Angle timing against the cameras**: the chronometer does not show the
  angles, so their rate and holes are checked, not their time offset. It
  needs a physical test (the gripper snapping shut in front of the camera,
  the instant compared in both streams).
- **Throttling**: under-voltage shows in the kernel lines; the thermal soft
  limit only in `monitor_capture.py`.
- **Downstream**: the LeRobot conversion (it must read the mp4 frame by frame,
  `-fps_mode passthrough`), and the OAK-D IMU on grabettes that have one.

## Digging into a regression

- **Holes**: run `python3 benchmarks/monitor_capture.py --out /dev/shm/mon` in
  a second shell during the session, then
  `python3 benchmarks/analyze_frame_gaps.py ~/grabette-data/episodes/<id> --monitor /dev/shm/mon --journal`.
  `/dev/shm` is in RAM: copy the folder before rebooting.
- **Angle holes**: the daemon logs each failed read with its errno and bus;
  `bench_i2c_angles.py --mode i2c|camera|record|orbbec|orbbec-record`
  reproduces the load without the daemon (stop it first) to find which part
  triggers them.
- **Sync**: `chrono_sync/results.json` has each stream's offset per 5 s window.

## Reading the numbers

- On `develop` before PR 239, the RGB timestamps don't match the written
  frames (`frame_count_mismatch`): RGB timing figures are then reported as not
  comparable.
- The angle loop rate depends on the Pi's load: ~99 Hz with the camera alone,
  75 to 93 Hz while recording.
- `capture_started_utc − scheduled_start_utc` is a few ms when the start is on
  time (PR 244); a start delayed by the depth camera's readiness wait shows as
  several seconds here and in `start_capture_ms`.
