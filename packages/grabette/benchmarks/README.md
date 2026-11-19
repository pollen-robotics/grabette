# Benchmarks: does a branch regress against develop?

Run these before merging a PR that touches capture: cameras, angles,
start/stop, sync or the daemon's load. You record the same session on
`develop` (the baseline) and on the branch (the candidate), on the same
grabette(s), then compare the two reports.

Commands are run from `packages/grabette`. They use the Pi's system
`python3`, except `measure_chrono_sync.py` (`uv run`) and
`bench_i2c_angles.py` (the grabette venv).

## Quick start

### 0. Before each side

- Same grabette(s), same place, same scene in front of the depth camera.
- Pi cooled down to the same temperature (`vcgencmd measure_temp`; above
  80 °C the CPU slows down and frames get lost).
- Dashboard closed (its camera preview loads the Pi), same Bluetooth state.
- Clock synchronised: `timedatectl timesync-status`.
- Enough free space: `df -h ~/grabette-data`.
- A throwaway task selected on the grabette.
- Inside `tmux`: a dropped SSH session kills the script before it saves its logs.

### 1. Record the baseline (on the Pi)

```bash
git checkout develop && git pull && sudo systemctl restart grabette
cd packages/grabette
python3 benchmarks/run_episodes.py --label "develop @ $(git rev-parse --short HEAD)"
```

It records 30 episodes of 5, 30 and 120 s (about 26 min) and writes
`~/benchmarks/<date>_episodes/`. When it ends it prints the next command;
run it:

```bash
python3 benchmarks/episode_report.py ~/grabette-data/episodes \
    --run ~/benchmarks/<date>_episodes --label "develop @ <sha>" \
    --out ~/benchmarks/report-develop.json
```

### 2. Record the candidate (on the Pi)

Same two commands after `git checkout <branch>` and
`sudo systemctl restart grabette`, with `--out ~/benchmarks/report-<branch>.json`.

### 3. Compare (anywhere)

```bash
python3 benchmarks/compare_reports.py ~/benchmarks/report-develop.json \
    ~/benchmarks/report-<branch>.json --out compare.md
```

Paste `compare.md` in the PR. The script exits with 1 when something
regressed. `--device <id>` compares one grabette only.

## Options worth knowing

| Need | Option |
|---|---|
| Rare events (angle I2C timeouts, long RGB holes: about one per 15 to 40 min) | `run_episodes.py --total 3600 --durations 120`: one hour of 2 min takes, on each side |
| Two grabettes in a fleet session | `run_episodes.py --group` (on one of them) |
| Daemon on another machine | `run_episodes.py --api http://<grabette>.local:8000` |
| Count the frames each video really decodes into | `episode_report.py --decode` (about 1 min per 2 min episode on a Pi: do it on a workstation, on both sides or neither) |
| Only some episodes | `episode_report.py --since/--until YYYYMMDD_HHMMSS` |

## Reading the comparison

Each line gets a verdict:

- **invariant** (must stay 0: alignment valid, every camera frame written,
  as many timestamps as frames...): `REGRESSION` when the branch breaks one
  that develop kept, or breaks it far more often.
- **rate** (rare events over time: holes, I2C timeouts, late starts,
  depth-camera losses): a statistical test (Poisson) says whether the
  branch's count is too high for develop's rate. With few minutes recorded it
  says `worse, not significant` instead of `REGRESSION`. Events are counted,
  not their cost: a USB hiccup that loses 30 depth frames is one loss (the
  frames are shown as info).
- **value** (medians and maxima: start delay, frame rate, angle rate...):
  `REGRESSION` beyond a tolerance, or over an absolute limit.
- **info**: shown, not judged.

Warnings above the table: the clock was not synchronised, or a side ran hot
or throttled, or the two sides' temperatures differ by more than 5 °C. Redo
the run in that case.

Typical values: the angle loop runs at about 99 Hz with the camera alone and
75 to 93 Hz while recording. `capture_started − scheduled` is a few ms when
the start is on time; a start waiting for the depth camera shows as seconds.

## Sync, when the branch touches start, timestamps or the cameras

Two grabettes in a fleet session, both filming the same on-screen
chronometer (`HH:MM:SS.mmm`). On one of them:

```bash
python3 benchmarks/run_episodes.py --group --durations 10,30,120 --count 12 --label "<branch> chrono"
```

Check `timedatectl timesync-status` on every grabette first: the script only
records the clock of the Pi it runs on. Download the session from the
dashboard (multi-device raw download), then on a workstation:

```bash
uv run benchmarks/measure_chrono_sync.py ~/Downloads/<session>-rawdl-<...>/
python3 benchmarks/episode_report.py ~/Downloads/<session>-rawdl-<...>/ \
    --chrono ~/Downloads/<session>-rawdl-<...>/chrono_sync/results.json \
    --label "<branch> chrono" --out report-<branch>-chrono.json
```

Do the same on develop and compare the two chrono reports with
`compare_reports.py`.

## Digging into a regression

- **Frame holes**: run `python3 benchmarks/monitor_capture.py --out /dev/shm/mon`
  in a second shell during the session, then
  `python3 benchmarks/analyze_frame_gaps.py ~/grabette-data/episodes/<id> --monitor /dev/shm/mon --journal`.
  It shows which streams stopped, what the Pi was doing and the daemon's log
  around each hole. `/dev/shm` is in RAM: copy the folder before rebooting.
- **Angle holes**: the report counts the angle loop's stalls and read errors
  (from the daemon log). With the daemon stopped,
  `bench_i2c_angles.py --mode i2c|camera|record|orbbec|orbbec-record|both-record`
  reproduces each load on its own to find which one triggers them.
- **Sync**: `chrono_sync/results.json` has each stream's offset per 5 s window.

## The scripts

| Script | Runs on | Does |
|---|---|---|
| `run_episodes.py` | Pi (or workstation with `--api`) | Records a session through the daemon's API (solo or group, mixed lengths), stops before the SD card fills up, keeps the daemon and kernel logs, the clock and the temperature |
| `episode_report.py` | Pi or workstation | Measures every episode (RGB, depth, angles, start, end, sync) and sums up per device |
| `compare_reports.py` | anywhere | Baseline vs candidate: Markdown table, exit code 1 on regression |
| `measure_chrono_sync.py` | workstation, `uv run` | Real sync, read from a chronometer filmed by every camera |
| `monitor_capture.py` | Pi, during a session | Samples CPU, I/O, page cache, blocked threads, throttling |
| `analyze_frame_gaps.py` | Pi or workstation | Explains each hole (`--monitor`, `--journal`) |
| `bench_i2c_angles.py` | Pi, daemon stopped, grabette venv | Angle-sensor reads under a chosen load |
| `_common.py` | — | Helpers shared by the scripts |

## What is measured

| Check | From |
|---|---|
| RGB timestamps ↔ frames: alignment valid, no unwritten camera frame, as many timestamps as frames, writer queue never full | `metadata.json` (`video_timestamp_alignment`) |
| RGB holes > 250 ms (the daemon's flag), > 150 ms, > 30 ms after the first 1.5 s; start-up burst; first frame after t=0; frame rate | `frame_timestamps.json` |
| Depth: holes > 100 ms, losses of frames the camera numbered, encoder-queue drops, IR = depth counts, first frame vs t=0 | `dcam_*_timestamps.json`, `metadata.dcam` |
| Angles: loop rate, stalls > 100 ms, I2C timeouts > 500 ms | `angle_data.json` |
| Start: `start_capture_ms`, starts later than 1 s, `capture_started − scheduled` | `metadata.sync` |
| End: spread of the streams' last samples, a stream stopping > 0.5 s early | each stream's last timestamp |
| Stop: duration of the stop call, the daemon's stop phases | `run.json`, `daemon.log` |
| Videos decodable (`--decode`) | ffprobe |
| Log lines: angle stalls and read errors, flagged frame holes, depth camera not ready, alignment failures, tracebacks, I2C / USB / mmc / under-voltage kernel messages | `daemon.log`, `kernel.log` |
| Clock and heat at the start and end of a run | `run.json` |
| Group: spread of `capture_started_utc` between devices; real sync from the chronometer | `metadata.sync`, `measure_chrono_sync.py` |

On `develop` before PR 239, RGB timestamps don't match the written frames:
RGB timing figures are then reported as not comparable.

## Not covered

- **Angle timing against the cameras**: the chronometer doesn't show the
  angles. It needs a physical test (the gripper snapping shut in front of the
  camera, the instant compared in both streams).
- **Thermal soft limit**: only `monitor_capture.py` sees it; under-voltage
  shows in the kernel lines.
- **Downstream**: the LeRobot conversion, and the IMU of grabettes that have one.
