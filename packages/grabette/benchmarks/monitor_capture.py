"""Sample the Pi's load while recording, to explain holes in the RPi camera.

Run on the Pi, with the SYSTEM python, in a second shell, BEFORE recording:
    python3 benchmarks/monitor_capture.py            # writes ~/capture_monitor/<date>/
    python3 benchmarks/monitor_capture.py --period 0.02 --out /tmp/mon

Leave it running for the whole session, then Ctrl-C. Feed its folder to
benchmarks/analyze_frame_gaps.py --monitor <folder> to see what the Pi was doing
during each hole.

Every --period (default 25 ms) it writes one row of monitor.csv:
  - boottime_s / utc   CLOCK_BOOTTIME (the RPi camera's SensorTimestamp clock)
                       and wall clock, so the analyzer can place a hole on it;
  - cpu_busy / iowait  % over the period, all cores;
  - psi_*_ms           ms stalled during the period (/proc/pressure): io_full
                       means EVERY task was waiting on I/O;
  - dirty_kb / writeback_kb   page cache waiting for / being written to disk;
  - disk_*             the SD card (or --disk): kB written, ms spent writing,
                       requests in flight;
  - d_threads          grabette threads in uninterruptible sleep (blocked in
                       the kernel, usually on I/O), with their names.
Once a second it adds the throttling flags, SoC temperature and ARM clock
(vcgencmd). kernel.log gets the kernel messages (camera/CSI, USB, mmc errors).
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CLOCK_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", 7)


def read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def cpu_times() -> tuple[int, int, int]:
    """(total, idle, iowait) jiffies, all cores."""
    v = [int(x) for x in read("/proc/stat").split("\n", 1)[0].split()[1:]]
    if len(v) < 5:
        return 0, 0, 0
    return sum(v), v[3], v[4]


def psi_totals() -> dict[str, int]:
    """Cumulative stall time (µs) per resource, from /proc/pressure."""
    out = {}
    for res in ("cpu", "io", "memory"):
        for line in read(f"/proc/pressure/{res}").splitlines():
            kind, *fields = line.split()
            for f in fields:
                if f.startswith("total="):
                    out[f"{res}_{kind}"] = int(f[6:])
    return out


def meminfo() -> tuple[int, int]:
    dirty = wb = 0
    for line in read("/proc/meminfo").splitlines():
        if line.startswith("Dirty:"):
            dirty = int(line.split()[1])
        elif line.startswith("Writeback:"):
            wb = int(line.split()[1])
    return dirty, wb


def default_disk() -> str:
    """Block device holding the data dir (mmcblk0 on an SD-booted Pi)."""
    for line in read("/proc/diskstats").splitlines():
        name = line.split()[2]
        if name in ("mmcblk0", "nvme0n1", "sda"):
            return name
    return "mmcblk0"


def disk_stats(disk: str) -> tuple[int, int, int]:
    """(sectors written, ms spent writing, I/Os in flight)."""
    for line in read("/proc/diskstats").splitlines():
        f = line.split()
        if len(f) > 11 and f[2] == disk:
            return int(f[9]), int(f[10]), int(f[11])
    return 0, 0, 0


def find_pids(pattern: str) -> list[int]:
    pids = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid():
            cmd = read(f"/proc/{d}/cmdline").replace("\0", " ")
            if pattern in cmd and "monitor_capture" not in cmd:
                pids.append(int(d))
    return pids


def blocked_threads(pids: list[int]) -> list[str]:
    """Names of the threads of `pids` in state D (uninterruptible sleep)."""
    names = []
    for pid in pids:
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            stat = read(f"/proc/{pid}/task/{tid}/stat")
            # "tid (comm) S ..." — comm may contain spaces/parens.
            close = stat.rfind(")")
            if close > 0 and stat[close + 2:close + 3] == "D":
                names.append(stat[stat.find("(") + 1:close])
    return names


def vcgencmd(*args: str) -> str:
    try:
        out = subprocess.run(["vcgencmd", *args], capture_output=True,
                             text=True, timeout=1).stdout.strip()
        return out.split("=", 1)[-1]
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", type=float, default=0.025, help="seconds (default 0.025)")
    p.add_argument("--out", type=Path, help="output folder (default ~/capture_monitor/<date>)")
    p.add_argument("--disk", default=None, help="block device in /proc/diskstats (default: auto)")
    p.add_argument("--process", default="grabette",
                   help="cmdline substring of the daemon (default: grabette)")
    args = p.parse_args()

    out = args.out or Path.home() / "capture_monitor" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    disk = args.disk or default_disk()

    # Kernel messages with precise wall-clock stamps; best effort (may need
    # the systemd-journal or adm group).
    klog = open(out / "kernel.log", "w")
    try:
        kproc = subprocess.Popen(["journalctl", "-k", "-f", "-n", "0", "-o", "short-iso-precise"],
                                 stdout=klog, stderr=subprocess.STDOUT)
    except OSError:
        kproc = None

    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if not os.path.exists("/proc/pressure/io"):
        print("warning: /proc/pressure is absent, the psi_* columns will stay at 0. "
              "Enable PSI with psi=1 in /boot/firmware/cmdline.txt and reboot.")

    pids = find_pids(args.process)
    print(f"Writing {out}/monitor.csv every {args.period * 1000:.0f} ms "
          f"(disk {disk}, {args.process} pids {pids or 'not found yet'}). Ctrl-C to stop.")

    fields = ["boottime_s", "utc", "cpu_busy", "iowait",
              "psi_cpu_some_ms", "psi_io_some_ms", "psi_io_full_ms", "psi_memory_some_ms",
              "dirty_kb", "writeback_kb", "disk_write_kb", "disk_write_ms", "disk_inflight",
              "d_threads", "throttled", "temp", "arm_mhz"]
    f = open(out / "monitor.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(fields)

    prev_cpu, prev_psi, prev_disk = cpu_times(), psi_totals(), disk_stats(disk)
    next_t = time.monotonic()
    last_slow = 0.0
    last_pid_scan = time.monotonic()
    rows = 0
    while not stop:
        next_t += args.period
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.monotonic()  # we fell behind: don't burst to catch up

        now = time.monotonic()
        if now - last_pid_scan > 5:  # follow a daemon restart
            pids = find_pids(args.process)
            last_pid_scan = now

        cpu, psi, dsk = cpu_times(), psi_totals(), disk_stats(disk)
        dt = max(cpu[0] - prev_cpu[0], 1)
        dpsi = {k: (psi.get(k, 0) - prev_psi.get(k, 0)) / 1000 for k in
                ("cpu_some", "io_some", "io_full", "memory_some")}
        dirty, wb = meminfo()
        slow = ["", "", ""]
        if now - last_slow >= 1.0:
            slow = [vcgencmd("get_throttled"), vcgencmd("measure_temp"),
                    str(int(vcgencmd("measure_clock", "arm") or 0) // 1_000_000 or "")]
            last_slow = now
        w.writerow([
            f"{time.clock_gettime(CLOCK_BOOTTIME):.4f}",
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            round(100 * (dt - (cpu[1] - prev_cpu[1]) - (cpu[2] - prev_cpu[2])) / dt, 1),
            round(100 * (cpu[2] - prev_cpu[2]) / dt, 1),
            round(dpsi["cpu_some"], 1), round(dpsi["io_some"], 1),
            round(dpsi["io_full"], 1), round(dpsi["memory_some"], 1),
            dirty, wb,
            (dsk[0] - prev_disk[0]) // 2, dsk[1] - prev_disk[1], dsk[2],
            "|".join(blocked_threads(pids)),
            *slow,
        ])
        prev_cpu, prev_psi, prev_disk = cpu, psi, dsk
        rows += 1
        if rows % 400 == 0:
            f.flush()

    f.close()
    if kproc:
        kproc.terminate()
    klog.close()
    print(f"\n{rows} samples in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
