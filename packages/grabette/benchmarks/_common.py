"""Helpers shared by the benchmark scripts (standard library only: they run with
the Pi's system python3). Each script finds this file next to itself."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def parse_utc(s) -> datetime | None:
    """An ISO time, as UTC when it carries no zone; None if it isn't one."""
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def episode_time(name: str) -> datetime | None:
    """When an episode was created, from its id (UTC YYYYMMDD_HHMMSS, see
    grabette.task.episode_id_for)."""
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def find_episodes(paths: list[Path]) -> list[Path]:
    """Folders holding one device's take of an episode (a frame_timestamps.json),
    given those folders or folders containing them."""
    out = []
    for p in paths:
        if not p.exists():
            # An unmatched shell glob arrives here verbatim.
            print(f"warning: {p} does not exist", file=sys.stderr)
            continue
        if (p / "frame_timestamps.json").exists():
            out.append(p)
            continue
        found = sorted(f.parent for f in p.rglob("frame_timestamps.json"))
        if not found:
            print(f"warning: no episode (frame_timestamps.json) under {p}", file=sys.stderr)
        out += found
    return out


def journal(args: list[str], output: str = "short-iso-precise", timeout: float = 120) -> str:
    """journalctl's output for these arguments, or why it failed."""
    try:
        r = subprocess.run(["journalctl", "--no-pager", "-o", output, *args],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout or r.stderr
    except (OSError, subprocess.SubprocessError) as e:
        return f"journalctl failed: {e}\n"


def journal_since(since_epoch: float, *args: str) -> str:
    """The journal since a time (epoch s), in UTC."""
    return journal(["--utc", "--since", f"@{int(since_epoch)}", *args], output="short-precise")
