"""Free space on the card recordings land on.

One measurement for everyone: the backend refuses a start below
settings.min_free_disk_gb, and the dashboard shows the same reading so the
refusal never comes as a surprise.
"""
from __future__ import annotations

import os
from pathlib import Path

GB = 1024**3

# Below this many times the minimum, Storage turns orange on the dashboard:
# a session or two before the device starts refusing.
LOW_FACTOR = 2.0


def free_bytes(path: Path) -> int | None:
    """Free bytes on the filesystem holding `path` (None: cannot tell).

    data_dir may not exist yet on a fresh device: measure its nearest existing
    parent, which is the same card."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return st.f_bavail * st.f_frsize


def total_bytes(path: Path) -> int | None:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return st.f_blocks * st.f_frsize


def disk_space_message(free: int | None, min_free_gb: float) -> str:
    """The fault for `free` bytes left ("" = enough, or unknown)."""
    if free is None or free >= min_free_gb * GB:
        return ""
    return (f"SD card almost full ({free / GB:.1f} GB free, "
            f"{min_free_gb:g} GB needed): free some space")


def storage_level(free_gb: float | None, min_free_gb: float) -> str:
    """"full" (recording refused), "low" (soon), "ok", or "unknown"."""
    if free_gb is None:
        return "unknown"
    if free_gb < min_free_gb:
        return "full"
    if free_gb < min_free_gb * LOW_FACTOR:
        return "low"
    return "ok"
