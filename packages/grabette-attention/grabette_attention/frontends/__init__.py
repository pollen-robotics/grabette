"""Front ends. Both consume `FrameAnalysis`; neither computes anything."""

from typing import Iterable


def camera_labels(camera_keys: Iterable[str]) -> dict[str, str]:
    """A short, collision-free display label per camera key.

    A camera key's last dot-segment (e.g. "cam0" from
    "observation.images.cam0") makes a fine label on its own, but two keys
    that differ only in an earlier segment -- exactly the shape of a stereo
    pair, e.g. "observation.images.left.cam0" and
    "observation.images.right.cam0" -- would collapse onto the same label and
    silently drop one view's overlay: same output filename, same rerun entity.
    So a key keeps its short label only when no other key would produce the
    same one; a colliding key falls back to its full, and therefore unique,
    key instead.
    """
    keys = list(camera_keys)
    short = {key: key.rsplit(".", 1)[-1] for key in keys}
    counts: dict[str, int] = {}
    for label in short.values():
        counts[label] = counts.get(label, 0) + 1
    return {key: (label if counts[label] == 1 else key) for key, label in short.items()}
