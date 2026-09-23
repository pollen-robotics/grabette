"""Verdict for a finished test recording.

Its own module, free of gradio, so the rules can be imported and tested
anywhere — `grabette.ui.app` needs the `ui` extra, which CI does not install,
and a rule nobody runs is a rule nobody checks.

Reads the episode back from disk (`GET /api/episodes/{id}` +
`/check`) rather than the stop response: with the recording started and stopped
on the grabette's own button, the dashboard never sees a stop response, and what
is on disk is the better answer anyway.
"""

from __future__ import annotations

import html

# A test recording shorter than this is almost certainly a stray press rather
# than a recording, and reports counters too small to mean anything.
_TEST_MIN_SECONDS = 2.0

# episode_check names the depth camera's artifacts canonically (dcam_*); an
# episode recorded before the rename carries oakd_* and is reported under the
# canonical name anyway, so one prefix would do — both are kept because the
# check's vocabulary is the one thing here we do not own.
_DEPTH_PREFIXES = ("dcam_", "oakd_")

_OK = "#10b981"
_BAD = "#ef4444"
_UNKNOWN = "#94a3b8"


def _chip(state: str, label: str, detail: str) -> str:
    """One counter as a pill. state is "ok", "bad" or "unknown"."""
    color = {"ok": _OK, "bad": _BAD}.get(state, _UNKNOWN)
    mark = {"ok": "✓", "bad": "✗"}.get(state, "?")
    return (
        '<span style="display:inline-flex;align-items:baseline;gap:.4rem;'
        'padding:.3rem .7rem;border-radius:999px;font-size:.82rem;'
        'background:var(--background-fill-secondary,#f1f5f9);'
        f'border:1px solid {color}33;">'
        f'<span style="color:{color};font-weight:700;">{mark}</span>'
        f'<span style="opacity:.75;">{html.escape(label)}</span>'
        f'<strong>{html.escape(detail)}</strong></span>'
    )


def _card(color: str, headline: str, body: str) -> str:
    return (
        '<div style="border-radius:12px;padding:1rem 1.1rem;'
        'background:var(--background-fill-secondary,#f8fafc);'
        f'border:1px solid {color}55;border-left:4px solid {color};">'
        f'<div style="font-weight:700;font-size:1rem;color:{color};">'
        f'{html.escape(headline)}</div>{body}</div>'
    )


def _depth_missing(check: dict | None) -> list[str] | None:
    """The depth-camera artifacts this episode lacks — None if nothing checked."""
    if check is None:
        return None
    return [name for name in check.get("missing") or []
            if name.startswith(_DEPTH_PREFIXES)]


def _name_list(names: list[str], limit: int = 2) -> str:
    """Filenames, named while there are few enough to be worth reading.

    A depth camera that wrote nothing is missing every one of its artifacts, and
    seven filenames in a row say no more than one of them does — the finding is
    "the camera wrote nothing", not which file to go and look for.
    """
    shown = ", ".join(f"<code>{html.escape(n)}</code>" for n in names[:limit])
    rest = len(names) - limit
    return f"{shown} and {rest} more" if rest > 0 else shown


def recording_summary(episode: dict | None, check: dict | None) -> str:
    """Verdict for one finished test recording, as HTML.

    Not named test_* on purpose: pytest collects any module-level name starting
    with "test_" in a test module, including one that got there by import.

    Pure (no network) so the rules can be unit-tested: what counts as a usable
    episode is a judgement, and it is the part worth pinning down.

    `episode` is the EpisodeInfo from `GET /api/episodes/{id}`, `check` the
    reply from its `/check` route. Either may be None — that is "could not
    ask", never "the episode is bad".
    """
    if episode is None:
        return _card(_UNKNOWN, "Could not read the episode",
                     '<div style="opacity:.8;font-size:.88rem;margin-top:.3rem;">'
                     "Nothing is wrong with the recording as far as we know — "
                     "the device just did not answer.</div>")

    episode_id = episode.get("episode_id") or "?"
    duration = float(episode.get("duration_seconds") or 0.0)
    frames = int(episode.get("frame_count") or 0)
    angles = int(episode.get("angle_sample_count") or 0)
    metadata_ok = bool(episode.get("metadata_ok", True))
    has_video = bool(episode.get("has_video", True))

    depth_missing = _depth_missing(check)

    # Faults make the episode unusable; warnings leave it usable but worth
    # knowing about. Only a fault changes the headline.
    faults, warnings = [], []
    if not has_video or frames == 0:
        faults.append("The RGB camera recorded nothing.")
    if depth_missing:
        faults.append(
            "The depth camera recorded no RGB-D data — "
            f"{_name_list(depth_missing)} missing. "
            "This episode cannot be converted."
        )
    if angles == 0:
        faults.append("The gripper angle sensors recorded nothing.")
    if not metadata_ok:
        faults.append("<code>metadata.json</code> is missing or unreadable, "
                      "so the counters above are not facts.")

    if duration < _TEST_MIN_SECONDS:
        warnings.append(f"Only {duration:.1f}s long — record a few seconds "
                        "to test properly.")

    depth_state = "unknown" if depth_missing is None else (
        "bad" if depth_missing else "ok")
    chips = "".join([
        _chip("bad" if (frames == 0 or not has_video) else "ok",
              "RGB", f"{frames} frames"),
        _chip(depth_state, "RGB-D",
              {"ok": "recorded", "bad": "missing"}.get(depth_state, "not checked")),
        _chip("bad" if angles == 0 else "ok", "Angles", f"{angles}"),
    ])

    color = _BAD if faults else _OK
    head = "Something is wrong" if faults else "Recording looks good"
    body = [
        '<div style="opacity:.7;font-size:.82rem;margin:.15rem 0 .7rem;">'
        f"<code>{html.escape(episode_id)}</code> · {duration:.1f}s</div>",
        '<div style="display:flex;flex-wrap:wrap;gap:.4rem;">' + chips + "</div>",
    ]
    for f in faults:
        body.append('<div style="margin-top:.6rem;font-size:.88rem;">'
                    f'<span style="color:{_BAD};">✗</span> {f}</div>')
    for w in warnings:
        body.append('<div style="margin-top:.6rem;font-size:.88rem;opacity:.85;">'
                    f"⚠ {html.escape(w)}</div>")
    return _card(color, ("✗ " if faults else "✓ ") + head, "".join(body))
