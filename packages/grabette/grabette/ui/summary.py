"""Verdicts for a finished test recording.

Its own module, free of gradio, so the rules can be imported and tested
anywhere — `grabette.ui.app` needs the `ui` extra, which CI does not install,
and a rule nobody runs is a rule nobody checks.
"""

from __future__ import annotations


# A test recording shorter than this is almost certainly a mis-click rather
# than a recording, and reports counters too small to mean anything.
_TEST_MIN_SECONDS = 2.0


def recording_summary(
    status: dict, episode: dict | None, depth_on: bool
) -> str:
    """Verdict lines for a finished test recording, as Markdown.

    Not named test_* on purpose: pytest collects any module-level name starting
    with "test_" in a test module, including one that got there by import.

    Pure (no network) so the rules can be unit-tested: what counts as a usable
    episode is a judgement, and it is the part worth pinning down.

    `status` is the CaptureStatus from POST /api/episodes/stop; `episode` is the
    EpisodeInfo read back afterwards, or None when it could not be fetched —
    which is "could not check", never "the episode is bad".
    """
    if status.get("error"):
        return f"### ✗ Recording failed\n\n{status['error']}"

    episode_id = status.get("episode_id")
    if not episode_id:
        return "### ✗ No episode was written\n\nNothing was saved to disk."

    duration = float(status.get("duration_seconds") or 0.0)
    frames = int(status.get("frame_count") or 0)
    angles = int(status.get("angle_sample_count") or 0)
    imu = int(status.get("imu_sample_count") or 0)

    # Faults make the episode unusable; warnings leave it usable but worth
    # knowing about. Only a fault changes the headline.
    faults, warnings = [], []
    if frames == 0:
        faults.append("**No camera frames.** The RGB camera recorded nothing.")
    if angles == 0:
        faults.append("**No angle samples.** The finger encoders recorded nothing.")
    if episode is not None and not episode.get("metadata_ok", True):
        faults.append(
            "**`metadata.json` is missing or unreadable**, so the counters above "
            "are not trustworthy."
        )
    if episode is not None and not episode.get("has_video", True):
        faults.append("**No video file** was written for this episode.")

    if duration < _TEST_MIN_SECONDS:
        warnings.append(
            f"Only {duration:.1f}s long — record a few seconds to test properly."
        )
    if not depth_on:
        warnings.append(
            "The depth camera was off. Fine for this test, but SLAM needs it — "
            "turn it on in Live View before recording for real."
        )

    head = "### ✗ Something is wrong" if faults else "### ✓ Recording looks good"
    lines = [
        head,
        "",
        f"**Episode** `{episode_id}` — {duration:.1f}s",
        "",
        f"RGB **{frames}** frames · angles **{angles}** · IMU **{imu}**",
    ]
    for f in faults:
        lines += ["", f"✗ {f}"]
    for w in warnings:
        lines += ["", f"⚠ {w}"]
    return "\n".join(lines)
