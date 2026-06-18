"""Pure rendering helpers for the Grabette Space UI — HTML/Markdown cards, the
progress bar, and the button/field visibility updates.

Everything here is stateless: given inputs, it returns HTML strings or gr.update
tuples. No cooperative flags, no I/O — so it has no dependency on the run
controller or the review panel (they import from here, not the reverse).
"""

import html
import re

import gradio as gr

VISUALIZER = "https://huggingface.co/spaces/lerobot/visualize_dataset"


def visualizer_url(repo_id: str) -> str:
    return f"{VISUALIZER}?dataset={repo_id}&episode=0"


def bar(frac: float, label: str) -> str:
    """An HTML progress bar updated via yield (coexists with the log textbox —
    unlike gr.Progress, which overlays the output and makes the UI flip).

    The fill sits behind a centered, always-visible label (readable at 0% too).
    """
    pct = max(0, min(100, int(frac * 100)))
    return (
        '<div style="position:relative;height:28px;width:100%;border-radius:8px;'
        'overflow:hidden;background:var(--neutral-200,#e5e7eb);'
        'font-family:var(--font,sans-serif)">'
        f'<div style="position:absolute;inset:0 auto 0 0;width:{pct}%;'
        'background:linear-gradient(90deg,#34d399,#10b981);transition:width .35s ease"></div>'
        '<div style="position:absolute;inset:0;display:flex;align-items:center;'
        'justify-content:center;font-size:13px;font-weight:600;color:#0f172a">'
        f'{pct}% · {label}</div></div>'
    )


def error_card(msg: str) -> str:
    """A red-background error card for the summary slot — shown instead of raising
    gr.Error, which would stamp an "Error" badge on every output component.

    The card lives in a gr.Markdown, so newlines in `msg` are turned into <br>:
    a literal blank line would otherwise terminate the inline-HTML block (a
    CommonMark rule) and the text after it would escape the styled div. html.escape
    keeps arbitrary exception/repo text from breaking the markup."""
    safe = html.escape(msg).replace("\n", "<br>")
    return (
        '<div style="background:#fee2e2;border:1px solid #fca5a5;border-radius:8px;'
        'padding:12px 16px;color:#991b1b;font-weight:500;'
        'font-family:var(--font,sans-serif)">'
        f'{safe}</div>'
    )


def selection_card(source="", target="", task="") -> str:
    """Small read-only recap of the user's inputs, shown in place of the editable
    fields while a run is in progress."""
    rows = []
    if source:
        rows.append(f"Source: <code>{html.escape(source)}</code>")
    if target:
        rows.append(f"Target: <code>{html.escape(target)}</code>")
    if task:
        rows.append(f"Task: {html.escape(task)}")
    return (
        '<div style="font-size:12px;color:var(--body-text-color-subdued);'
        'border:1px solid var(--border-color-primary,#e5e7eb);border-radius:8px;'
        'padding:8px 10px;line-height:1.6">'
        '<b style="font-size:12px">Selected</b><br>' + "<br>".join(rows) + '</div>'
    )


def run_recap(logs: list[str]) -> str:
    """A compact per-episode recap pulled from the captured log lines, appended to
    the final summary (since the live log is hidden once a run finishes).

    Keeps the lines that matter for assessing the run: SLAM tracking %, trajectory
    quality verdicts (GOOD/WARN/BAD/FAIL), input-check flags, and failures.
    Returns "" when there's nothing worth recapping (e.g. a push-only retry).
    """
    keep = []
    for ln in logs:
        s = ln.strip()
        if (s.startswith("✓") and "tracking" in s) \
                or s.startswith("✗") \
                or s.startswith("•") \
                or re.match(r"^\[(GOOD|WARN|BAD|FAIL|input/)", s):
            keep.append(s)
    if not keep:
        return ""
    return "\n\n**Run recap**\n```\n" + "\n".join(keep) + "\n```"


def success_summary(target_repo: str, n: int, link: str | None, mode: str) -> str:
    """Final success card — same for a first-try push and a retried push."""
    ds_url = f"https://huggingface.co/datasets/{target_repo}"
    if mode == "branch":
        # A "Sign in with HF" OAuth token can push commits (so the branch works) but
        # is refused by the PR-creation endpoint — and HF has no "open a PR from this
        # branch" URL (a PR is its own refs/pr/N ref, created from the Community tab
        # or the API). So we link the branch + the New-PR page; opening the PR itself
        # is a manual step the user's own browser session is allowed to do.
        branch = link.rsplit("/tree/", 1)[-1] if link and "/tree/" in link else "the branch"
        new_pr_url = f"{ds_url}/discussions/new"
        return (
            f"### ✅ Done — {n} episode(s) — pushed to a branch\n"
            f"_(A “Sign in with HF” token can’t open a pull request — that needs the "
            f"discussions/PR permission — so the result landed on a branch, leaving "
            f"`main` untouched.)_\n"
            f"- **Branch:** 👉 [`{branch}`]({link}) 👈 — review the dataset here.\n"
            f"- **Open a PR:** [New pull request / discussion]({new_pr_url}) "
            f"(your browser session can; the Space token can’t), then point it at "
            f"`{branch}` — or merge the branch into `main` via git when ready.\n"
        )
    viz_url = visualizer_url(target_repo)
    # The LeRobot visualizer sends X-Frame-Options: deny, so link out.
    return (
        f"### ✅ Done — {n} episode(s)\n"
        f"- **Dataset:** [{target_repo}]({ds_url})\n"
        f"- **Visualize:** [open in LeRobot visualizer]({viz_url})\n\n"
        f"_(The visualizer needs the dataset to be public.)_"
    )


def btns(state, *, paused=False, cancel_pending=False,
         branch=False, retry=False, allow_pause=True):
    """Updates for the 8 buttons, in `outputs` order:
    (branch, retry, run, pause, cancel, confirm_cancel, keep, reset).

    state: 'idle' (only Run, enabled), 'running' (Run greyed 'Running' + Pause +
    Cancel, or the confirm/keep pair when a cancel is pending), or 'finished'
    (only Reset). retry reveals the retry button. branch=True is the "target already
    exists" case: Run is replaced by the "Push to a new branch" button (primary) and
    Reset is offered so the user can edit the target instead.
    """
    H = gr.update(visible=False)
    branch_u = gr.update(visible=branch)
    retry_u = gr.update(visible=retry)
    if state == "running":
        run_u = gr.update(visible=True, interactive=False, value="●  Running")
        if cancel_pending:
            pause_u = H
            cancel_u = H
            confirm_u = gr.update(visible=True)
            keep_u = gr.update(visible=True)
        else:
            pause_u = (gr.update(visible=True, value=("▶  Resume" if paused else "⏸  Pause"))
                       if allow_pause else H)
            cancel_u = gr.update(visible=True)
            confirm_u = H
            keep_u = H
        reset_u = H
    elif state == "finished":
        run_u = pause_u = cancel_u = confirm_u = keep_u = H
        reset_u = gr.update(visible=True)
    else:  # idle
        if branch:
            # Target exists: hide Run (it would just re-trigger this warning); the
            # branch button (branch_u) is the primary action, and Reset lets the
            # user go back to edit the target.
            run_u = H
            reset_u = gr.update(visible=True)
        else:
            run_u = gr.update(visible=True, interactive=True, value="▶  Run")
            reset_u = H
        pause_u = cancel_u = confirm_u = keep_u = H
    return (branch_u, retry_u, run_u, pause_u, cancel_u, confirm_u, keep_u, reset_u)


def io(state, bar_html, log_text):
    """Updates for the progress bar + log box: shown only while running/paused
    (state == 'running'), hidden when idle or finished. Returns 2 updates in
    `outputs` order (bar, log)."""
    show = state == "running"
    return (gr.update(value=bar_html, visible=show),
            gr.update(value=log_text, visible=show))


def inputs_view(state, *, source="", target="", task=""):
    """Updates for [source, target, task, selection_card].

    Once a run starts and until it's reset (state 'running' or 'finished') the
    editable fields are hidden and replaced by a small read-only recap card. Only
    the idle page (and the recoverable preflight/target-exists states) shows the
    editable fields; there the card is hidden."""
    hide = state in ("running", "finished")
    fld = gr.update(visible=not hide)
    if hide:
        card = gr.update(visible=True,
                         value=selection_card(source, target, task))
    else:
        card = gr.update(visible=False)
    return (fld, fld, fld, card)


# ---- Episode-review cards -------------------------------------------------

_VERDICT_BADGE = {
    "FAIL": ("#7f1d1d", "#fecaca"),
    "ERROR": ("#7f1d1d", "#fecaca"),
    "BAD": ("#991b1b", "#fee2e2"),
    "WARN": ("#92400e", "#fef3c7"),
}


def episode_issues(it: dict) -> list[tuple[str, str]]:
    """The issue(s) to show for one flagged episode, each as (badge, memo).

    For the pre-SLAM gate (kind "input") the items already carry explicit
    completeness/sync messages, so they're shown verbatim. For the trajectory gate
    only the metric(s) that actually fired in check_trajectory are shown (not the
    full dist/step/jumps dump), each with a one-liner on what the anomaly means;
    falls back to the report's own error/warning text if (somehow) no check
    re-fires, so a flagged card is never left unexplained.
    """
    if it.get("kind") == "input":
        return [(msg, "") for msg in it.get("messages", [])]

    out: list[tuple[str, str]] = []
    if it["n_tracked"] < 2:
        out.append((f"{it['n_tracked']} tracked frame(s)",
                    "SLAM tracked almost nothing — unusable trajectory."))
        return out
    avg_speed = it["total_distance_m"] / max(it["duration_s"], 0.1)
    if avg_speed > 2.0:
        out.append((f"speed {avg_speed:.1f} m/s",
                    "Unrealistic average speed for a gripper — usually drift."))
    if it["median_step_mm"] > 15 and it["median_angle_deg"] < 5:
        out.append((f"drift · median step {it['median_step_mm']:.0f} mm",
                    "Suspiciously straight, steady motion — likely IMU drift."))
    if it["n_jumps"] > 5 and it["median_angle_deg"] > 90:
        out.append((f"zigzag · {it['n_jumps']} jumps",
                    "Jumps with direction reversals — failed relocalizations."))
    elif it["n_jumps"] > it["n_tracked"] * 0.1:
        out.append((f"{it['n_jumps']} jumps > 50 mm",
                    "Abrupt position jumps — unstable tracking."))
    if it["tracking_pct"] < 50:
        out.append((f"tracking {it['tracking_pct']:.0f}%",
                    "SLAM lost the camera over much of the episode."))
    if not out:
        for msg in it.get("messages", []):
            out.append((msg, ""))
    return out


def episode_card(it: dict) -> str:
    """Orange flagged-episode card: name + verdict badge + only the abnormal
    metric(s), each with a concise memo."""
    fg, bg = _VERDICT_BADGE.get(it["verdict"], ("#92400e", "#fef3c7"))
    rows = "".join(
        f'<div style="margin-top:5px;font-size:12.5px;line-height:1.35">'
        f'<b style="color:#9a3412">{html.escape(badge)}</b>'
        + (f' — <span style="color:var(--body-text-color-subdued)">{html.escape(memo)}</span>'
           if memo else "")
        + '</div>'
        for badge, memo in episode_issues(it))
    return (
        '<div style="background:#fff7ed;border:1px solid #fdba74;'
        'border-left:4px solid #f97316;border-radius:8px;padding:9px 12px;'
        'font-family:var(--font,sans-serif)">'
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<b style="font-size:13.5px">{html.escape(it["name"])}</b>'
        f'<span style="font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;'
        f'color:{fg};background:{bg}">{html.escape(it["verdict"])}</span></div>'
        f'{rows}</div>'
    )


THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important;}
#app-header {text-align:center; margin: 2px 0 6px;}
#app-header h1 {font-size: 1.45rem; margin: 0;}
#app-header p {color: var(--body-text-color-subdued); margin: 2px 0 0; font-size: .9rem;}
#logbox textarea {font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  font-size: 12px; line-height: 1.35;}
footer {display: none !important;}
/* Episode-review actions: orange to match the flagged-episode cards (warning),
   not the default emerald primary. */
.btn-orange button {background: #f97316 !important; border: none !important;
                    color: #fff !important;}
.btn-orange button:hover {background: #ea580c !important;}
/* The 🗑 remove-episode button: solid red (the Soft theme renders variant="stop"
   as a pale/outline style, so force it). */
.btn-red button {background: #dc2626 !important; border: none !important;
                 color: #fff !important;}
.btn-red button:hover {background: #b91c1c !important;}
"""
