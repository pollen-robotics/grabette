"""Grabette SLAM → LeRobot pipeline — HuggingFace Space (Gradio + HF OAuth).

The user signs in with their HF account; the OAuth token is used to download the
source dataset and push the generated LeRobot dataset under their account.
Gradio auto-injects the gr.OAuthToken parameter — it is NOT a UI input.
"""

import contextlib
import html
import os
import queue
import re
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from pipeline import build_lerobot, push_lerobot

VISUALIZER = "https://huggingface.co/spaces/lerobot/visualize_dataset"

# Cooperative control flags, shared between the worker and the control buttons.
# The Space processes one run at a time, so module-level flags are enough.
#   _stop           — abandon the run (Cancel confirmed)
#   _pause          — hold the worker at the next safe checkpoint (Pause)
#   _cancel_pending — Cancel was clicked, awaiting confirmation
# The streaming generator is the ONLY writer of button state: it re-renders the
# buttons from these flags on every tick. The Pause/Cancel/Keep handlers just
# flip a flag (outputs=None, like the old Stop), so they never fight the
# generator for the same components.
_stop = threading.Event()
_pause = threading.Event()
_cancel_pending = threading.Event()

# Episode review (drop flagged trajectories before building). When SLAM flags at
# least one non-GOOD episode, the worker publishes them and blocks on _review_done
# while _review is set; the streaming generator reveals the checkbox panel, and the
# Continue button records the user's choice (_review_drop) and releases the worker.
_review = threading.Event()        # worker is awaiting the user's review decision
_review_done = threading.Event()   # Continue clicked → release the worker
_review_items: list[dict] = []     # flagged episodes published for the panel
_review_drop: list[str] = []       # episode names the user chose to drop


def _toggle_pause():
    """Pause ⇄ Resume: flip the cooperative pause flag (button relabels itself on
    the generator's next tick)."""
    _pause.clear() if _pause.is_set() else _pause.set()


def _request_cancel():
    """Cancel clicked: reveal the confirm/keep buttons (handled by the generator)."""
    _cancel_pending.set()


def _keep_running():
    """'Keep running' clicked: dismiss the cancel confirmation."""
    _cancel_pending.clear()


def _drop_episode(name: str) -> dict:
    """🗑 clicked on a flagged episode: add it to the drop set and return the fresh
    review state so the @gr.render panel re-runs and the card disappears."""
    if name not in _review_drop:
        _review_drop.append(name)
    return _review_state()


def _restore_episodes() -> dict:
    """'Restore all' clicked: clear the drop set (un-hide every flagged card)."""
    _review_drop.clear()
    return _review_state()


def _submit_review():
    """'Continue' clicked during episode review: release the worker. The episodes
    to drop already live in _review_drop (set as the user clicked 🗑). outputs=None
    — like Pause/Cancel, this only flips a flag; the worker clears _review and the
    panel closes on the next poll."""
    _review_done.set()


def _review_state() -> dict:
    """Snapshot of the review panel state for the reactive @gr.render, built from
    the worker-published flags. `open` drives whether the panel shows at all."""
    return {"open": _review.is_set(),
            "items": list(_review_items),
            "dropped": list(_review_drop)}


def _poll_review(cur: dict):
    """gr.Timer tick: mirror the worker's review flags into the review State so the
    reactive panel opens/closes and its drop set stays in sync. Returns gr.skip()
    when nothing changed, so an idle session never churns the @gr.render block."""
    nxt = _review_state()
    if nxt == cur:
        return gr.skip()
    return nxt


def _wait_if_paused():
    """Block while paused (used as the worker's checkpoint gate). Returns at once
    if a stop/cancel is requested, so pausing then cancelling never deadlocks."""
    while _pause.is_set() and not _stop.is_set():
        time.sleep(0.2)


def _btns(state, *, paused=False, cancel_pending=False,
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


def _io(state, bar_html, log_text):
    """Updates for the progress bar + log box: shown only while running/paused
    (state == 'running'), hidden when idle or finished. Returns 2 updates in
    `outputs` order (bar, log)."""
    show = state == "running"
    return (gr.update(value=bar_html, visible=show),
            gr.update(value=log_text, visible=show))


def _selection_card(source="", target="", task="", include_right=None) -> str:
    """Small read-only recap of the user's inputs, shown in place of the editable
    fields while a run is in progress."""
    rows = []
    if source:
        rows.append(f"Source: <code>{html.escape(source)}</code>")
    if target:
        rows.append(f"Target: <code>{html.escape(target)}</code>")
    if task:
        rows.append(f"Task: {html.escape(task)}")
    if include_right is not None:
        rows.append("Right camera: " + ("on" if include_right else "off"))
    return (
        '<div style="font-size:12px;color:var(--body-text-color-subdued);'
        'border:1px solid var(--border-color-primary,#e5e7eb);border-radius:8px;'
        'padding:8px 10px;line-height:1.6">'
        '<b style="font-size:12px">Selected</b><br>' + "<br>".join(rows) + '</div>'
    )


def _inputs_view(state, *, source="", target="", task="", include_right=None):
    """Updates for [source, target, task, include_right, selection_card].

    Once a run starts and until it's reset (state 'running' or 'finished') the
    editable fields are hidden and replaced by a small read-only recap card. Only
    the idle page (and the recoverable preflight/target-exists states) shows the
    editable fields; there the card is hidden."""
    hide = state in ("running", "finished")
    fld = gr.update(visible=not hide)
    if hide:
        card = gr.update(visible=True,
                         value=_selection_card(source, target, task, include_right))
    else:
        card = gr.update(visible=False)
    return (fld, fld, fld, fld, card)


def _issues(it: dict) -> list[tuple[str, str]]:
    """The trajectory metric(s) that are actually abnormal for one flagged episode,
    each as (badge, memo). Mirrors the checks in analyze_trajectory so only the
    parameter that fired is shown — not the full dist/step/jumps dump. memo is a
    one-liner on what the anomaly means. Falls back to the report's own
    error/warning text if (somehow) no check re-fires, so a flagged card is never
    left unexplained.
    """
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


_VERDICT_BADGE = {
    "FAIL": ("#7f1d1d", "#fecaca"),
    "BAD": ("#991b1b", "#fee2e2"),
    "WARN": ("#92400e", "#fef3c7"),
}


def _episode_card(it: dict) -> str:
    """Orange flagged-episode card: name + verdict badge + only the abnormal
    metric(s), each with a concise memo."""
    fg, bg = _VERDICT_BADGE.get(it["verdict"], ("#92400e", "#fef3c7"))
    rows = "".join(
        f'<div style="margin-top:5px;font-size:12.5px;line-height:1.35">'
        f'<b style="color:#9a3412">{html.escape(badge)}</b>'
        + (f' — <span style="color:var(--body-text-color-subdued)">{html.escape(memo)}</span>'
           if memo else "")
        + '</div>'
        for badge, memo in _issues(it))
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


def _reset_view():
    """Full reset to the idle page (clears log/summary/bar, only Run shown).
    Returns the 12-output tuple.

    Clears the UI-facing flags (pause, cancel-pending) but NOT _stop: when this
    runs as the .then() after a confirmed Cancel, the background worker is still
    winding down and relies on _stop staying set to abandon (skip the push).
    _run clears _stop itself at the start of the next run.
    """
    _pause.clear()
    _cancel_pending.clear()
    _review.clear()
    _review_done.clear()
    _review_drop.clear()
    return (*_io("idle", _bar(0, "ready"), ""), "", None, *_btns("idle"),
            *_inputs_view("idle"))


def _visualizer_url(repo_id: str) -> str:
    return f"{VISUALIZER}?dataset={repo_id}&episode=0"


class _LineQueueWriter:
    """A file-like object that pushes complete lines onto a queue.

    Used to redirect stdout: every print() from convert / SLAM / build_dataset
    becomes a streamed log line, not just the explicit log() callback.
    """

    def __init__(self, q: "queue.Queue"):
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(("log", line))
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.q.put(("log", self._buf))
        self._buf = ""


def _bar(frac: float, label: str) -> str:
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


def _error_card(msg: str) -> str:
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


def _run_recap(logs: list[str]) -> str:
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


def _success_summary(target_repo: str, n: int, link: str | None, mode: str) -> str:
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
    viz_url = _visualizer_url(target_repo)
    # The LeRobot visualizer sends X-Frame-Options: deny, so link out.
    return (
        f"### ✅ Done — {n} episode(s)\n"
        f"- **Dataset:** [{target_repo}]({ds_url})\n"
        f"- **Visualize:** [open in LeRobot visualizer]({viz_url})\n\n"
        f"_(The visualizer needs the dataset to be public.)_"
    )


def _preflight(api, source_repo, target_repo):
    """Quick access/existence checks. Returns (exists, writable, error_or_None).

    error_or_None is a clear, user-facing message (never a raw HF traceback) when
    the source can't be read or the target can't be written — shown in the red
    error card before any heavy work starts. Both checks run up front so a bad
    source or an unwritable target fails fast, not after a long SLAM run.
    """
    # ---- Source: must exist and be readable with this token ----
    try:
        api.repo_info(source_repo, repo_type="dataset")
    except RepositoryNotFoundError:
        return None, None, (
            f"Error on the source dataset '{source_repo}':\nEither it doesn't exist, or it's private and "
            f"your account can't see it.\nCheck the spelling; it should look like "
            f"'username/dataset-name'."
        )
    except GatedRepoError:
        return None, None, (
            f"Error on the source dataset '{source_repo}':\nIt is gated. Accept its access terms on "
            f"the Hub first, then re-run."
        )
    except Exception as e:
        return None, None, f"Cannot access source dataset '{source_repo}': {e}"

    # ---- Target: resolve namespace + whether this account can write to it ----
    # whoami() returns each org with the user's role; only write-capable roles can
    # push datasets, so being *in* an org isn't enough — check the role too.
    WRITE_ROLES = {"admin", "write", "contributor"}
    try:
        me = api.whoami()
        username = me.get("name")
        org_roles = {o.get("name"): o.get("roleInGroup") for o in me.get("orgs", [])}
    except Exception:
        username, org_roles = None, {}
    ns = target_repo.split("/")[0] if "/" in target_repo else username

    if ns is None:
        writable = False
    elif ns == username:
        writable = True
    elif ns in org_roles:
        writable = org_roles[ns] in WRITE_ROLES
    else:
        writable = False

    exists = api.repo_exists(target_repo, repo_type="dataset")

    if not writable:
        if username is None:
            return exists, False, (
                "Couldn't confirm your Hugging Face identity from the sign-in "
                "token. Sign out and back in, then re-run."
            )
        verb = "push to" if exists else "create"
        if ns in org_roles:
            role = org_roles[ns] or "read-only"
            return exists, False, (
                f"Your role in the '{ns}' org is '{role}', which can't write "
                f"datasets : so you can't {verb} '{target_repo}'. \nAsk an org admin "
                f"for write access, or set the target to a namespace you own."
            )
        who = f"'{username}'" + (
            f" (orgs: {', '.join(sorted(org_roles))})" if org_roles else " (no orgs)")
        return exists, False, (
            f"You don't have write access to namespace '{ns}', so you can't "
            f"{verb} '{target_repo}'.\nYou're signed in as {who} : set the target "
            f"to a namespace you own."
        )

    return exists, writable, None


def _run(source_repo, target_repo, task, include_right, oauth_token, to_branch):
    """Streaming generator behind Run / branch. Yields the 12-element `outputs`
    tuple: [bar, log, summary, retry state, branch, retry, run, pause, cancel,
    confirm_cancel, keep, reset].

    A progress bar (gr.HTML, updated via yield — no gr.Progress overlay, so no
    UI flip) sits alongside the live log. The heavy work runs in a worker thread;
    its output (log() + every print()) streams through a queue. Pre-flight checks
    run synchronously up front: source read access, target write access, and
    target existence — an existing target reveals the "push to a branch" button.

    While running, the buttons reflect the _pause / _cancel_pending flags every
    tick: Run is greyed to 'Running', with Pause (⇄ Resume) and Cancel. The run
    ends 'idle' (Run back) for recoverable cases (bad inputs / target exists) or
    'finished' (only Reset) on done / failure.

    include_right: when False (default), oakd_right.mp4 is excluded from the
    download (SLAM never uses it — see the worker).
    """
    if oauth_token is None:
        raise gr.Error("Sign in with your Hugging Face account first.")
    # Trim stray whitespace from the textboxes — a leading/trailing space in a
    # repo id is an easy copy-paste mistake that would otherwise 404 the source.
    source_repo = (source_repo or "").strip()
    target_repo = (target_repo or "").strip()
    task = (task or "").strip()
    if not (source_repo and target_repo and task):
        raise gr.Error("Fill in source repo, target repo and task.")

    # OAuth access tokens aren't accepted by huggingface_hub.login() (it expects a
    # classic token's role). Expose the token via env + pass it explicitly.
    token = oauth_token.token
    os.environ["HF_TOKEN"] = token

    _stop.clear()
    _pause.clear()
    _cancel_pending.clear()  # fresh run
    _review.clear()
    _review_done.clear()

    logs: list[str] = []
    start = time.time()
    frac, label = 0.02, "checking"
    retry_ctx = None  # set when a push fails but the built dataset is reusable

    def render() -> str:
        return "\n".join(logs)

    def view(summary="", *, state="running", branch=False, retry=False):
        # 17 outputs: bar, log, summary, retry state, the 8 buttons, the 5
        # inputs-view fields. The episode-review panel is a separate reactive
        # @gr.render block driven by a gr.Timer — not part of this tuple — so it
        # reads the worker's _review flags directly (see _poll_review).
        in_review = _review.is_set()
        lbl = ("awaiting review" if in_review
               else f"paused · {label}" if _pause.is_set()
               else f"{label} · {int(time.time() - start)}s")
        return (*_io(state, _bar(frac, lbl), render()), summary, retry_ctx,
                *_btns(state, paused=_pause.is_set(),
                       cancel_pending=_cancel_pending.is_set(),
                       branch=branch, retry=retry,
                       allow_pause=not in_review),
                *_inputs_view(state, source=source_repo, target=target_repo,
                              task=task, include_right=include_right))

    # ---- Pre-flight (synchronous, fast) ---------------------------------
    logs.append("Checking repo access…")
    yield view()  # 'Running' + Pause + Cancel appear immediately
    api = HfApi(token=token)
    exists, _writable, err = _preflight(api, source_repo, target_repo)
    if err:
        logs.append(err)
        # Recoverable: back to idle so the user can fix the inputs and re-run.
        frac, label = 0.0, "error"
        yield view(summary=_error_card(err), state="idle")
        return
    logs.append("  ✓ source readable")
    logs.append(f"  ✓ write access to '{target_repo.split('/')[0]}'")

    if not exists:
        logs.append(f"Target '{target_repo}' is new and writable")
    elif not to_branch:
        logs.append(f"Target '{target_repo}' already exists")
        warn = (
            f"### ⚠️ Target dataset already exists\n"
            f"`{target_repo}` already exists on the Hub.\nTo avoid overwriting it, "
            f"click **Push to a new branch** below: your result lands on a "
            f"`grabette-…` branch and `main` is left untouched.\n\n"
            f"_(Want a different target instead? Click **Reset** and change it.)_"
        )
        frac, label = 0.05, "target exists"
        # Recoverable: back to idle (re-run after editing the target) but reveal
        # the "push to a branch" button.
        yield view(summary=warn, state="idle", branch=True)
        return
    else:
        logs.append(f"  ✓ target '{target_repo}' exists — will push to a branch")
    yield view()

    # ---- Heavy work in a worker thread ----------------------------------
    q: "queue.Queue[tuple]" = queue.Queue()
    result: dict = {}

    def on_progress(done, total, phase):
        if phase == "slam":
            f = 0.30 + 0.55 * (done / total if total else 1.0)
            lbl = f"SLAM {done}/{total}" if total else "SLAM"
        elif phase == "build":
            f, lbl = 0.88, "building dataset"
        elif phase == "push":
            f, lbl = 0.95, ("pushing branch" if to_branch else "pushing")
        else:
            f, lbl = 0.30, phase
        q.put(("progress", f, lbl))

    def review_cb(results):
        """Called by build_lerobot between SLAM and the build. results is a list of
        (episode_dir, TrajectoryReport). Returns the episode dirs to keep.

        Every-GOOD run continues untouched. Otherwise the flagged (non-GOOD)
        episodes are published for the panel and the worker blocks until Continue
        (or a cancel) before dropping the ones the user checked."""
        flagged = [(ep, rep) for ep, rep in results if rep.verdict != "GOOD"]
        if not flagged:
            return [ep for ep, _ in results]
        _review_items[:] = [
            {"name": ep.name, "verdict": rep.verdict,
             "n_tracked": rep.n_tracked, "tracking_pct": rep.tracking_pct,
             "total_distance_m": rep.total_distance_m, "duration_s": rep.duration_s,
             "median_step_mm": rep.median_step_mm, "median_angle_deg": rep.median_angle_deg,
             "n_jumps": rep.n_jumps,
             "messages": [*rep.errors, *rep.warnings]}
            for ep, rep in flagged
        ]
        _review_drop.clear()
        _review_done.clear()
        _review.set()
        print(f"⏸ Review: {len(flagged)} of {len(results)} episode(s) flagged "
              f"— remove the ones to drop (🗑), then click Continue.")
        while not _review_done.is_set():
            if _stop.is_set():
                _review.clear()
                return [ep for ep, _ in results]  # build hits should_stop and raises
            time.sleep(0.2)
        _review.clear()
        drop = set(_review_drop)
        kept = [ep for ep, _ in results if ep.name not in drop]
        if drop:
            print(f"Dropped {len(drop)} episode(s): {', '.join(sorted(drop))}.")
        print(f"Keeping {len(kept)} episode(s).")
        return kept

    def worker():
        writer = _LineQueueWriter(q)
        try:
            with contextlib.redirect_stdout(writer):
                q.put(("progress", 0.07, "download"))
                work = Path(tempfile.mkdtemp())
                # The right OAK camera (oakd_right.mp4) is recorded but never
                # consumed: SLAM runs RGB-D on the LEFT image + depth, and the
                # dataset uses the Arducam raw_video. Skipping it cuts a whole
                # mono H.264 stream per episode out of the download for free.
                ignore = [] if include_right else ["*oakd_right.mp4", "*oakd_right_timestamps.json"]
                try:
                    info = api.repo_info(source_repo, repo_type="dataset", files_metadata=True)
                    skip = [] if include_right else [
                        s for s in info.siblings
                        if s.rfilename.endswith(("oakd_right.mp4", "oakd_right_timestamps.json"))]
                    mb = sum((s.size or 0) for s in info.siblings) / 1e6
                    mb_skip = sum((s.size or 0) for s in skip) / 1e6
                    print(f"Downloading {source_repo} — {len(info.siblings) - len(skip)} files, "
                          f"{mb - mb_skip:.0f} MB"
                          + (f" (skipping {mb_skip:.0f} MB of unused right-camera video)"
                             if mb_skip else "") + "…")
                except Exception:
                    print(f"Downloading {source_repo}…")
                # Raw episodes are dominated by many small depth PNGs (≈600/ep),
                # so the download is bound by per-file request overhead, not
                # bandwidth — more concurrent connections is the main lever.
                raw = snapshot_download(source_repo, repo_type="dataset",
                                        local_dir=work / "raw", token=token,
                                        ignore_patterns=ignore,
                                        max_workers=48)
                print("Download complete.\n\n")
                q.put(("progress", 0.30, "converting"))
                ds_root = work / "lerobot"
                processed = build_lerobot(
                    raw, target_repo, task, root=ds_root,
                    log=print, should_stop=_stop.is_set, to_branch=to_branch,
                    on_progress=on_progress, token=token, gate=_wait_if_paused,
                    review=review_cb)
                # Build done — dataset cached on disk; record it so that if the push
                # fails, the "Retry push" button can reuse it (no re-running SLAM).
                result["built"] = {"root": str(ds_root), "n": len(processed)}
                n, link, mode = push_lerobot(
                    target_repo, task, ds_root, len(processed),
                    to_branch=to_branch, token=token, log=print,
                    on_progress=on_progress, gate=_wait_if_paused)
                writer.flush()
            result["n"], result["link"], result["mode"] = n, link, mode
        except Exception as e:
            writer.flush()
            result["error"] = e
        finally:
            q.put(("done",))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        try:
            # Short timeout so Pause/Cancel button changes reflect within a tick.
            item = q.get(timeout=0.3)
        except queue.Empty:
            yield view()  # tick the elapsed counter / reflect pause+cancel flags
            continue
        tag = item[0]
        if tag == "done":
            break
        if tag == "progress":
            frac, label = item[1], item[2]
        else:  # ("log", line)
            logs.append(item[1])
        yield view()

    t.join()
    if "error" in result:
        err = result["error"]
        if result.get("built"):
            # Build succeeded, push didn't — offer a push-only retry, no re-SLAM.
            retry_ctx = {"target_repo": target_repo, "task": task, "to_branch": to_branch,
                         "root": result["built"]["root"], "n": result["built"]["n"]}
            msg = _error_card(
                f"Push failed: {err}\n\nThe dataset is built and cached — click "
                f"“Retry push” to push it again without re-running SLAM.")
            label = "push failed"
            yield view(summary=msg + _run_recap(logs), state="finished", retry=True)
        else:
            label = "failed"
            yield view(summary=_error_card(f"Pipeline failed: {err}") + _run_recap(logs),
                       state="finished")
        return

    n = result["n"]
    link = result.get("link")
    mode = result.get("mode")
    logs.append(f"✅ Done — {n} episode(s).")
    frac, label = 1.0, "done"
    yield view(summary=_success_summary(target_repo, n, link, mode) + _run_recap(logs),
               state="finished")


def retry_push(retry_ctx, oauth_token: gr.OAuthToken | None = None):
    """Retry just the push of an already-built dataset — no re-download, no SLAM.
    Reuses the on-disk dataset captured in retry_ctx by a previous failed run."""
    if oauth_token is None:
        raise gr.Error("Sign in with your Hugging Face account first.")
    if not retry_ctx:
        raise gr.Error("Nothing to retry — run the pipeline first.")
    token = oauth_token.token
    os.environ["HF_TOKEN"] = token

    _stop.clear()
    _pause.clear()
    _cancel_pending.clear()
    _review.clear()
    _review_done.clear()

    logs: list[str] = []
    start = time.time()
    frac, label = 0.95, "pushing"

    def render() -> str:
        return "\n".join(logs)

    def view(summary="", *, state="running", retry=False, ctx=retry_ctx):
        # 17-output shape, same as _run. A push can't be paused, so allow_pause
        # is False; Cancel still works (abandons the retry).
        return (*_io(state, _bar(frac, f"{label} · {int(time.time() - start)}s"), render()),
                summary, ctx,
                *_btns(state, cancel_pending=_cancel_pending.is_set(),
                       retry=retry, allow_pause=False),
                *_inputs_view(state, target=retry_ctx["target_repo"],
                              task=retry_ctx["task"]))

    logs.append("Retrying push (dataset already built — skipping SLAM)…")
    yield view()  # 'Running' + Cancel

    q: "queue.Queue[tuple]" = queue.Queue()
    result: dict = {}

    def on_progress(done, total, phase):
        q.put(("progress", 0.95, "pushing branch" if retry_ctx["to_branch"] else "pushing"))

    def worker():
        writer = _LineQueueWriter(q)
        try:
            with contextlib.redirect_stdout(writer):
                n, link, mode = push_lerobot(
                    retry_ctx["target_repo"], retry_ctx["task"], Path(retry_ctx["root"]),
                    retry_ctx["n"], to_branch=retry_ctx["to_branch"], token=token,
                    log=print, on_progress=on_progress)
                writer.flush()
            result["n"], result["link"], result["mode"] = n, link, mode
        except Exception as e:
            writer.flush()
            result["error"] = e
        finally:
            q.put(("done",))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        try:
            item = q.get(timeout=0.3)
        except queue.Empty:
            yield view()
            continue
        if item[0] == "done":
            break
        if item[0] == "progress":
            frac, label = item[1], item[2]
        else:
            logs.append(item[1])
        yield view()
    t.join()

    if "error" in result:
        err = result["error"]
        # Keep the dataset cached: finished state with Reset, plus Retry to try
        # the push again.
        label = "push failed"
        yield view(summary=_error_card(f"Push failed again: {err}"),
                   state="finished", retry=True, ctx=retry_ctx)
        return

    n = result["n"]
    link = result.get("link")
    mode = result.get("mode")
    logs.append(f"✅ Pushed — {n} episode(s).")
    frac, label = 1.0, "done"
    yield view(summary=_success_summary(retry_ctx["target_repo"], n, link, mode),
               state="finished", ctx=None)


def run_pipeline(source_repo, target_repo, task, include_right,
                 oauth_token: gr.OAuthToken | None = None):
    yield from _run(source_repo, target_repo, task, include_right, oauth_token, to_branch=False)


def run_pipeline_branch(source_repo, target_repo, task, include_right,
                        oauth_token: gr.OAuthToken | None = None):
    yield from _run(source_repo, target_repo, task, include_right, oauth_token, to_branch=True)


def reset():
    """Reset button (shown when finished): clear the page and return to idle
    (only Run). Inputs (source/target/task) are kept so the user can re-run."""
    return _reset_view()


def confirm_cancel():
    """Confirm-cancel handler: only signal the worker to abandon (outputs=None,
    like the old Stop, so it preempts the running generator via cancels=). The
    UI reset is done by reset() chained as .then() on this click — a single
    function that both cancels= and writes outputs can't reliably apply its
    outputs (it'd queue behind the generator it's cancelling)."""
    _stop.set()


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

with gr.Blocks(title="Grabette SLAM → LeRobot", fill_height=True) as demo:
    gr.HTML(
        "<div id='app-header'><h1>Grabette post-processing</h1>"
        "<p>Raw OAK-D recording → SLAM → LeRobot dataset on the Hub</p></div>"
    )
    with gr.Row():
        # ---- Inputs (left) ----
        with gr.Column(scale=2, min_width=280):
            gr.LoginButton(size="sm")
            source = gr.Textbox(label="Source dataset", info="raw OAK-D repo on the Hub",
                                placeholder="pollen-robotics/grabette-raw")
            target = gr.Textbox(label="Target dataset", info="repo to create or contribute to",
                                placeholder="your-account/grabette-demo")
            task = gr.Textbox(label="Task description", placeholder="cup manipulation")
            # The right OAK camera is never used by SLAM (RGB-D = left + depth);
            # off by default so it isn't downloaded — see the _run worker.
            include_right = gr.Checkbox(
                value=False, label="Download right OAK camera",
                info="Unused by SLAM (left + depth); leave off for a faster download.")
            # Read-only recap of the inputs, shown in their place while running.
            selection_card = gr.HTML(visible=False)
            # Idle: only Run. Running: Run greyed to 'Running', + Pause + Cancel.
            # Finished: only Reset. All visibility is driven by the generators.
            with gr.Row():
                run_btn = gr.Button("▶ Run", variant="primary", scale=2)
                pause_btn = gr.Button("⏸ Pause", variant="secondary",
                                      scale=1, visible=False)
                cancel_btn = gr.Button("✕ Cancel", variant="stop",
                                       scale=1, visible=False)
            # Cancel confirmation (revealed by Cancel; replaces Pause/Cancel).
            with gr.Row():
                confirm_cancel_btn = gr.Button("✓ Confirm cancel", variant="stop",
                                               size="sm", visible=False)
                keep_btn = gr.Button("Keep running", variant="secondary",
                                     size="sm", visible=False)
            # Revealed (as the primary action, replacing Run) only when the target
            # dataset already exists (see _run).
            confirm_branch_btn = gr.Button("⎇  Push to a new branch",
                                           variant="primary", visible=False)
            # Revealed only when a build succeeded but the push failed (see _run).
            retry_btn = gr.Button("⟳  Retry push", variant="secondary",
                                  size="sm", visible=False)
            # Shown once a run reaches a terminal state.
            reset_btn = gr.Button("↺  Reset", variant="secondary", visible=False)
            # Holds the built-but-unpushed dataset context for retry_push.
            retry_state = gr.State(None)
        # ---- Outputs (right) ----
        with gr.Column(scale=3, min_width=280):
            # Bar + log are revealed only while a run is in progress (running /
            # paused); the generators toggle their visibility via _io().
            bar = gr.HTML(_bar(0, "ready"), visible=False)
            log_out = gr.Textbox(label="Log", lines=15, max_lines=15, autoscroll=True,
                                 elem_id="logbox", visible=False)

            # ---- Episode review (reactive) ----
            # When SLAM flags non-GOOD trajectories the worker pauses and publishes
            # them; this panel lets the user drop episodes (🗑) before the dataset is
            # built. It is NOT part of the streaming generator's outputs — a Timer
            # mirrors the worker's _review flags into review_state, and @gr.render
            # redraws one orange card per still-kept flagged episode. Clicking 🗑
            # adds the episode to _review_drop and returns fresh state, so its card
            # disappears; Continue releases the worker with whatever remains.
            review_state = gr.State({"open": False, "items": [], "dropped": []})
            review_timer = gr.Timer(0.5)

            @gr.render(inputs=review_state)
            def _render_review(st):
                if not st.get("open"):
                    return
                items = st["items"]
                dropped = set(st["dropped"])
                kept = [it for it in items if it["name"] not in dropped]
                gr.Markdown(
                    f"### ⚠️ {len(items)} episode(s) flagged by the trajectory check\n"
                    f"Drop those to exclude from the dataset with 🗑, then **Continue**. "
                    f"The other episodes are kept. "
                    f"**{len(kept)}** flagged and kept."
                )
                for it in items:
                    if it["name"] in dropped:
                        continue  # removed → its card has disappeared
                    with gr.Row(equal_height=True):
                        gr.HTML(_episode_card(it))
                        gr.Button("🗑", size="sm", scale=0, min_width=52,
                                  elem_classes=["btn-red"]).click(
                            _drop_episode, inputs=gr.State(it["name"]),
                            outputs=review_state)
                with gr.Row():
                    gr.Button("▶  Continue (build & push kept episodes)",
                              size="sm", elem_classes=["btn-orange"]).click(
                        _submit_review, inputs=None, outputs=None)
                    if dropped:
                        gr.Button(f"↺  Restore {len(dropped)} removed",
                                  variant="secondary", size="sm").click(
                            _restore_episodes, inputs=None, outputs=review_state)

            # line_breaks=True so a single "\n" in a summary/warning renders as a
            # line break (GFM soft breaks) — plain Markdown would collapse it to a
            # space and only "\n\n" (a full paragraph gap) would show.
            summary_out = gr.Markdown(line_breaks=True)

    # Order MUST match the generators' yields: (bar, log, summary, retry_state),
    # the 8 buttons from _btns() (branch, retry, run, pause, cancel,
    # confirm_cancel, keep, reset), then the 5 from _inputs_view() (source, target,
    # task, include_right, selection_card). The review panel is reactive (Timer +
    # @gr.render), so it's deliberately NOT in this list.
    outputs = [bar, log_out, summary_out, retry_state,
               confirm_branch_btn, retry_btn,
               run_btn, pause_btn, cancel_btn, confirm_cancel_btn, keep_btn, reset_btn,
               source, target, task, include_right, selection_card]
    run_event = run_btn.click(run_pipeline, inputs=[source, target, task, include_right],
                              outputs=outputs)
    branch_event = confirm_branch_btn.click(run_pipeline_branch,
                                            inputs=[source, target, task, include_right],
                                            outputs=outputs)
    retry_event = retry_btn.click(retry_push, inputs=[retry_state], outputs=outputs)
    reset_btn.click(reset, inputs=None, outputs=outputs)
    # Pause/Resume + cancel-confirm just flip cooperative flags; the running
    # generator re-renders the buttons from them on its next tick (outputs=None
    # → these never write the same components as the stream, so no conflict).
    pause_btn.click(_toggle_pause, inputs=None, outputs=None)
    cancel_btn.click(_request_cancel, inputs=None, outputs=None)
    keep_btn.click(_keep_running, inputs=None, outputs=None)
    # Mirror the worker's review flags into review_state every tick; _poll_review
    # returns gr.skip() when nothing changed, so the @gr.render block only redraws
    # on a real open/close or drop-set change (no churn while idle or running).
    review_timer.tick(_poll_review, inputs=review_state, outputs=review_state)
    # Confirm cancel: first cancel the streaming generator + signal the worker to
    # abandon (outputs=None so it preempts at once), THEN reset the page to idle.
    # Split into click(cancel) → .then(reset) because a single function can't both
    # cancel a generator and reliably write that generator's outputs.
    cancel_evt = confirm_cancel_btn.click(
        confirm_cancel, inputs=None, outputs=None,
        cancels=[run_event, branch_event, retry_event])
    cancel_evt.then(reset, inputs=None, outputs=outputs)


if __name__ == "__main__":
    # queue() is required for streaming generators and for the Stop button's
    # cancels=[] to take effect. theme/css moved to launch() in Gradio 6.0.
    demo.queue().launch(theme=THEME, css=CSS,
                        server_name="0.0.0.0", server_port=7860)
