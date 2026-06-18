"""Run controller — cooperative flags, control handlers, and the streaming
generators behind Run / Push-to-branch / Retry.

The heavy work runs in a worker thread; its output (log() + every print()) streams
through a queue back to the generator, which re-renders the page on each tick. Pure
rendering lives in views.py; the episode-review state + panel live in review.py
(imported here so the worker can publish flagged episodes and block for the user's
decision).
"""

import contextlib
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

import review
from pipeline import build_lerobot, push_lerobot
from views import bar, btns, error_card, inputs_view, io, run_recap, success_summary

# Cooperative control flags, shared between the worker and the control buttons.
# The Space processes one run at a time, so module-level flags are enough.
#   _stop           — abandon the run (Cancel confirmed)
#   _pause          — hold the worker at the next safe checkpoint (Pause)
#   _cancel_pending — Cancel was clicked, awaiting confirmation
# The streaming generator is the ONLY writer of button state: it re-renders the
# buttons from these flags on every tick. The handlers below just flip a flag
# (outputs=None), so they never fight the generator for the same components.
_stop = threading.Event()
_pause = threading.Event()
_cancel_pending = threading.Event()


def toggle_pause():
    """Pause ⇄ Resume: flip the cooperative pause flag (button relabels itself on
    the generator's next tick)."""
    _pause.clear() if _pause.is_set() else _pause.set()


def request_cancel():
    """Cancel clicked: reveal the confirm/keep buttons (handled by the generator)."""
    _cancel_pending.set()


def keep_running():
    """'Keep running' clicked: dismiss the cancel confirmation."""
    _cancel_pending.clear()


def confirm_cancel():
    """Confirm-cancel handler: only signal the worker to abandon (outputs=None,
    like the old Stop, so it preempts the running generator via cancels=). The
    UI reset is done by reset() chained as .then() on this click — a single
    function that both cancels= and writes outputs can't reliably apply its
    outputs (it'd queue behind the generator it's cancelling)."""
    _stop.set()


def _wait_if_paused():
    """Block while paused (used as the worker's checkpoint gate). Returns at once
    if a stop/cancel is requested, so pausing then cancelling never deadlocks."""
    while _pause.is_set() and not _stop.is_set():
        time.sleep(0.2)


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


def _preflight(api, source_repo, target_repo, scopes=frozenset()):
    """Quick access/existence checks. Returns (exists, writable, error_or_None).

    error_or_None is a clear, user-facing message (never a raw HF traceback) when
    the source can't be read or the target can't be written — shown in the red
    error card before any heavy work starts. Both checks run up front so a bad
    source or an unwritable target fails fast, not after a long SLAM run.

    scopes: the OAuth scopes actually granted to the sign-in token. Owning the
    namespace isn't enough to *create* a repo there — HF gates creation behind the
    'manage-repos' scope (writing content to an existing repo only needs
    'write-repos'). So if the target doesn't exist yet and that scope wasn't
    granted (partial consent), creation would 403 *after* a long SLAM run; we
    catch it here instead.
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

    # Owning/writing the namespace is necessary but not sufficient to *create* a
    # repo: creation needs the 'manage-repos' scope. Only enforce it when the
    # target doesn't exist yet — an existing repo is reached by writing content
    # ('write-repos'), and a non-branch existing target is intercepted upstream.
    if not exists and "manage-repos" not in scopes:
        return exists, writable, (
            f"'{target_repo}' doesn't exist yet, and creating a new dataset needs "
            f"the 'manage-repos' permission — which your sign-in is missing.\n"
            f"Sign out and back in to grant it, then re-run. (Pushing to an "
            f"existing dataset only needs write access, which you have.)"
        )

    return exists, writable, None


def _reset_view():
    """Full reset to the idle page (clears log/summary/bar, only Run shown).

    Clears the UI-facing flags (pause, cancel-pending, review) but NOT _stop: when
    this runs as the .then() after a confirmed Cancel, the background worker is still
    winding down and relies on _stop staying set to abandon (skip the push). _run
    clears _stop itself at the start of the next run.
    """
    _pause.clear()
    _cancel_pending.clear()
    review.reset()
    return (*io("idle", bar(0, "ready"), ""), "", None, *btns("idle"),
            *inputs_view("idle"))


def _run(source_repo, target_repo, task, oauth_token, to_branch):
    """Streaming generator behind Run / Push-to-branch. Yields the 16-element
    `outputs` tuple: bar, log, summary, retry_state, the 8 buttons, then the 4
    inputs-view fields. The episode-review panel is a separate reactive @gr.render
    block (review.build_panel), not part of this tuple.

    Pre-flight checks run synchronously up front (source read, target write, target
    existence — an existing target reveals the "push to a branch" button); the heavy
    work then runs in a worker thread and streams through a queue.
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
    scopes = frozenset((getattr(oauth_token, "scope", "") or "").split())
    os.environ["HF_TOKEN"] = token

    _stop.clear()
    _pause.clear()
    _cancel_pending.clear()  # fresh run
    review.reset()

    logs: list[str] = []
    start = time.time()
    frac, label = 0.02, "checking"
    retry_ctx = None  # set when a push fails but the built dataset is reusable

    def render() -> str:
        return "\n".join(logs)

    def view(summary="", *, state="running", branch=False, retry=False):
        in_review = review._review.is_set()
        lbl = ("awaiting review" if in_review
               else f"paused · {label}" if _pause.is_set()
               else f"{label} · {int(time.time() - start)}s")
        return (*io(state, bar(frac, lbl), render()), summary, retry_ctx,
                *btns(state, paused=_pause.is_set(),
                      cancel_pending=_cancel_pending.is_set(),
                      branch=branch, retry=retry,
                      allow_pause=not in_review),
                *inputs_view(state, source=source_repo, target=target_repo,
                             task=task))

    # ---- Pre-flight (synchronous, fast) ---------------------------------
    logs.append("Checking repo access…")
    yield view()  # 'Running' + Pause + Cancel appear immediately
    api = HfApi(token=token)
    exists, _writable, err = _preflight(api, source_repo, target_repo, scopes)
    if err:
        logs.append(err)
        # Recoverable: back to idle so the user can fix the inputs and re-run.
        frac, label = 0.0, "error"
        yield view(summary=error_card(err), state="idle")
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
        if phase == "check":
            f = 0.10 + 0.18 * (done / total if total else 1.0)
            lbl = f"checking {done}/{total}" if total else "checking"
        elif phase == "slam":
            f = 0.30 + 0.55 * (done / total if total else 1.0)
            lbl = f"SLAM {done}/{total}" if total else "SLAM"
        elif phase == "build":
            f, lbl = 0.88, "building dataset"
        elif phase == "push":
            f, lbl = 0.95, ("pushing branch" if to_branch else "pushing")
        else:
            f, lbl = 0.30, phase
        q.put(("progress", f, lbl))

    def _review_gate(kind, items, all_eps):
        """Publish `items` to the (kind="input"/"trajectory") review panel, block
        the worker until Continue (or a cancel), then return the kept episode dirs.

        all_eps is the full [(name, ep_dir), …] in order; the kept subset is the
        episodes the user did NOT remove (🗑). On cancel the full set is returned
        and the build then hits should_stop and raises — so nothing is pushed."""
        review._review_kind[0] = kind
        review._review_items[:] = items
        review._review_drop.clear()
        review._review_done.clear()
        review._review.set()
        while not review._review_done.is_set():
            if _stop.is_set():
                review._review.clear()
                return [ep for _, ep in all_eps]
            time.sleep(0.2)
        review._review.clear()
        drop = set(review._review_drop)
        kept = [ep for name, ep in all_eps if name not in drop]
        if drop:
            print(f"Dropped {len(drop)} episode(s): {', '.join(sorted(drop))}.")
        print(f"Keeping {len(kept)} episode(s).")
        return kept

    def pre_review_cb(checks):
        """Called by build_lerobot between the completeness/sync prechecks and SLAM.
        checks is a list of {ep, name, errors, warnings, sync} dicts. Returns the
        episode dirs to keep — letting the user drop incomplete/desynced recordings
        BEFORE the slow SLAM runs. A clean run (nothing flagged) continues untouched."""
        def flagged(c):
            return bool(c["errors"] or c["warnings"]) or (
                c["sync"] is not None and c["sync"]["verdict"] != "GOOD")
        bad = [c for c in checks if flagged(c)]
        if not bad:
            return [c["ep"] for c in checks]
        items = [
            {"name": c["name"], "kind": "input",
             "verdict": ("ERROR" if c["errors"]
                         or (c["sync"] and c["sync"]["verdict"] == "BAD") else "WARN"),
             "messages": [
                 *(f"[error] {m}" for m in c["errors"]),
                 *c["warnings"],
                 *([c["sync"]["message"]]
                   if c["sync"] and c["sync"]["verdict"] != "GOOD" else []),
             ]}
            for c in bad
        ]
        print(f"⏸ Pre-SLAM review: {len(bad)} of {len(checks)} episode(s) flagged by "
              f"the dataset / sync check — remove the ones to drop (🗑), then Continue.")
        return _review_gate("input", items, [(c["name"], c["ep"]) for c in checks])

    def review_cb(results):
        """Called by build_lerobot between SLAM and the build. results is a list of
        (episode_dir, TrajectoryReport). Returns the episode dirs to keep — letting
        the user drop episodes whose trajectory came back flagged. A clean run
        (all GOOD) continues untouched."""
        flagged = [(ep, rep) for ep, rep in results if rep.verdict != "GOOD"]
        if not flagged:
            return [ep for ep, _ in results]
        items = [
            {"name": ep.name, "verdict": rep.verdict,
             "n_tracked": rep.n_tracked, "tracking_pct": rep.tracking_pct,
             "total_distance_m": rep.total_distance_m, "duration_s": rep.duration_s,
             "median_step_mm": rep.median_step_mm, "median_angle_deg": rep.median_angle_deg,
             "n_jumps": rep.n_jumps,
             "messages": [*rep.errors, *rep.warnings]}
            for ep, rep in flagged
        ]
        print(f"⏸ Trajectory review: {len(flagged)} of {len(results)} episode(s) flagged "
              f"— remove the ones to drop (🗑), then click Continue.")
        return _review_gate("trajectory", items, [(ep.name, ep) for ep, _ in results])

    def worker():
        writer = _LineQueueWriter(q)
        try:
            with contextlib.redirect_stdout(writer):
                q.put(("progress", 0.07, "download"))
                work = Path(tempfile.mkdtemp())
                try:
                    info = api.repo_info(source_repo, repo_type="dataset", files_metadata=True)
                    mb = sum((s.size or 0) for s in info.siblings) / 1e6
                    print(f"Downloading {source_repo} — {len(info.siblings)} files, {mb:.0f} MB…")
                except Exception:
                    print(f"Downloading {source_repo}…")
                # Raw episodes are dominated by many small depth PNGs (≈600/ep),
                # so the download is bound by per-file request overhead, not
                # bandwidth — more concurrent connections is the main lever.
                raw = snapshot_download(source_repo, repo_type="dataset",
                                        local_dir=work / "raw", token=token,
                                        max_workers=48)
                print("Download complete.\n\n")
                q.put(("progress", 0.30, "converting"))
                ds_root = work / "lerobot"
                processed = build_lerobot(
                    raw, target_repo, task, root=ds_root,
                    log=print, should_stop=_stop.is_set, to_branch=to_branch,
                    on_progress=on_progress, token=token, gate=_wait_if_paused,
                    pre_review=pre_review_cb, review=review_cb)
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
            msg = error_card(
                f"Push failed: {err}\n\nThe dataset is built and cached — click "
                f"“Retry push” to push it again without re-running SLAM.")
            label = "push failed"
            yield view(summary=msg + run_recap(logs), state="finished", retry=True)
        else:
            label = "failed"
            yield view(summary=error_card(f"Pipeline failed: {err}") + run_recap(logs),
                       state="finished")
        return

    n = result["n"]
    link = result.get("link")
    mode = result.get("mode")
    logs.append(f"✅ Done — {n} episode(s).")
    frac, label = 1.0, "done"
    yield view(summary=success_summary(target_repo, n, link, mode) + run_recap(logs),
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
    review.reset()

    logs: list[str] = []
    start = time.time()
    frac, label = 0.95, "pushing"

    def render() -> str:
        return "\n".join(logs)

    def view(summary="", *, state="running", retry=False, ctx=retry_ctx):
        # 16-output shape, same as _run. A push can't be paused, so allow_pause
        # is False; Cancel still works (abandons the retry).
        return (*io(state, bar(frac, f"{label} · {int(time.time() - start)}s"), render()),
                summary, ctx,
                *btns(state, cancel_pending=_cancel_pending.is_set(),
                      retry=retry, allow_pause=False),
                *inputs_view(state, target=retry_ctx["target_repo"],
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
        yield view(summary=error_card(f"Push failed again: {err}"),
                   state="finished", retry=True, ctx=retry_ctx)
        return

    n = result["n"]
    link = result.get("link")
    mode = result.get("mode")
    logs.append(f"✅ Pushed — {n} episode(s).")
    frac, label = 1.0, "done"
    yield view(summary=success_summary(retry_ctx["target_repo"], n, link, mode),
               state="finished", ctx=None)


def run_pipeline(source_repo, target_repo, task,
                 oauth_token: gr.OAuthToken | None = None):
    yield from _run(source_repo, target_repo, task, oauth_token, to_branch=False)


def run_pipeline_branch(source_repo, target_repo, task,
                        oauth_token: gr.OAuthToken | None = None):
    yield from _run(source_repo, target_repo, task, oauth_token, to_branch=True)


def reset():
    """Reset button (shown when finished): clear the page and return to idle
    (only Run). Inputs (source/target/task) are kept so the user can re-run."""
    return _reset_view()
