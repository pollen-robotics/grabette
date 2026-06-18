"""Episode review — the cooperative state + the reactive @gr.render panel that
lets the user drop flagged trajectories before the dataset is built.

When SLAM flags ≥1 non-GOOD episode, the run controller (worker thread) publishes
them via this module's flags and blocks on `_review_done` while `_review` is set.
A gr.Timer mirrors the flags into the panel's gr.State (poll), and @gr.render draws
one card per still-kept episode. The controller imports this module and touches the
flags module-qualified (review._review.set(), review._review_items[:] = …), so there
is exactly one copy of each flag — no cross-module duplication.
"""

import threading

import gradio as gr

from views import episode_card

# Published by the worker (controller's review callbacks), consumed by the panel.
# A run has two sequential review gates — pre-SLAM (kind "input": completeness +
# sync) and post-SLAM (kind "trajectory") — so the same flags are reused twice;
# `_review_kind` tells the panel which gate it is currently drawing.
_review = threading.Event()        # worker is awaiting the user's review decision
_review_done = threading.Event()   # Continue clicked → release the worker
_review_items: list[dict] = []     # flagged episodes published for the panel
_review_drop: list[str] = []       # episode names the user chose to drop
_review_kind = ["trajectory"]      # mutable holder: "input" or "trajectory"


def state() -> dict:
    """Snapshot for the reactive @gr.render; `open` drives whether the panel shows."""
    return {"open": _review.is_set(),
            "kind": _review_kind[0],
            "items": list(_review_items),
            "dropped": list(_review_drop)}


def reset() -> None:
    """Clear every review flag (called by the controller's run reset)."""
    _review.clear()
    _review_done.clear()
    _review_drop.clear()
    _review_kind[0] = "trajectory"


def drop_episode(name: str) -> dict:
    """🗑 clicked on a flagged episode: add it to the drop set and return the fresh
    state so the @gr.render panel re-runs and the card disappears."""
    if name not in _review_drop:
        _review_drop.append(name)
    return state()


def restore_episodes() -> dict:
    """'Restore' clicked: clear the drop set (un-hide every flagged card)."""
    _review_drop.clear()
    return state()


def submit_review() -> None:
    """'Continue' clicked: release the worker. The episodes to drop already live in
    _review_drop (set as the user clicked 🗑). outputs=None — like Pause/Cancel, this
    only flips a flag; the worker clears _review and the panel closes on the next poll."""
    _review_done.set()


def poll(cur: dict):
    """gr.Timer tick: mirror the worker's flags into the panel State so it opens/
    closes and its drop set stays in sync. Returns gr.skip() when nothing changed,
    so an idle session never churns the @gr.render block."""
    nxt = state()
    if nxt == cur:
        return gr.skip()
    return nxt


def build_panel(review_state) -> None:
    """Register the reactive review panel on the *current* Blocks. Call this inside
    `with gr.Blocks():` — @gr.render binds to the active Blocks. `review_state` is the
    gr.State the Timer feeds (see poll)."""
    @gr.render(inputs=review_state)
    def _render(st):
        if not st.get("open"):
            return
        items = st["items"]
        dropped = set(st["dropped"])
        kept = [it for it in items if it["name"] not in dropped]
        # Pre-SLAM gate (kind "input") flags incomplete/desynced recordings before
        # the slow SLAM; post-SLAM gate (kind "trajectory") flags bad odometry.
        if st.get("kind") == "input":
            source, action = "dataset / sync check", "run SLAM on"
        else:
            source, action = "trajectory check", "build & push"
        gr.Markdown(
            f"### ⚠️ {len(items)} episode(s) flagged by the {source}\n"
            f"Drop those to exclude with 🗑, then **Continue**. "
            f"The other episodes are kept. "
            f"**{len(kept)}** flagged and kept."
        )
        for it in items:
            if it["name"] in dropped:
                continue  # removed → its card has disappeared
            with gr.Row(equal_height=True):
                gr.HTML(episode_card(it))
                gr.Button("🗑", size="sm", scale=0, min_width=52,
                          elem_classes=["btn-red"]).click(
                    drop_episode, inputs=gr.State(it["name"]), outputs=review_state)
        with gr.Row():
            gr.Button(f"▶  Continue ({action} kept episodes)",
                      size="sm", elem_classes=["btn-orange"]).click(
                submit_review, inputs=None, outputs=None)
            if dropped:
                gr.Button(f"↺  Restore {len(dropped)} removed",
                          variant="secondary", size="sm").click(
                    restore_episodes, inputs=None, outputs=review_state)
