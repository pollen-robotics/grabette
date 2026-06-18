"""Grabette SLAM → LeRobot pipeline — HuggingFace Space (Gradio + HF OAuth).

The user signs in with their HF account; the OAuth token is used to download the
source dataset and push the generated LeRobot dataset under their account.
Gradio auto-injects the gr.OAuthToken parameter — it is NOT a UI input.

This module is just the UI: it builds the Blocks layout and wires events. The
rendering helpers live in views.py, the streaming run generators + cooperative
flags in controller.py, and the episode-review state + reactive panel in review.py.
"""

import gradio as gr

import review
from controller import (
    confirm_cancel,
    keep_running,
    request_cancel,
    reset,
    retry_push,
    run_pipeline,
    run_pipeline_branch,
    toggle_pause,
)
from views import CSS, THEME, bar

with gr.Blocks(title="Grabette SLAM → LeRobot", fill_height=True) as demo:
    gr.HTML(
        "<div id='app-header'><h1>Grabette post-processing</h1>"
        "<p>Raw Grabette recording → SLAM → LeRobot dataset on the Hub</p></div>"
    )
    with gr.Row():
        # ---- Inputs (left) ----
        with gr.Column(scale=2, min_width=280):
            gr.LoginButton(size="sm")
            source = gr.Textbox(label="Source dataset", info="raw Grabette repo on the Hub",
                                placeholder="pollen-robotics/grabette-raw")
            target = gr.Textbox(label="Target dataset", info="repo to create or contribute to",
                                placeholder="your-account/grabette-dataset")
            task = gr.Textbox(label="Task description", placeholder="cup grasping")
            # Read-only recap of the inputs, shown in their place while running.
            selection_card = gr.HTML(visible=False)
            # Idle: only Run. Running: Run greyed to 'Running', + Pause + Cancel.
            # Finished: only Reset. All visibility is driven by the generators.
            with gr.Row():
                run_btn = gr.Button("▶  Run", variant="primary", scale=2)
                pause_btn = gr.Button("⏸  Pause", variant="secondary",
                                      scale=1, visible=False)
                cancel_btn = gr.Button("✕  Cancel", variant="stop",
                                       scale=1, visible=False)
            # Cancel confirmation (revealed by Cancel; replaces Pause/Cancel).
            with gr.Row():
                confirm_cancel_btn = gr.Button("✓  Confirm cancel", variant="stop",
                                               size="sm", visible=False)
                keep_btn = gr.Button("Keep running", variant="secondary",
                                     size="sm", visible=False)
            # Revealed (as the primary action, replacing Run) only when the target
            # dataset already exists (see the controller).
            confirm_branch_btn = gr.Button("⎇  Push to a new branch",
                                           variant="primary", visible=False)
            # Revealed only when a build succeeded but the push failed.
            retry_btn = gr.Button("⟳  Retry push", variant="secondary",
                                  size="sm", visible=False)
            # Shown once a run reaches a terminal state.
            reset_btn = gr.Button("↺  Reset", variant="secondary", visible=False)
            # Holds the built-but-unpushed dataset context for retry_push.
            retry_state = gr.State(None)
        # ---- Outputs (right) ----
        with gr.Column(scale=3, min_width=280):
            # Bar + log are revealed only while a run is in progress (running /
            # paused); the generators toggle their visibility via views.io().
            bar_out = gr.HTML(bar(0, "ready"), visible=False)
            log_out = gr.Textbox(label="Log", lines=15, max_lines=15, autoscroll=True,
                                 elem_id="logbox", visible=False)

            # Episode review (reactive): a gr.Timer mirrors the worker's review
            # flags into review_state, and review.build_panel draws one orange card
            # per still-kept flagged episode. Not part of the streaming outputs.
            review_state = gr.State({"open": False, "kind": "trajectory",
                                     "items": [], "dropped": []})
            review_timer = gr.Timer(0.5)
            review.build_panel(review_state)

            # line_breaks=True so a single "\n" in a summary/warning renders as a
            # line break (GFM soft breaks) — plain Markdown would collapse it to a
            # space and only "\n\n" (a full paragraph gap) would show.
            summary_out = gr.Markdown(line_breaks=True)

    # Order MUST match the generators' yields (views.io → bar, log; then summary,
    # retry_state; then views.btns → branch, retry, run, pause, cancel,
    # confirm_cancel, keep, reset; then views.inputs_view → source, target, task,
    # selection_card). The review panel is reactive, NOT in this list.
    outputs = [bar_out, log_out, summary_out, retry_state,
               confirm_branch_btn, retry_btn,
               run_btn, pause_btn, cancel_btn, confirm_cancel_btn, keep_btn, reset_btn,
               source, target, task, selection_card]
    run_event = run_btn.click(run_pipeline, inputs=[source, target, task],
                              outputs=outputs)
    branch_event = confirm_branch_btn.click(run_pipeline_branch,
                                            inputs=[source, target, task],
                                            outputs=outputs)
    retry_event = retry_btn.click(retry_push, inputs=[retry_state], outputs=outputs)
    reset_btn.click(reset, inputs=None, outputs=outputs)
    # Pause/Resume + cancel-confirm just flip cooperative flags; the running
    # generator re-renders the buttons from them on its next tick (outputs=None
    # → these never write the same components as the stream, so no conflict).
    pause_btn.click(toggle_pause, inputs=None, outputs=None)
    cancel_btn.click(request_cancel, inputs=None, outputs=None)
    keep_btn.click(keep_running, inputs=None, outputs=None)
    # Mirror the worker's review flags into review_state every tick; review.poll
    # returns gr.skip() when nothing changed, so the @gr.render block only redraws
    # on a real open/close or drop-set change (no churn while idle or running).
    review_timer.tick(review.poll, inputs=review_state, outputs=review_state)
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
