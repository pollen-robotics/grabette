"""Grabette SLAM → LeRobot pipeline — HuggingFace Space (Gradio + HF OAuth).

The user signs in with their HF account; the OAuth token is used to download the
source dataset and push the generated LeRobot dataset under their account.
Gradio auto-injects the gr.OAuthToken parameter — it is NOT a UI input.
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
from huggingface_hub import snapshot_download

from pipeline import process_dataset

VISUALIZER = "https://huggingface.co/spaces/lerobot/visualize_dataset"


def _visualizer_url(repo_id: str) -> str:
    return f"{VISUALIZER}?dataset={repo_id}&episode=0"


def run(source_repo, target_repo, task,
        oauth_token: gr.OAuthToken | None, progress=gr.Progress()):
    if oauth_token is None:
        raise gr.Error("Sign in with your Hugging Face account first.")
    if not (source_repo and target_repo and task):
        raise gr.Error("Fill in source repo, target repo and task.")

    # OAuth access tokens aren't accepted by huggingface_hub.login() (it expects a
    # classic token's role). Expose the token via env + pass it explicitly so
    # snapshot_download and push_to_hub pick it up.
    os.environ["HF_TOKEN"] = oauth_token.token

    logs: list[str] = []

    def log(msg: str):
        logs.append(msg)
        progress(0.5, desc=msg)  # keep the UI progress bar alive

    work = Path(tempfile.mkdtemp())
    log(f"Downloading {source_repo}…")
    raw = snapshot_download(source_repo, repo_type="dataset",
                            local_dir=work / "raw", token=oauth_token.token)

    n = process_dataset(raw, target_repo, task, root=work / "lerobot", log=log)

    ds_url = f"https://huggingface.co/datasets/{target_repo}"
    viz_url = _visualizer_url(target_repo)
    log(f"✅ Pushed {n} episode(s).")

    # The LeRobot visualizer sends X-Frame-Options: deny, so it can't be embedded
    # in an iframe — link out (opens in a new tab) instead.
    summary = (
        f"### ✅ Done — {n} episode(s)\n"
        f"- **Dataset:** [{target_repo}]({ds_url})\n"
        f"- **Visualize:** 👉 [open in LeRobot visualizer]({viz_url}) 👈\n\n"
        f"_(The visualizer needs the dataset to be public.)_"
    )
    return "\n".join(logs), summary


with gr.Blocks(title="Grabette SLAM → LeRobot") as demo:
    gr.Markdown(
        "# 🤖 Grabette SLAM → LeRobot\n"
        "Convert a raw OAK-D dataset from the Hub into a LeRobot dataset and push "
        "it back to the Hub. Each episode is converted, run through SLAM, assembled "
        "into LeRobot v3 format, then uploaded."
    )
    gr.LoginButton()
    source = gr.Textbox(label="Source dataset (raw repo_id)",
                        placeholder="pollen-robotics/grabette-raw")
    target = gr.Textbox(label="Target dataset (repo_id to create)",
                        placeholder="your-account/grabette-demo")
    task = gr.Textbox(label="Task description", placeholder="cup manipulation")
    btn = gr.Button("Run pipeline", variant="primary")

    log_out = gr.Textbox(label="Log", lines=15)
    summary_out = gr.Markdown()

    btn.click(run, inputs=[source, target, task],
              outputs=[log_out, summary_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
