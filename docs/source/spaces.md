# Hugging Face Spaces

Two Hugging Face Spaces extend Grabette beyond the device: one turns your recordings into a LeRobot dataset without installing anything, the other lets you drive several devices as a fleet. Both use your Hugging Face account to decide what you can see and touch, so [log the device in](./dashboard.md#settings) first.

## Grabette SLAM → LeRobot

**[pollen-robotics/grabette-slam](https://huggingface.co/spaces/pollen-robotics/grabette-slam)**

Takes a raw Grabette recording that lives on the Hub and gives you back a [LeRobot](https://github.com/huggingface/lerobot) dataset, pushed under your own account.

1. Sign in with Hugging Face.
2. Give it a **source** `repo_id` — the raw recording dataset you uploaded from the dashboard's **Datasets** section — a **target** `repo_id` to create, and a task description.
3. For every episode the Space runs the full pipeline in-process: expand the recording, run RGBD-inertial odometry to recover the trajectory, assemble a LeRobot v3 dataset, and push it to the Hub.
4. When it finishes you get a link and an embedded [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) view of the result. The dataset has to be public for the visualizer to open it.

This is the same pipeline as the local one in [Getting started](./get_started.md#on-your-workstation) — the SLAM step is a compiled RTAB-Map binary, which normally runs in the `pollenrobotics/oak-vslam` Docker image. Spaces can't run Docker inside Docker, so this Space *is* that image, with the binary built in and called directly.

Use the Space when you don't want a local Docker and a multi-gigabyte pull; use the local pipeline when you want to inspect intermediate results, tune the checks, or work offline.

## Grabette Fleet

**[pollen-robotics/grabette-fleet](https://huggingface.co/spaces/pollen-robotics/grabette-fleet)**

An operator dashboard and command broker for several devices at once. Recording a manipulation task from two viewpoints, or having several people record in parallel, is much easier when one start press starts everything.

```
operator (Space, HF login) ──queue command──▶ broker ◀──poll── device (Pi)
                           ◀──device status──                  (Bearer hf_token)
device data ─────────────────────────────────────────────────▶ HF dataset
```

Devices connect outbound with their Hugging Face token; the Space resolves the owner and groups devices by identity, so you only ever see and control your own. Once devices are grouped, a start or stop from *any* trigger — the physical button, the device's own dashboard, or the Space — is broadcast to the whole group.

Group synchronization is deliberately best-effort. If the device isn't grouped, isn't logged in, or the Space is unreachable, the local recording still starts and stops normally; it just runs solo. A sleeping Space delays nothing more than a few seconds.

### Running your own

One Space is one owner's fleet. To run yours, open the Space menu and **Duplicate this Space** — the OAuth setup is provisioned automatically — then point your devices at it:

```bash
GRABETTE_RELAY_URL=https://<your-username>-grabette-fleet.hf.space
```

Both Spaces run on the free CPU tier.

<Tip>

The `-test` variants of these Spaces are development deployments. Use the ones linked above.

If you do need to work against them, set `GRABETTE_FLEET_ENV=test` on the device
rather than writing a URL: it moves the relay and the OAuth redirect together, and
the dashboard then shows an amber fleet tile naming the Space, so no device stays
pointed there by accident. The fleet Space picks its own conversion Space to
match, so a test fleet never converts through the production pipeline.

</Tip>
