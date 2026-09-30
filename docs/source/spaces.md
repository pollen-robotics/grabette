# Hugging Face Spaces

Two Hugging Face Spaces extend Grabette beyond the device: one turns your recordings into a LeRobot dataset without installing anything, the other lets you drive several devices as a fleet. Both use your Hugging Face account to decide what you can see and touch, so [sign the device in](./dashboard.md#signing-in-to-hugging-face) first.

## Grabette SLAM → LeRobot

**[pollen-robotics/grabette-slam](https://huggingface.co/spaces/pollen-robotics/grabette-slam)**

Takes a raw Grabette recording that lives on the Hub and gives you back a [LeRobot](https://github.com/huggingface/lerobot) dataset, pushed under your own account.

1. Sign in with Hugging Face.
2. Give it a **source** `repo_id` — the raw recording dataset your devices uploaded through the [Fleet Space](#grabette-fleet) — a **target** `repo_id` to create, and a task description.
3. For every episode the Space runs the full pipeline in-process: expand the recording, run RGB-D odometry (inertial too, with an OAK-D recording) to recover the trajectory, assemble a LeRobot v3 dataset, and push it to the Hub.
4. When it finishes you get a link and an embedded [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) view of the result. The dataset has to be public for the visualizer to open it.

This is the same pipeline as the local one in [Getting started](./get_started.md#on-your-workstation) — the SLAM step is a compiled RTAB-Map binary, which normally runs in the `pollenrobotics/oak-vslam` Docker image. Spaces can't run Docker inside Docker, so this Space is built on that image, with the binary compiled in and called directly.

Use the Space when you don't want a local Docker and a multi-gigabyte pull; use the local pipeline when you want to inspect intermediate results, tune the checks, or work offline.

## Grabette Fleet

**[pollen-robotics/grabette-fleet](https://huggingface.co/spaces/pollen-robotics/grabette-fleet)**

An operator dashboard and command broker for your devices, and where organized recording happens. From the Space you:

- create **tasks** and open recording **sessions** for them;
- **upload** the devices' episodes to a Hugging Face dataset, and **build the LeRobot dataset** through the SLAM Space above;
- run a **Trajectory Check** on the last few episodes of a session, to catch SLAM tracking failures without waiting for the end of the shoot;
- **power off** a device.

Recording a manipulation task from two viewpoints, or having several people record in parallel, is much easier when one start press starts everything.

```
operator (Space, HF login) ──queue command──▶ broker ◀──poll── device (Pi)
                           ◀──device status──                  (Bearer hf_token)
device data ─────────────────────────────────────────────────▶ HF dataset
```

Devices connect outbound with their Hugging Face token; the Space resolves the owner, so you only ever see and control your own devices. Put several of them into a **group** in the Space, and a start or stop from *either* trigger — a device's physical button, or the Space — is broadcast to the whole group.

Group synchronization is deliberately best-effort. If the device isn't grouped, isn't logged in, or the Space is unreachable, the local recording still starts and stops normally; it just runs solo. A sleeping Space delays nothing more than a few seconds.

### Running your own

One Space is one owner's fleet. To run yours, open the Space menu and **Duplicate this Space** — the OAuth setup is provisioned automatically — then point your devices at it:

```bash
# in /etc/grabette/env on each device, then: sudo systemctl restart grabette
GRABETTE_RELAY_URL=https://<your-username>-grabette-fleet.hf.space
```

The Fleet Space runs on the free CPU tier. The SLAM Space runs on upgraded CPU hardware, so a duplicate of it needs the same, or it will be slow.

<Tip>

The `-test` variants of these Spaces are development deployments. Use the ones linked above.

</Tip>
