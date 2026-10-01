# Data collection

Real data is recorded through the **Grabette Fleet** Space: it organizes recordings into tasks and sessions, drives several devices at once, and turns the episodes into a LeRobot dataset. The device has to be [signed in to Hugging Face](./usage.md#signing-in-to-hugging-face) first.

## Introduction to grabette-fleet

**[pollen-robotics/grabette-fleet](https://huggingface.co/spaces/pollen-robotics/grabette-fleet)**

An operator dashboard and command broker for your devices. From the Space you:

- create **tasks** and open recording **sessions** for them;
- **upload** the devices' episodes to a Hugging Face dataset, and **build the LeRobot dataset** through the [SLAM Space](#build-the-lerobot-dataset);
- run a **Trajectory Check** on the last few episodes of a session, to catch SLAM tracking failures without waiting for the end of the shoot;
- **power off** a device.

```
operator (Space, HF login) ──queue command──▶ broker ◀──poll── device (Pi)
                           ◀──device status──                  (Bearer hf_token)
device data ─────────────────────────────────────────────────▶ HF dataset
```

Devices connect outbound with their Hugging Face token; the Space resolves the owner, so you only ever see and control your own devices.

<!-- TODO: screenshot of the Fleet Space. -->

### Running your own

One Space is one owner's fleet. To run yours, open the Space menu and **Duplicate this Space** — the OAuth setup is provisioned automatically — then point your devices at it:

```bash
# in /etc/grabette/env on each device, then: sudo systemctl restart grabette
GRABETTE_RELAY_URL=https://<your-username>-grabette-fleet.hf.space
```

The Fleet Space runs on the free CPU tier.

<Tip>

The `-test` variants of the Spaces are development deployments. Use the ones linked on this page.

</Tip>

## Start a session

<!-- TODO: how to open, run and close a recording session in the Fleet Space. -->

## Manage tasks

<!-- TODO: creating, editing and selecting tasks in the Fleet Space. A task is the natural-language description of what is demonstrated, such as "pick up the cup"; it follows each episode into the LeRobot dataset. -->

## Upload a dataset

<!-- TODO: uploading a session's episodes from the Fleet Space. -->

### Build the LeRobot dataset

**[pollen-robotics/grabette-slam](https://huggingface.co/spaces/pollen-robotics/grabette-slam)**

The SLAM Space takes a raw Grabette recording that lives on the Hub and gives you back a [LeRobot](https://github.com/huggingface/lerobot) dataset, pushed under your own account. The Fleet Space triggers it for you; it can also be used on its own:

1. Sign in with Hugging Face.
2. Give it a **source** `repo_id` — the raw recording dataset uploaded through the Fleet Space — a **target** `repo_id` to create, and a task description.
3. For every episode the Space runs the full pipeline: expand the recording, run RGB-D odometry (inertial too, with an OAK-D recording) to recover the trajectory, assemble a LeRobot v3 dataset, and push it to the Hub.
4. When it finishes you get a link and an embedded [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) view of the result. The dataset has to be public for the visualizer to open it.

The SLAM step is a compiled RTAB-Map binary, which normally runs in the `pollenrobotics/oak-vslam` Docker image. Spaces can't run Docker inside Docker, so this Space is built on that image, with the binary compiled in and called directly. It runs on upgraded CPU hardware, so a duplicate of it needs the same, or it will be slow.

## Bimanual recording

Two Grabettes — one built as a left hand, one as a right hand — can record the same demonstration together. Put both devices into a **group** in the Fleet Space: a start or stop from either trigger — a device's physical button, or the Space — is then broadcast to the whole group, and every device starts at the same shared instant. Each device's LED shows its own state, including the one started from the other device.

Group synchronization is best-effort: if a device isn't grouped, isn't signed in, or the Space is unreachable, its recording still starts and stops normally, just solo. The devices' clocks must agree for the recordings to line up — see the [FAQ](./faq.md#recordings-from-two-devices-dont-line-up).

<!-- TODO: bimanual specifics — setting up the group and task, how a bimanual episode lands in the dataset. -->
