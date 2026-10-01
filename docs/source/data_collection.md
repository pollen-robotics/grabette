# Data collection

Real data is recorded through the **Grabette Fleet** Space: it organizes recordings into tasks and sessions, drives several devices at once, and turns the episodes into a LeRobot dataset. The device has to be [signed in to Hugging Face](./get_started.md#signing-in-to-hugging-face) first.

## Grabette Fleet

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

### Manage tasks

<!-- TODO: creating, editing and selecting tasks in the Fleet Space. A task is the natural-language description of what is demonstrated, such as "pick up the cup"; it follows each episode into the LeRobot dataset. -->

### Start a session

<!-- TODO: how to open, run and close a recording session in the Fleet Space. -->

### Start and stop a recording

Press the button on the device to start a recording, and press it again to stop. The device's dashboard has no record button, so you keep both hands on the task.

The button's LED shows the device's state:

| LED | State |
| :--- | :--- |
| Off | Idle |
| Blinking | Starting — the cameras are warming up |
| Solid | Recording |
| Fast blinking | Stopping and saving the episode |
| 1 pulse, pause, repeat | Busy uploading or converting data — recording is refused until it finishes |
| 3 pulses, pause, repeat | Hardware fault — the device refuses to record |

If a speaker is fitted, the device also beeps: an **ascending** beep when the recording actually starts (after the warm-up, not when you press), and a **descending** beep when it stops. Wait for the first beep, or the solid LED, before you start the motion.

### Upload a dataset

<!-- TODO: uploading a session's episodes from the Fleet Space. -->

### Build the LeRobot dataset

**[pollen-robotics/grabette-slam](https://huggingface.co/spaces/pollen-robotics/grabette-slam)**

The SLAM Space takes a raw Grabette recording that lives on the Hub and gives you back a [LeRobot](https://github.com/huggingface/lerobot) dataset, pushed under your own account. The Fleet Space triggers it for you; it can also be used on its own:

1. Sign in with Hugging Face.
2. Give it a **source** `repo_id` — the raw recording dataset uploaded through the Fleet Space — a **target** `repo_id` to create, and a task description.
3. For every episode the Space runs the full pipeline: expand the recording, run RGB-D odometry (inertial too, with an OAK-D recording) to recover the trajectory, assemble a LeRobot v3 dataset, and push it to the Hub.
4. When it finishes you get a link and an embedded [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) view of the result. The dataset has to be public for the visualizer to open it.

The SLAM step is a compiled RTAB-Map binary, which normally runs in the `pollenrobotics/oak-vslam` Docker image. Spaces can't run Docker inside Docker, so this Space is built on that image, with the binary compiled in and called directly. It runs on upgraded CPU hardware, so a duplicate of it needs the same, or it will be slow.

### Running your own fleet

One Space is one owner's fleet. To run yours, open the Space menu and **Duplicate this Space** — the OAuth setup is provisioned automatically — then point your devices at it:

```bash
# in /etc/grabette/env on each device, then: sudo systemctl restart grabette
GRABETTE_RELAY_URL=https://<your-username>-grabette-fleet.hf.space
```

The Fleet Space runs on the free CPU tier.

<Tip>

The `-test` variants of the Spaces are development deployments. Use the ones linked on this page.

</Tip>

## Guidelines for recording a dataset

A policy reproduces the *statistics* of your demonstrations. It keeps the variation it can predict from what the camera sees, and **averages** the variation it can't. Two rules follow, and almost every recording mistake breaks one of them:

1. **Be consistent** where the camera can't tell your intent — stylistic choices. Variation there gets averaged into one compromise, often worse than any demonstration you gave.
2. **Be diverse** where you need the policy to generalize, and the cause is visible to the camera.

### Be consistent

- **Grasp angle** — pick one natural, easy angle and stay within about ±10° of it. Grasping the same object from very different angles makes the policy average them into a bad grasp.
- **Reach decisively** — drive the object deep into the jaw in one continuous motion. A gentle settle as the fingers touch teaches the policy to stall short of the object.
- **Close firmly** — one committed close. The policy imitates hesitation, and ends up touching without grasping.
- **Move smoothly, at a steady pace** — no jitter, no micro-corrections. Shaky demonstrations give an erratic policy.

### Be diverse

- **Object position** — cover the whole workspace you'll use the policy in. It only works where it has seen the object.
- **Object orientation** — only if your grasp follows the object's orientation; otherwise it's a free angle choice in disguise.
- **Start poses** you'll actually encounter.
- **Lighting, above all** — one recording session captures exactly one lighting condition, and a change at deployment degrades the policy. Record batches at different times of day, with blinds open and closed, lights on and off. Hard shadows and sun patches can't be covered by training augmentation, only by data.

### Failures and retries

Record clean, first-try successes by default: a miss-then-fix in an episode teaches the policy to miss. If you want recovery behavior, record it on purpose — about 10 % of the episodes, with misses in varied directions, and a failure the camera can see.

### Keep the SLAM tracking

The trajectory is recovered by SLAM from the camera, which needs to see the static scene. When it can't, the trajectory jumps, and it's usually the grasp — the most important part — that is lost.

- **Don't let the object fill the camera** as you grasp and lift it. Keep some background in view. This is the most common cause of tracking loss.
- **No fast swings** — motion blur and large jumps between frames lose the tracking.

### Record what the robot can do

The hand-held device has no joint limits and no speed limit; the robot arm has both, and the policy can't see them. Keep lifts modest and central, and keep your hand speed moderate — around 0.5 m/s at most.

### Checklist

- [ ] One natural grasp angle, within ±10°.
- [ ] Reach through, to seat the object deep in the jaw.
- [ ] A firm, decisive close.
- [ ] Smooth motion at a steady pace, no fast swings.
- [ ] The object never fills the camera view.
- [ ] Varied object positions — and orientations only if the grasp follows them.
- [ ] Varied lighting across sessions.
- [ ] Modest, central lifts; hand speed around 0.5 m/s at most.
- [ ] Clean first-try successes, no accidental misses.

The [full recording guide](https://github.com/pollen-robotics/grabette/blob/develop/integrations/DiffusionPolicy/recording_demonstrations_guide.md) explains each rule and the failure it prevents.
