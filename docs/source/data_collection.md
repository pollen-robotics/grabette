# Data collection

Real data is recorded through the **Grabette Fleet** Space: it organizes recordings into tasks and sessions, drives several devices at once, and turns the episodes into a LeRobot dataset. The device has to be [signed in to Hugging Face](./get_started.md#signing-in-to-hugging-face) first.

## Grabette Fleet

**[pollen-robotics/grabette-fleet](https://huggingface.co/spaces/pollen-robotics/grabette-fleet)**

<video controls preload="metadata" src="https://huggingface.co/spaces/pollen-robotics/grabette-fleet/resolve/main/static/tutorial-space.mp4"></video>

Before you start: power on your Grabette, sign it in to Hugging Face from its [dashboard](./get_started.md#signing-in-to-hugging-face), and sign in to the Fleet Space with the same account.

An operator dashboard and command broker for your devices. From the Space you:

- create **tasks** and open recording **sessions** for them;
- **upload** the devices' episodes to a Hugging Face dataset, and **build the LeRobot dataset** through the [SLAM Space](#build-the-lerobot-dataset);
- run a **Trajectory Check** on the last few episodes of a session, to catch SLAM tracking failures without waiting for the end of the shoot;
- **power off** a device.


Devices connect outbound with their Hugging Face token; the Space resolves the owner, so you only ever see and control your own devices.

Each device in the **Fleet** list shows its battery, IP address and WiFi network, with buttons to open its dashboard, sign it out of Hugging Face, power it off, or remove it from the list. Recordings made outside a task, such as test recordings, are counted in a banner: **Sort them** assigns them to a task.

<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/fleet.png" alt="The Fleet Space: the device list, then the Record tab" width="500" style="display: block; margin: 0 auto;"/>

The rest of the page has two tabs: **1. Record** and **2. Build dataset**.

### Manage tasks

A task is the natural-language description of what you demonstrate, such as *"put the red cube in the blue box"*. It follows each episode into the LeRobot dataset.

In the **Record** tab, **Tasks** lists the tasks on your devices with their number of episodes. **Create task** adds one: give it a name, an optional description, then **Validate task**. Each task can be edited or deleted.

### Start a session

1. In the **Record** tab, select the task to record.
2. Select the devices to use, then **Launch session**. One session runs at a time.

The session panel has:

- a record button, which starts and stops a recording like the device's button;
- **Delete last episode**, for a bad take;
- the number of episodes recorded, and an **Audio signal** switch: a chime every 10 episodes;
- **Trajectory check**: pick the last 1 to 5 episodes and **Run check**, to see whether SLAM can compute their trajectory. Recording pauses while it runs.

**Close session** when you are done.

<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/fleet-session.png" alt="A recording session in the Fleet Space" width="500" style="display: block; margin: 0 auto;"/>

### Start and stop a recording

Press the button on the device to start a recording, and press it again to stop. The record button in the [session](#start-a-session) does the same.

When no episode was recorded in the last 30 seconds, the cameras first initialize: wait for the LED to stop blinking and the beep, or for the session's *Initializing* message to disappear, before you start the motion.

The button's LED shows the device's state:

| LED | State |
| :--- | :--- |
| Off | Idle |
| Blinking | Starting — the cameras are warming up |
| Solid | Recording |
| Fast blinking | Stopping and saving the episode |
| 1 pulse, pause, repeat | Busy uploading or converting data — recording is refused until it finishes |
| 3 pulses, pause, repeat | Hardware fault — the device refuses to record |

If a speaker is fitted, the device also beeps — set the volume, or mute it, on the dashboard's [Overview](./get_started.md#overview): an **ascending** beep when the recording actually starts (after the warm-up, not when you press). When you stop, a first beep marks the end of the recording, and a second one when the episode is saved.

### Build the LeRobot dataset

In the **Build dataset** tab:

1. Select the tasks to include.
2. Name the destination dataset, under your account, and tick **Private** if needed.
3. **Generate LeRobot dataset**.

The devices upload their episodes to the Hub, then the [SLAM Space](https://huggingface.co/spaces/pollen-robotics/grabette-slam) turns them into the LeRobot dataset. Both steps show their progress and can take several minutes; **Cancel** stops the build.

When it's done, the dataset is on your Hugging Face account: **Visualize this dataset**, on its page, opens it in the [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).

<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/fleet-build-dataset.png" alt="The Build dataset tab of the Fleet Space" width="500" style="display: block; margin: 0 auto;"/>

#### The SLAM Space (advanced)

<Tip>

You don't need to open this Space: the Fleet Space runs it for you when you generate a dataset. This section is for information, or for advanced users who want to run it themselves.

</Tip>

**[pollen-robotics/grabette-slam](https://huggingface.co/spaces/pollen-robotics/grabette-slam)**

The SLAM Space takes a raw Grabette recording that lives on the Hub and gives you back a [LeRobot](https://github.com/huggingface/lerobot) dataset, pushed under your own account. To use it on its own:

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

<video controls preload="metadata" src="https://huggingface.co/spaces/pollen-robotics/grabette-fleet/resolve/main/static/tutorial-episode.mp4"></video>

A policy reproduces your demonstrations, mistakes included, and the trajectory comes from SLAM on the cameras. Six rules:

### 1. Good texture, nothing blocking the cameras

SLAM tracks visual landmarks across frames to compute the Grabette's motion. Choose a scene with texture, and tasks whose objects don't block the cameras from below. Don't let the object fill the view as you grasp and lift it: that is the most common cause of tracking loss.

### 2. Enough contrast and good exposure

The scene is seen by the wide-angle RGB camera on top. Make sure the object stands out from the background, and that it is well lit.

### 3. Stable background

Avoid people or objects moving in the background: the only thing that moves should be the object you manipulate. Keep the lighting stable while you record.

### 4. Start simple, then increase complexity

Start with the same setup, the same object and the same manipulation. After a few episodes, vary the object's position, then the environment — including the lighting across sessions (time of day, blinds, lights): the policy only works in conditions it has seen.

### 5. Fluid movements

- Move smoothly, at a natural rhythm: no jitter, no fast swings.
- Keep one natural grasp angle, within about ±10°.
- Reach decisively, so the object sits deep in the gripper, and open and close the gripper firmly.
- Stay within what the robot can do: modest, central lifts, and a hand speed of about 0.5 m/s at most.

### 6. No failures

Don't record failed manipulations: the policy would learn them. Delete a bad take right away with **Delete last episode** in the [session](#start-a-session).

The [full recording guide](https://github.com/pollen-robotics/grabette-policy/blob/develop/policies/DiffusionPolicy/recording_demonstrations_guide.md) explains each rule and the failure it prevents.
