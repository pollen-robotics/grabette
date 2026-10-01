# Usage

## Power on and off

To power on, press the button once, then press it a second time and hold it until the blue light appears. Give the device a moment to start up.

To power off, use the **Power off** button in the [dashboard](#the-dashboard)'s title bar, or power the device off from the [Fleet Space](./data_collection.md#introduction-to-grabette-fleet).

<!-- TODO: document powering off with the physical button, if supported. -->

## Charging

<!-- TODO: how to charge the device (connector, charge time, battery indicator). -->

The battery level is shown on the dashboard's **Overview** page.

## The dashboard

Every Grabette serves a web dashboard on port **8000**. It is how you check the device, make a test recording and review episodes — no SSH, no command line.

```
http://<hostname>.local:8000     # e.g. http://R-grabette.local:8000
```

The device's IP address works too, and is shown on the **Overview** and **Network** pages if `.local` name resolution isn't available on your network.

<!-- TODO(after 170/171 merge): add a screenshot of the Overview page. -->

Every page has a **Power off** button in the title bar, the clean way to shut the device down.

### Overview

The landing page.

- **Cameras** — the live image, with an RGB / Depth toggle. Choosing Depth turns the depth camera on if it's off.
- **3D model** — the gripper, moving with the live finger-joint angles. Open and close it to check the angle sensors.
- **Device** — hostname, side (left or right hand, as installed), IP address and WiFi network, with a **Change network** shortcut.
- **Health** — battery, temperature and storage used.
- **Grabette data** — shortcuts to **Test Recording** and **Episodes**.
- **Account & fleet** — the Hugging Face sign-in, and the **Fleet Space** button to record data and create datasets.

### Signing in to Hugging Face

Click the Hugging Face button to sign in with your account (OAuth, one click). Pasting an access token is available as a fallback under *or use a token*; create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with write access to the datasets you intend to push.

The Fleet Space and every upload depend on this login.

### Test Recording

A guided first recording, to check the whole device before a real session:

1. **Record a few seconds** — press the button, pick an object up, press again to stop.
2. **Check what was recorded** — a summary, a replay of the RGB and depth video with the angle chart, and a download link.
3. **Delete it** — a test recording is not training data. If you keep it, it is filed under the *Unassigned* task in **Episodes**.
4. **Record real data** — sign in to Hugging Face if needed, and open the Fleet Space.

<!-- TODO(after 170/171 merge): add a screenshot of the Test Recording page. -->

### Episodes

Where you review what is on the device.

- **Tasks** — pick a task to list its episodes. Tasks are created in the Fleet Space, not here.
- **Episodes table** — duration, frame count, angles and status for each episode.
- **Replay** — the video with the angle chart and a timeline.
- **Check** — a full verdict on the episode's files and quality. Deleting a bad take now is much cheaper than discovering it after SLAM.
- **Download** — the episode as an archive.
- **Delete**, and **Delete all Grabette data** at the bottom of the page.

Recordings are written to `~/grabette-data/` on the device.

<!-- TODO(after 170/171 merge): add a screenshot of the Episodes page. -->

### Network

The device card again, and **Change WiFi network** to see the current connection and switch networks. The [Bluetooth tool](https://pollen-robotics.github.io/grabette/#wifi) is the fallback when the device isn't reachable at all.

## Start and stop a recording

Press the button to start a recording, and press it again to stop. The dashboard has no record button: the button is the only trigger on the device, so you keep both hands on the task. For organized recording — tasks, sessions, several devices at once — use the [Fleet Space](./data_collection.md).

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
