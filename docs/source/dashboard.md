# The dashboard

Every Grabette serves a web dashboard on port **8000**. It is how you check the device, make a test recording and review episodes — no SSH, no command line.

```
http://<hostname>.local:8000     # e.g. http://R-grabette.local:8000
http://localhost:8000            # in mock mode on your laptop
```

The device's IP address works too, and is shown on the **Overview** and **Network** pages if `.local` name resolution isn't available on your network.

<!-- TODO(after 170/171 merge): add a screenshot of the Overview page. -->

The dashboard doesn't start recordings: the **physical button** on the device does. Organized recording — tasks, sessions, several devices at once — and dataset creation happen in the [Fleet Space](./spaces.md#grabette-fleet), which the dashboard links to.

Every page has a **Power off** button in the title bar, the clean way to shut the device down.

## Overview

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

## Test Recording

A guided first recording, to check the whole device before a real session:

1. **Record a few seconds** — press the button, pick an object up, press again to stop.
2. **Check what was recorded** — a summary, a replay of the RGB and depth video with the angle chart, and a download link.
3. **Delete it** — a test recording is not training data. If you keep it, it is filed under the *Unassigned* task in **Episodes**.
4. **Record real data** — sign in to Hugging Face if needed, and open the Fleet Space.

<!-- TODO(after 170/171 merge): add a screenshot of the Test Recording page. -->

## Episodes

Where you review what is on the device.

- **Tasks** — pick a task to list its episodes. Tasks are created in the Fleet Space, not here.
- **Episodes table** — duration, frame count, angles and status for each episode.
- **Replay** — the video with the angle chart and a timeline.
- **Check** — a full verdict on the episode's files and quality. Deleting a bad take now is much cheaper than discovering it after SLAM.
- **Download** — the episode as an archive.
- **Delete**, and **Delete all Grabette data** at the bottom of the page.

Recordings are written to `~/grabette-data/` on the device.

<!-- TODO(after 170/171 merge): add a screenshot of the Episodes page. -->

## Network

The device card again, and **Change WiFi network** to see the current connection and switch networks. The [Bluetooth tool](https://pollen-robotics.github.io/grabette/#wifi) is the fallback when the device isn't reachable at all.

## Gripette's status page

Gripette — the robot-mounted gripper — is a gRPC service and has no dashboard of its own, but it does have a small status page on port **8080**, installed with `make install-web`. It reports whether the service is healthy, shows the live camera at about 1 Hz, and gives you **Restart service** and **Shut down device** buttons. That shutdown button is the clean way to power off a Gripette, which has no power switch.

<Tip warning={true}>

The Gripette status page has no authentication: anyone on the local network can restart or shut down the device.

</Tip>
