# The dashboard

Every Grabette serves a web dashboard on port **8000**. It is how you record, review and publish episodes — no SSH, no command line.

```
http://<hostname>.local:8000     # e.g. http://R-grabette.local:8000
http://localhost:8000            # in mock mode on your laptop
```

The device's IP address works too, and is shown on the **Home** page if `.local` name resolution isn't available on your network. The dashboard is laid out to be usable on a phone — which is what you will have on you next to the robot.

![The Grabette dashboard](https://github.com/pollen-robotics/grabette/raw/develop/packages/grabette/docs/images/grabette-dashboard.png)

## The page header

Every page opens with the same band: the **device's hostname** on the left, its **battery** on the right, and the theme switch beside it. Same content, same position, whichever page you are on — with several Grabettes open in several tabs, that is how you tell them apart.

Just under it, on the right, a small **Light / Dark** switch. The choice is remembered in the browser and applies to every page; with no choice made, the dashboard follows the operating system's setting. (`?__theme=light` or `?__theme=dark` in the URL overrides both, for a bookmark or a kiosk.)

## Home

Where you land, and where the device says what it is.

- **Network** — the network this Grabette is on, and its IP address.
- **Switch network** folds out one panel with everything network in it:
  - **Known networks** — every network this device already has credentials for. One click moves it to any of them, no password. A network is known however it was set up: from a previous switch, or over Bluetooth.
  - **Join a new network (Bluetooth tool)** — opens the [Bluetooth provisioning tool](https://pollen-robotics.github.io/grabette/) in a new tab. Joining a network the device has never seen is that tool's job, because it works when the device is on no network you can reach — which is exactly when this page won't load either. Worth knowing where it is *before* you need it. Chrome or Edge only (Web Bluetooth). The URL is the `bt_tool_url` setting, so a fork can point at its own deployment.
- **Speaker** — the volume of the recording cues (the beeps that mark the start and end of a take). Drag the slider and press **Test sound** to hear it. The level is remembered on the device and survives a restart. A Grabette built without a speaker is a supported configuration, not a fault: the control is greyed out and says so, and a level set anyway is kept in case one is fitted later.
- **Hugging Face account** — paste an access token to log the device in. Everything under **Datasets**, and the fleet grouping described in [Hugging Face Spaces](./spaces.md), depends on this. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with write access to the datasets you intend to push.
- **Open fleet dashboard** — links out to grabette-fleet.

## Episodes

Where recording happens.

- **Record.** Start and stop an episode. The physical button on the device does exactly the same thing, so you can keep both hands on the object and never touch the screen.
- **Tasks.** Create a task — the natural-language description of what is being demonstrated, such as *"pick up the cup"*. It follows the episode all the way into the LeRobot dataset.
- **Sessions.** Group a run of episodes recording the same task, so you can record twenty demonstrations without re-typing anything.
- **Review.** Replay a captured episode, and delete the ones that went wrong. Doing this now is much cheaper than discovering a bad take after SLAM.

Recordings are written to `~/grabette-data/` on the device.

## Datasets

Where episodes leave the device.

Upload the recordings to a Hugging Face dataset repository, and trigger post-processing — SLAM followed by LeRobot dataset generation. This is the on-device entry point to the same pipeline described in [Getting started](./get_started.md#turn-recordings-into-a-lerobot-dataset); the heavy lifting runs in the [SLAM Space](./spaces.md#grabette-slam--lerobot).

You need to be logged in to Hugging Face first — see **Home** above.

## Live View

Preview the cameras and watch the sensor charts in real time.

Use it before a recording session to confirm the framing, that the OAK-D is on and streaming depth, and that the finger-joint angles move as expected when you open and close the gripper. A minute here saves a session's worth of unusable episodes.

## Power Off

A clean shutdown of the Raspberry Pi. The card names the device it is about to shut down, right where the button is. The button is disabled while a recording is running.

## Gripette's status page

Gripette — the robot-mounted gripper — is a gRPC service and has no dashboard of its own, but it does have a small status page on port **8080**, installed with `make install-web`. It reports whether the service is healthy, shows the live camera at about 1 Hz, and gives you **Restart service** and **Shut down device** buttons. That shutdown button is the clean way to power off a Gripette, which has no power switch.

<Tip warning={true}>

The Gripette status page has no authentication: anyone on the local network can restart or shut down the device.

</Tip>
