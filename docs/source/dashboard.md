# The dashboard

Every Grabette serves a web dashboard on port **8000**. It is how you record, review and publish episodes — no SSH, no command line.

```
http://<hostname>.local:8000     # e.g. http://R-grabette.local:8000
http://localhost:8000            # in mock mode on your laptop
```

The device's IP address works too, and is shown on the **Home** page if `.local` name resolution isn't available on your network.

![The Grabette dashboard](https://github.com/pollen-robotics/grabette/raw/develop/packages/grabette/docs/images/grabette-dashboard.png)

## Home

Where you land, and where the device says what it is.

- **Essentials** — battery, hostname, IP address and the network it is on, in one strip at the top. Refreshed every 30 s.
- **Network** — **Switch network** folds out the network panel: the networks this Grabette already knows are listed first and switch in one click, no password; below them, any other network in range can be added by picking it and typing its password. A network is "known" however it was set up — from here or over Bluetooth.
- **Bluetooth tool** — opens the [Bluetooth provisioning tool](https://pollen-robotics.github.io/grabette/) in a new tab. That is the way in when the device is on a network you cannot reach at all, so this page won't load either — worth knowing where it is *before* you need it. Chrome or Edge only (Web Bluetooth).
- **Hugging Face** — paste an access token to log the device in. Everything under **Datasets**, and the fleet grouping described in [Hugging Face Spaces](./spaces.md), depends on this. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with write access to the datasets you intend to push.
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

A clean shutdown of the Raspberry Pi, with the device's hostname printed on the button's card — several grabettes open in several tabs look identical otherwise. The button is disabled while a recording is running.

## Gripette's status page

Gripette — the robot-mounted gripper — is a gRPC service and has no dashboard of its own, but it does have a small status page on port **8080**, installed with `make install-web`. It reports whether the service is healthy, shows the live camera at about 1 Hz, and gives you **Restart service** and **Shut down device** buttons. That shutdown button is the clean way to power off a Gripette, which has no power switch.

<Tip warning={true}>

The Gripette status page has no authentication: anyone on the local network can restart or shut down the device.

</Tip>
