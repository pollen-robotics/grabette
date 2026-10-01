# FAQ

<Tip warning={true}>

This page is a starting point, seeded from the questions the repository's own documentation already answers. If you hit something that isn't here, [open an issue](https://github.com/pollen-robotics/grabette/issues) — that's how this page grows.

</Tip>

## Do I need a robot to collect data?

No. You demonstrate the task with your hand, holding the Grabette. The recording contains the camera trajectory and the finger-joint angles, and carries no assumption about which arm will eventually replay it. A robot only enters the picture at training and deployment time.

## Do I need a depth camera?

On **Grabette**, yes. Trajectory recovery uses the depth stream; without it there is no SLAM and therefore no dataset. The default is the Orbbec Gemini 305, which has no IMU, so SLAM runs IMU-free. A Luxonis OAK-D SR also works (`make install-rpi CAMERA=oakd`), and its IMU is then used too. The camera is off by default to save battery: a recording turns it on automatically, and it powers down about 30 s after the recording stops.

On **Gripette** a depth camera is optional — the standard motor-and-camera service doesn't need it.

## Why does `make install-rpi` insist on `HAND=`?

A device is built as a left or a right hand, and the angle sensors are mounted mirrored between the two. The daemon has to know which it is to interpret them. The value is written to `/etc/grabette/env` (`/etc/gripette/env` on a Gripette) and persists across reboots.

## I re-ran `install-rpi` and my OAK-D Grabette stopped working

`CAMERA` defaults to `gemini305`, and `install-rpi` writes the setting rather than reading the previous one. Re-running it without `CAMERA=oakd` switches the device to the Gemini. Re-run `make install-rpi HAND=... CAMERA=oakd`.

## I can't reach `http://<hostname>.local:8000`

`.local` resolution needs mDNS, which some networks and some corporate laptops block. Use the device's IP address instead. If the device isn't on the network at all, provision WiFi over Bluetooth with the [BT tool](https://pollen-robotics.github.io/grabette/).

## The Bluetooth tool won't connect

It relies on Web Bluetooth, so it needs Chrome or Edge on a computer or an Android phone — Safari and Firefox won't work, and on iPhone or iPad no browser does, Chrome included. Chrome may also need `chrome://flags/#enable-experimental-web-platform-features` enabled, notably on Linux.

If a connection hangs at pairing, the stale bond is on your computer — the device clears its own side automatically. Remove the pairing in your Bluetooth settings (on Linux: `bluetoothctl devices | grep -i grabette | cut -d' ' -f2 | xargs -rn1 bluetoothctl remove`), and *Forget* the device in `chrome://bluetooth-internals`.

## On Linux, the Bluetooth tool keeps saying "GATT Server is disconnected"

The first connection to a device always triggers a Bluetooth pairing, and a Linux desktop without a registered pairing agent rejects it, so every attempt drops after a fraction of a second. Pair once from a terminal:

```bash
bluetoothctl
# then, inside bluetoothctl:
agent on
default-agent
scan on          # wait for the device to appear, note its address
pair <device-mac>
```

Once bonded, Chrome connects without pairing again. Windows and macOS ship a pairing agent, so this only affects Linux.

## Recordings from two devices don't line up

Synchronized group recording works by having every device wait out a shared UTC instant *on its own clock*, so a clock offset between devices becomes an offset between recordings. `make install-ntp`, which `install-rpi` runs for you, pins all devices to a single coordinated time service instead of the default pool, where each device would otherwise land on a different physical server. Check it with `timedatectl timesync-status`. For sub-millisecond sync, run an NTP server on the LAN and point the devices at it.

## Can I use Grabette with my own robot?

Yes — that's the design. The core packages carry no dependency on any particular arm; the OpenArm integration in the repository is a worked example, not a requirement. To target another platform, add your own integration alongside it.
