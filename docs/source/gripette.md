# Gripette

Gripette is the robot-mounted version of Grabette: the same fingers, driven by two Feetech STS3215 servos, so a robot can reproduce what you demonstrated. It runs a gRPC service on a Raspberry Pi Zero 2W that streams camera frames at about 10 Hz, synchronized with the motor positions, and accepts motor commands over the network.

## Assembly

Gripette shares the [Bill of Materials](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3LyyWI-CiplVPtgrWkmLRYjdDqYhbVJXYt8PNa71FDzbTSMVj1YGV0Zpo5PJeBGJURaz8nZt1_v-8/pubhtml) and the [CAD](https://cad.onshape.com/documents/0c6175c392788391992ff2ec/w/9f773e5f0eeae1577ae36a05/e/13a89fef2591d863bb0bf186) with Grabette. Build it with the [assembly guide](https://github.com/pollen-robotics/grabette/blob/develop/packages/gripette/assembly/Gripette_Assembly.pdf) and the 3D-print guide in the same folder.

## Install

Download `image_<date>-gripetteos-<version>.zip` from the [latest grabette-os release](https://github.com/pollen-robotics/grabette-os/releases/latest) and flash it as for [Grabette](./build_your_own.md#flash-the-raspberry-pi): no Imager settings, WiFi through the Bluetooth tool afterwards, SSH as `pollen` (password `root`). The image has the dependencies, the UART setup and the `gripette` + `gripette-bluetooth` services already in place.

The motors are mounted mirrored, so the hand is read from the **hostname**, which must contain `left` or `right`. The device boots as `gripette-<serial digits>`, and the service waits for a hand until you SSH in and run:

```bash
sudo hostnamectl set-hostname gripette-right-1f9a2b   # or gripette-left-…
sudo systemctl restart gripette
```

The [Bluetooth tool](https://pollen-robotics.github.io/grabette/#wifi) provisions a Gripette's WiFi the same way as a Grabette's — pick Gripette in its device chooser.

Before first use, calibrate the encoders' zero offset: see [calibration](https://github.com/pollen-robotics/grabette/blob/develop/packages/gripette/docs/calibration.md).

The [Gripette README](https://github.com/pollen-robotics/grabette/tree/develop/packages/gripette) covers the gRPC client, the CLI scripts and the motor setup.

### Manual install (stock Raspberry Pi OS)

Only if you don't use the image. Flash **Raspberry Pi OS Lite (64-bit)** for the Pi Zero 2W, as in the [Grabette manual install](./build_your_own.md#manual-install-stock-raspberry-pi-os), then on the Pi:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/pollen-robotics/grabette.git
cd grabette/packages/gripette

sudo usermod -aG dialout $USER   # serial bus access — log out and back in
make install-rpi HAND=right      # or HAND=left: the motors are mounted mirrored
sudo reboot
make check                       # camera and motor diagnostic
make install-systemd             # start the service and Bluetooth WiFi setup on boot
```

## Status page

Gripette has a dashboard on port **8000** (as on Grabette), laid out like the Grabette dashboard's Overview and installed with the services by `make install-systemd` (`make install-web` alone on a device set up before; on the image: `sudo systemctl enable --now gripette-web`). It reports whether the service is healthy, shows the live camera at about 1 Hz and a 3D model that follows the motors, the device and health cards, and gives you **Restart service** and **Power off** buttons. That power-off button is the clean way to shut down a Gripette, which has no power switch.

<Tip warning={true}>

The status page has no authentication: anyone on the local network can restart or shut down the device.

</Tip>
