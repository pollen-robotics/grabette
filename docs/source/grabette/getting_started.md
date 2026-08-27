# Getting Started

## 1. Assembly

Grabette can be made at home with a 3D printer and a few off-the-shelf components. The CAD files are open-source, and the assembly is straightforward.

-  1. **Buy** all the components you need thanks to the **[Bill of Materials](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3LyyWI-CiplVPtgrWkmLRYjdDqYhbVJXYt8PNa71FDzbTSMVj1YGV0Zpo5PJeBGJURaz8nZt1_v-8/pubhtml)**
- 2. **Print** the elements with the **[CAD Files](https://github.com/pollen-robotics/grabette/tree/develop/packages/grabette/assembly/CAD_files)** and the **[3D Print Guide](https://github.com/pollen-robotics/grabette/blob/develop/packages/grabette/assembly/Grabette_3DPrint_Guide.pdf)**
- 3. **Build** your Grabette thanks to the **[Assembly guide](https://github.com/pollen-robotics/grabette/blob/develop/packages/grabette/assembly/Grabette_Assembly.pdf)** 


## 2. Install the software on the Raspberry Pi

<details><summary><b>Prerequisites : install the OS on the SD Card</b></summary>
<ol>
    <li> Download the <b><a href="https://www.raspberrypi.com/software/">Raspberry Pi Imager</a></b> and install it on your computer.</li>
    <li> Insert the SD card into your computer.</li>
    <li> Open the Raspberry Pi Imager and select **Raspberry Pi 4** as the target device.</li>
    <li> Select **Raspberry Pi OS Lite (64-bit)** as the operating system</li>
    <li> Click on the **Settings** icon and set a hostname (for example `r-grabette`), a user and password, your WiFi credentials, and enable SSH.</li>
    <li> Click on **Write** to flash the SD card.</li>
</ol>
</details>

VOIR POUR INSTALLATION DE L'ISO ET LANCEMENT DES SERVICES


## 3. Wi-Fi configuration

If the WiFi you set at flash time isn't the one you need, you can configure the wireless connexion over Bluetooth.

Open the [Bluetooth tool](https://pollen-robotics.github.io/grabette/). in Chrome or Edge, connect to the device, enter the PIN (`00000` by default), then scan and pick your network. 


## 4. Angle calibration

Connect to your Grabette via ssh : 

```bash
ssh <user>@<hostname>.local
```

Open the gripper so that **both joints are fully extended**, then :

```bash
cd grabette/packages/grabette
uv run python scripts/calibrate_angles.py
sudo reboot
```

Check the result in the dashboard's **Live View**: the finger-joint angles should move smoothly and reach their expected range when you open and close the gripper.


## 5. What's next

Now that your Grabette is assembled, configured and calibrated, you can start recording episodes. See [Usage](./usage.md) for the day-to-day operation of your device.


