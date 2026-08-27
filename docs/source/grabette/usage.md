# Usage

## How to turn it on / off

### To turn it on

<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/turn_on.gif" width="40%"/><br>

- Press the power button once 
- Then press and hold the power button until the blue LED light up (about 2 seconds)

### To turn it off

You can either : 
- Press and hold the power button until the blue LED turn off (about 5 seconds)
- Use the dashboard (see below) to shut down the device safely : 
    - Reach `<hostname>.local:8000` in a browser
    - Go to **Power Off** and click **Power off now**. The device will power off after a few seconds.


## How to charge it

Use an USB-C cable to connect the device to a power source. :
<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/usb_c_port_rasp.png" width="40%"/> <br> 


## Getting to the dashboard

Every Grabette serves a web dashboard on port **8000**. It enables you to authenticate on your Hugging Face account, to check that every sensors is working, to replay episodes or to access to the settings of your device. 


### Accessing the dashboard

On a computer connected to the same WiFi network as the device, open a browser and go to either of these URLs:

```
http://<hostname>.local:8000     # e.g. http://r-grabette.local:8000/
http://<ip_address>:8000         # e.g. http://192.168.10.127:8000/
```

<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/grabette_dashboard.png" width="50%"/>

### Dashboard sections

The dashboard has five sections:

- **Connection** : allows you to authenticate on your Hugging Face account, to be able to reach the Fleet Dashboard (see next). 
- **Episodes** — record, replay and delete takes. 
- **Live View** — preview the cameras and watch the sensor charts in real time. 
- **Settings** — device info (hostname, current network, IP address) and Hugging Face login. 
- **Power Off** — shut down the device safely. The device will power off after a few seconds.

## Recording an episode

You can directly record an episode with the Grabette, by pressing the physical button on the device, do you task and then press the button again to stop the recording. This episode will be saved in the _Unassigned_ task. 

<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/start_recording.gif" align='left' width="40%"/>
<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/stop_recording.gif" align='right' width="40%"/> <br>


<br clear="left"/>

>[!IMPORTANT]
>The OAK-D SR can take time to warm up at the first recording of a session, you need to wait for the red LED to stabilize before starting your task. </tip>





>[!NOTE]
>Recordings are written to `~/grabette-data/` on the device.

## What's next 

To collect your data in a more organized way—by creating specific tasks and conducting recording sessions—continue to the next page, which covers **Grabette-Fleet**. 

