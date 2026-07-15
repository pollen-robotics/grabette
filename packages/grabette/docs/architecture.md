# Grabette — Architecture

Internal architecture of the on-device Grabette daemon (see the
[README](../README.md) for install and usage).

```
                       ┌──────────────────────────┐
                       │   Web UI (Gradio)         │
                       │   HuggingFace Spaces      │
                       └────────────┬─────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────┐
│                    FastAPI + WebSocket API (:8000)                    │
│                                                                       │
│  /api/state     Live sensor polling + WS stream @10Hz                │
│  /api/camera    JPEG snapshot + WS video stream ~15fps               │
│  /api/episodes  Capture start/stop, download, delete                 │
│  /api/sessions  Session CRUD, episode grouping                       │
│  /api/replay    Episode playback with pause/seek                     │
│  /api/hf        HuggingFace auth, upload, SLAM jobs                  │
│  /api/system    System info, logs, OTA updates                       │
│  /api/daemon    Daemon status + restart                              │
│  /viewer        3D URDF model with live joint angles (Three.js)      │
│  /charts/*      Real-time IMU + angle charts (uPlot)                 │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────┐
│                         Daemon Core                                   │
│          State machine · 50Hz poll loop · Replay engine               │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                   RpiBackend              MockBackend
                  (real hardware)          (development)
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     VideoCapture   OakdCapture   AngleCapture
     (picamera2)    (RGB-D + IMU,  (AS5600L, I2C)
                     toggleable)
```
