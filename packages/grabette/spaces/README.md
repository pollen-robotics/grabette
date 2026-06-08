---
title: Grabette
emoji: "\U0001F916"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
---

# Grabette

Remote dashboard for the Grabette robotic manipulation data collection system.

## Setup

1. Start the grabette service on your RPi
2. Expose it via tunnel: `cloudflared tunnel --url http://localhost:8000`
3. Set the tunnel URL as the `GRABETTE_API_URL` Space secret, or enter it in the UI
