.PHONY: install-rpi install-udev-oak install-systemd ensure-ble-only help

# This package is a uv workspace member; the environment uv actually uses
# lives at the workspace ROOT, not here. All venv ops target that root.
WORKSPACE_ROOT := $(abspath $(CURDIR)/../..)

help:
	@echo "Targets:"
	@echo "  install-rpi       Install apt deps + udev rule, create venv, sync, verify"
	@echo "  install-udev-oak  Install only the OAK-D / Movidius udev rule"
	@echo "  install-systemd   Install + enable the grabette + bluetooth systemd services (BLE-only)"
	@echo "  ensure-ble-only   Force BlueZ into BLE-only mode (ControllerMode = le)"

# One-shot bring-up for a fresh Raspberry Pi (Bookworm or Trixie).
# Handles the apt-package + correct-Python + system-site-packages combo that's
# easy to get wrong by hand and silently falls back to MockBackend on failure.
install-rpi: install-udev-oak
	@command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
	sudo apt update
	sudo apt install -y python3-libcamera python3-picamera2 libcap-dev ffmpeg
	cd $(WORKSPACE_ROOT) && rm -rf .venv \
	  && uv venv --python /usr/bin/python3 --system-site-packages \
	  && uv sync --package grabette --extra rpi --extra ui
	@echo "--- verifying imports ---"
	@$(WORKSPACE_ROOT)/.venv/bin/python -c "import picamera2, depthai, cv2, gpiod, numpy; print('all imports OK')"
	@echo "--- pyvenv.cfg ---"
	@grep -E "home|version_info|system-site" $(WORKSPACE_ROOT)/.venv/pyvenv.cfg
	@echo
	@echo "Done. Run 'uv run --package grabette python -m grabette' to start the daemon."
	@echo "Watch journalctl for 'RPi hardware detected, using RpiBackend' (NOT MockBackend)."

# Movidius / OAK-D USB access rule. Without this, depthai needs sudo to talk
# to the device. Idempotent.
install-udev-oak:
	@if [ ! -f /etc/udev/rules.d/80-movidius.rules ]; then \
		echo "Installing OAK-D udev rule..."; \
		echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules >/dev/null; \
		sudo udevadm control --reload-rules; \
		sudo udevadm trigger; \
	else \
		echo "OAK-D udev rule already present."; \
	fi

# Force BlueZ into BLE-only mode so the bluetooth WiFi service advertises
# cleanly: in dual-mode, classic BR/EDR audio profiles (A2DP/Handsfree/AVRCP)
# get negotiated on connect and destabilise the LE link (writes fail, links
# drop). Idempotent — replaces any existing ControllerMode line (even a
# commented one) or adds one under [General], then restarts bluetoothd. The
# setting lives in main.conf so it persists across reboots.
ensure-ble-only:
	@CONF=/etc/bluetooth/main.conf; \
	sudo grep -q '^\[General\]' "$$CONF" || echo '[General]' | sudo tee -a "$$CONF" >/dev/null; \
	if sudo grep -qE '^[[:space:]]*#?[[:space:]]*ControllerMode' "$$CONF"; then \
	  sudo sed -i -E 's|^[[:space:]]*#?[[:space:]]*ControllerMode.*|ControllerMode = le|' "$$CONF"; \
	else \
	  sudo sed -i '/^\[General\]/a ControllerMode = le' "$$CONF"; \
	fi
	sudo systemctl restart bluetooth
	@echo "BlueZ set to BLE-only (ControllerMode = le)"

install-systemd: ensure-ble-only
	@# Patch each unit's placeholder path to this device's actual workspace root
	@# (where the venv lives) so boot-time start uses the right interpreter.
	sed 's#/home/rasp/Project/Repo/grabette#$(WORKSPACE_ROOT)#g' systemd/grabette.service \
	  | sudo tee /etc/systemd/system/grabette.service >/dev/null
	sed 's#/home/rasp/Project/Repo/grabette#$(WORKSPACE_ROOT)#g' systemd/grabette-bluetooth.service \
	  | sudo tee /etc/systemd/system/grabette-bluetooth.service >/dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable --now grabette
	sudo systemctl enable --now grabette-bluetooth
	@echo "Logs: journalctl -u grabette -f"
	@echo "      journalctl -u grabette-bluetooth -f"
