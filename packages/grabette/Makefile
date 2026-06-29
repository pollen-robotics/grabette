.PHONY: install-rpi install-udev-oak install-systemd install-netdev install-poweroff ensure-ble-only help

# This package is a uv workspace member; the environment uv actually uses
# lives at the workspace ROOT, not here. All venv ops target that root.
WORKSPACE_ROOT := $(abspath $(CURDIR)/../..)

help:
	@echo "Targets:"
	@echo "  install-rpi       Install apt deps + udev rule, create venv, sync, verify"
	@echo "  install-udev-oak  Install only the OAK-D / Movidius udev rule"
	@echo "  install-systemd   Install + enable the grabette + bluetooth systemd services (BLE-only)"
	@echo "  ensure-ble-only   Force BlueZ into BLE-only mode (ControllerMode = le)"
	@echo "  install-netdev    Allow 'rasp' user to trigger WiFi scans via nmcli"
	@echo "  install-poweroff  Allow 'rasp' user to power off the Pi from the web UI"

# One-shot bring-up for a fresh Raspberry Pi (Bookworm or Trixie).
# Handles the apt-package + correct-Python + system-site-packages combo that's
# easy to get wrong by hand and silently falls back to MockBackend on failure.
install-rpi: install-udev-oak
	@command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
	sudo apt update
	sudo apt install -y python3-libcamera python3-picamera2 libcap-dev ffmpeg \
	  python3-dbus python3-gi
	@# python3-dbus / python3-gi: system deps the bluetooth WiFi service imports
	@# (dbus, gi.repository.GLib); the venv reaches them via --system-site-packages.
	cd $(WORKSPACE_ROOT) && rm -rf .venv \
	  && uv venv --python /usr/bin/python3 --system-site-packages \
	  && uv sync --package grabette --extra rpi --extra ui --extra hf
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

# nmcli --rescan yes triggers a real radio scan, gated by two things:
# 1. the 'netdev' group (NM membership check)
# 2. the polkit action org.freedesktop.NetworkManager.wifi.scan
# Both are required on Bookworm/Trixie; the group alone is not sufficient.
POLKIT_RULE := /etc/polkit-1/rules.d/10-grabette-wifi-scan.rules
install-netdev:
	sudo usermod -aG netdev rasp
	@if [ ! -f $(POLKIT_RULE) ]; then \
		echo "Installing polkit rule for WiFi scan..."; \
		printf 'polkit.addRule(function(action, subject) {\n    if (action.id === "org.freedesktop.NetworkManager.wifi.scan" &&\n        subject.isInGroup("netdev")) {\n        return polkit.Result.YES;\n    }\n});\n' \
		  | sudo tee $(POLKIT_RULE) >/dev/null; \
	else \
		echo "Polkit WiFi scan rule already present."; \
	fi
	@echo "Done. A reboot (or 'sudo systemctl restart grabette polkit') is needed."

# Lets the web UI's "Power Off" button halt the Pi. Grants the service user
# the right to run exactly `systemctl poweroff` (nothing else) without a
# password. Validated with `visudo -c` before install so a typo can never lock
# out sudo. Idempotent.
install-poweroff:
	@tmp=$$(mktemp); \
	echo 'rasp ALL=(root) NOPASSWD: /usr/bin/systemctl poweroff' > $$tmp; \
	if sudo visudo -cf $$tmp; then \
		sudo install -m 0440 -o root -g root $$tmp /etc/sudoers.d/grabette-poweroff; \
		echo "Installed /etc/sudoers.d/grabette-poweroff"; \
	else \
		echo "sudoers syntax check failed, aborting"; rm -f $$tmp; exit 1; \
	fi; \
	rm -f $$tmp

# Force BlueZ into BLE-only mode so the bluetooth WiFi service advertises cleanly
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
