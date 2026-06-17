.PHONY: install-rpi install-udev-oak install-systemd install-netdev help

# This package is a uv workspace member; the environment uv actually uses
# lives at the workspace ROOT, not here. All venv ops target that root.
WORKSPACE_ROOT := $(abspath $(CURDIR)/../..)

help:
	@echo "Targets:"
	@echo "  install-rpi       Install apt deps + udev rule, create venv, sync, verify"
	@echo "  install-udev-oak  Install only the OAK-D / Movidius udev rule"
	@echo "  install-systemd   Install + enable the grabette systemd service"
	@echo "  install-netdev    Allow 'rasp' user to trigger WiFi scans via nmcli"

# One-shot bring-up for a fresh Raspberry Pi (Bookworm or Trixie).
# Handles the apt-package + correct-Python + system-site-packages combo that's
# easy to get wrong by hand and silently falls back to MockBackend on failure.
install-rpi: install-udev-oak
	@command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
	sudo apt update
	sudo apt install -y python3-libcamera python3-picamera2 libcap-dev ffmpeg
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

# nmcli --rescan yes triggers a real radio scan, which NetworkManager restricts
# to members of the 'netdev' group. Without this, the scan returns empty or errors
# on machines where the service user doesn't already have wifi management rights.
install-netdev:
	sudo usermod -aG netdev rasp
	@echo "Group change applied. A reboot (or 'sudo systemctl restart grabette') is needed."

install-systemd:
	@# Patch the unit's hardcoded path to this device's actual workspace root
	@# (where the venv lives) so boot-time start uses the right interpreter.
	sed 's#/home/rasp/Project/Repo/grabette#$(WORKSPACE_ROOT)#g' systemd/grabette.service \
	  | sudo tee /etc/systemd/system/grabette.service >/dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable --now grabette
	@echo "Logs: journalctl -u grabette -f"
