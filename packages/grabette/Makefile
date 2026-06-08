.PHONY: install-rpi install-udev-oak install-systemd help

help:
	@echo "Targets:"
	@echo "  install-rpi       Install apt deps + udev rule, create venv, sync, verify"
	@echo "  install-udev-oak  Install only the OAK-D / Movidius udev rule"
	@echo "  install-systemd   Install + enable the grabette systemd service"

# One-shot bring-up for a fresh Raspberry Pi (Bookworm or Trixie).
# Handles the apt-package + correct-Python + system-site-packages combo that's
# easy to get wrong by hand and silently falls back to MockBackend on failure.
install-rpi: install-udev-oak
	@command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
	sudo apt update
	sudo apt install -y python3-libcamera python3-picamera2 libcap-dev ffmpeg
	rm -rf .venv
	uv venv --python /usr/bin/python3 --system-site-packages
	uv sync --extra rpi --extra ui
	@echo "--- verifying imports ---"
	@.venv/bin/python -c "import picamera2, depthai, cv2, gpiod, numpy; print('all imports OK')"
	@echo "--- pyvenv.cfg ---"
	@grep -E "home|version_info|system-site" .venv/pyvenv.cfg
	@echo
	@echo "Done. Run 'uv run python -m grabette' to start the daemon."
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

install-systemd:
	sudo cp systemd/grabette.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable --now grabette
	@echo "Logs: journalctl -u grabette -f"
