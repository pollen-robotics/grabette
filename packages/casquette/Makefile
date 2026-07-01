.PHONY: install-rpi enable-i2c install-systemd check help

# This package is a uv workspace member; the environment uv actually uses
# lives at the workspace ROOT, not here. All venv ops target that root.
WORKSPACE_ROOT := $(abspath $(CURDIR)/../..)

help:
	@echo "Targets:"
	@echo "  install-rpi       One-shot bring-up: apt deps + I2C enable + venv + sync + verify"
	@echo "  enable-i2c        Enable hardware I2C bus 1 (GPIO 2/3) via raspi-config nonint"
	@echo "  install-systemd   Install + enable casquette.service and casquette-bluetooth.service"
	@echo "  check             Run scripts/check_hardware.py against current install"
	@echo ""
	@echo "Prereqs before 'install-rpi':"
	@echo "  - uv installed (https://docs.astral.sh/uv/)"
	@echo "  - the grabette monorepo cloned on this device (any path; install-systemd"
	@echo "    patches the unit to this device's workspace root automatically)"

# One-shot bring-up for a fresh Raspberry Pi (Bookworm or Trixie).
# Mirrors grabette/Makefile's install-rpi but for casquette's lighter stack
# (no OAK-D, no Gradio, no scipy). Idempotent — safe to re-run.
install-rpi: enable-i2c
	@command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
	sudo apt update
	sudo apt install -y python3-libcamera python3-picamera2 libcap-dev ffmpeg i2c-tools git-lfs
	cd $(WORKSPACE_ROOT) && rm -rf .venv \
	  && uv venv --python /usr/bin/python3 --system-site-packages \
	  && uv sync --package casquette --extra rpi
	@echo "--- verifying imports ---"
	@$(WORKSPACE_ROOT)/.venv/bin/python -c "import picamera2, numpy, adafruit_extended_bus; print('all imports OK')"
	@echo "--- pyvenv.cfg ---"
	@grep -E "home|version_info|system-site" $(WORKSPACE_ROOT)/.venv/pyvenv.cfg
	@echo
	@echo "Done. Plug the BMI088 into the HAT's I2C port (GPIO 2/3) and run 'make check'."
	@echo "Then 'uv run --package casquette python -m casquette' to start the daemon, or 'make install-systemd' for boot-time start."

# Enable hardware I2C bus 1 (GPIO 2/3) — the conventional Pi I2C bus where
# the HAT exposes its 'I2C' port. raspi-config handles both /boot/config.txt
# (Bullseye) and /boot/firmware/config.txt (Bookworm+) and triggers the
# i2c-dev module on next boot. A reboot is required for /dev/i2c-1 to appear.
enable-i2c:
	@if [ ! -e /dev/i2c-1 ]; then \
		echo "Enabling hardware I2C bus 1 via raspi-config..."; \
		sudo raspi-config nonint do_i2c 0; \
		echo ""; \
		echo "*** REBOOT REQUIRED — /dev/i2c-1 will appear after next boot. ***"; \
	else \
		echo "I2C bus 1 already enabled (/dev/i2c-1 present)."; \
	fi

# Install both casquette services. Run after install-rpi has verified the
# venv builds. The bluetooth service needs root (BlueZ DBus access); the
# main service runs as rasp.
install-systemd:
	@# Patch the units' hardcoded path to this device's actual workspace root
	@# (where the venv lives) so boot-time start uses the right interpreter.
	sed 's#/home/rasp/Project/Repo/casquette#$(WORKSPACE_ROOT)#g' systemd/casquette.service \
	  | sudo tee /etc/systemd/system/casquette.service >/dev/null
	sed 's#/home/rasp/Project/Repo/casquette#$(WORKSPACE_ROOT)#g' systemd/casquette-bluetooth.service \
	  | sudo tee /etc/systemd/system/casquette-bluetooth.service >/dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable --now casquette
	sudo systemctl enable --now casquette-bluetooth
	@echo "Logs:"
	@echo "  journalctl -u casquette -f"
	@echo "  journalctl -u casquette-bluetooth -f"

check:
	@if [ ! -x $(WORKSPACE_ROOT)/.venv/bin/python ]; then \
		echo "No $(WORKSPACE_ROOT)/.venv/bin/python found. Run 'make install-rpi' first —"; \
		echo "the venv must be created with --system-site-packages so apt's"; \
		echo "picamera2 satisfies the dependency tree (otherwise uv tries to"; \
		echo "build python-prctl from PyPI, which needs libcap-dev)."; \
		exit 1; \
	fi
	$(WORKSPACE_ROOT)/.venv/bin/python scripts/check_hardware.py
