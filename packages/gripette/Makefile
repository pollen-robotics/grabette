.PHONY: install-rpi enable-uart install-systemd ensure-ble-only check help

# This package is a uv workspace member; the environment uv actually uses
# lives at the workspace ROOT, not here. All venv ops target that root.
WORKSPACE_ROOT := $(abspath $(CURDIR)/../..)

help:
	@echo "Targets:"
	@echo "  install-rpi       One-shot bring-up: apt deps + UART enable + venv + sync + verify"
	@echo "  enable-uart       Disable serial console + enable hardware UART (/dev/serial0)"
	@echo "  install-systemd   Install + enable gripette.service and gripette-bluetooth.service"
	@echo "  ensure-ble-only   Force BlueZ into BLE-only mode (ControllerMode = le)"
	@echo "  check             Run scripts/check_hardware.py against current install"
	@echo ""
	@echo "Prereqs before 'install-rpi':"
	@echo "  - uv installed (https://docs.astral.sh/uv/)"
	@echo "  - the grabette monorepo cloned on this device (any path; install-systemd"
	@echo "    patches the unit to this device's workspace root automatically)"

# One-shot bring-up for a fresh Raspberry Pi Zero 2W (Bookworm or Trixie).
# Mirrors casquette/Makefile's install-rpi but for gripette's stack
# (picamera2 + Feetech serial bus, no I2C). Idempotent — safe to re-run.
#
# --no-install-package numpy: with --system-site-packages, apt's picamera2
# already provides numpy. Skipping the wheel avoids a slow source build on
# the Zero 2W. (See note in pyproject.toml.)
install-rpi: enable-uart
	@command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
	sudo apt update
	sudo apt install -y python3-libcamera python3-picamera2 libcap-dev
	cd $(WORKSPACE_ROOT) && rm -rf .venv \
	  && uv venv --python /usr/bin/python3 --system-site-packages \
	  && uv sync --package gripette --extra rpi --no-install-package numpy
	@echo "--- verifying imports ---"
	@$(WORKSPACE_ROOT)/.venv/bin/python -c "import picamera2, serial, rustypot; print('all imports OK')"
	@echo "--- pyvenv.cfg ---"
	@grep -E "home|version_info|system-site" $(WORKSPACE_ROOT)/.venv/pyvenv.cfg
	@echo
	@echo "Done. Run 'make check' to verify the camera + motors, then"
	@echo "'uv run --package gripette python -m gripette' to start the service,"
	@echo "or 'make install-systemd' for boot-time start."

# Free /dev/serial0 for motor traffic. Two independent boot-config edits:
#
#   A. cmdline.txt: strip 'console=serial0,...' so the kernel doesn't
#      dump boot messages onto the servo bus. We also run
#      'raspi-config nonint do_serial_cons 1' and 'do_serial_hw 0' to
#      cover the getty mask + enable_uart=1 in config.txt.
#
#   B. config.txt: ensure 'dtoverlay=miniuart-bt' is present. This routes
#      the PL011 (ttyAMA0) to the GPIO header and pushes Bluetooth onto
#      the mini UART. Without it, GPIO gets the mini UART whose clock is
#      tied to the core frequency — unreliable at 1 Mbaud (packet drops
#      on the Feetech bus).
#
# Editing cmdline.txt is delicate: corrupting 'root=PARTUUID=...' bricks
# boot. We capture root= before the edit and roll back from a backup if
# it isn't byte-identical after. The console= regex CAN'T match root=
# (it requires the literal 'console=' prefix), so this is paranoid
# defense-in-depth, not load-bearing.
#
# config.txt is an idempotent append — we only add the dtoverlay line if
# it's not already uncommented somewhere in the file.
#
# A reboot is required after either change.
CMDLINE := $(firstword $(wildcard /boot/firmware/cmdline.txt /boot/cmdline.txt))
CONFIG  := $(firstword $(wildcard /boot/firmware/config.txt /boot/config.txt))

enable-uart:
	@if [ -z "$(CMDLINE)" ] || [ -z "$(CONFIG)" ]; then \
		echo "No cmdline.txt/config.txt found — not a Raspberry Pi? Skipping UART setup."; \
		exit 0; \
	fi
	@# --- A. cmdline.txt: strip 'console=serial0,...' ------------------
	@if ! grep -qE 'console=(serial0|ttyAMA0|ttyS0)' $(CMDLINE); then \
		echo "[cmdline.txt] Serial console already disabled."; \
	else \
		ROOT_BEFORE=$$(grep -oE 'root=[^ ]+' $(CMDLINE)); \
		if [ -z "$$ROOT_BEFORE" ]; then \
			echo "ERROR: no 'root=' found in $(CMDLINE) — refusing to edit (file looks malformed)."; \
			exit 1; \
		fi; \
		echo "[cmdline.txt] Disabling serial console + enabling UART hardware..."; \
		echo "  root= will be preserved as: $$ROOT_BEFORE"; \
		echo "  backing up $(CMDLINE) -> $(CMDLINE).gripette.bak"; \
		sudo cp $(CMDLINE) $(CMDLINE).gripette.bak; \
		sudo raspi-config nonint do_serial_cons 1; \
		sudo raspi-config nonint do_serial_hw 0; \
		sudo sed -i -E 's/ ?console=(serial0|ttyAMA0|ttyS0),[0-9]+ ?/ /g; s/  +/ /g; s/^ +//; s/ +$$//' $(CMDLINE); \
		ROOT_AFTER=$$(grep -oE 'root=[^ ]+' $(CMDLINE)); \
		if [ "$$ROOT_BEFORE" != "$$ROOT_AFTER" ]; then \
			echo ""; \
			echo "*** ABORT: root= changed during edit ***"; \
			echo "  before: $$ROOT_BEFORE"; \
			echo "  after:  $$ROOT_AFTER"; \
			echo "Rolling back from $(CMDLINE).gripette.bak — boot is safe."; \
			sudo cp $(CMDLINE).gripette.bak $(CMDLINE); \
			exit 1; \
		fi; \
		echo "  verified: root= unchanged."; \
	fi
	@# --- B. config.txt: route PL011 to GPIO via miniuart-bt -----------
	@if grep -qE '^[[:space:]]*dtoverlay=miniuart-bt' $(CONFIG); then \
		echo "[config.txt] dtoverlay=miniuart-bt already present — PL011 routed to GPIO."; \
	else \
		echo "[config.txt] Adding dtoverlay=miniuart-bt (PL011 -> GPIO for reliable 1Mbaud)..."; \
		echo "  backing up $(CONFIG) -> $(CONFIG).gripette.bak"; \
		sudo cp $(CONFIG) $(CONFIG).gripette.bak; \
		echo "" | sudo tee -a $(CONFIG) >/dev/null; \
		echo "# gripette: route PL011 (ttyAMA0) to GPIO for reliable 1Mbaud motor traffic" | sudo tee -a $(CONFIG) >/dev/null; \
		echo "dtoverlay=miniuart-bt" | sudo tee -a $(CONFIG) >/dev/null; \
		if ! grep -qE '^[[:space:]]*dtoverlay=miniuart-bt' $(CONFIG); then \
			echo "*** ERROR: append verification failed. Rolling back. ***"; \
			sudo cp $(CONFIG).gripette.bak $(CONFIG); \
			exit 1; \
		fi; \
		echo "  verified: dtoverlay=miniuart-bt now in $(CONFIG)."; \
	fi
	@echo ""
	@echo "If either step above reported a change, *** REBOOT REQUIRED *** before /dev/serial0"
	@echo "will be free for motor traffic at reliable 1Mbaud."

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

# Install both gripette services. Run after install-rpi has verified the
# venv builds. The bluetooth service needs root (BlueZ DBus access); the
# main service runs as rasp.
install-systemd: ensure-ble-only
	@# Patch the units' hardcoded path to this device's actual workspace root
	@# (where the venv lives) so boot-time start uses the right interpreter.
	sed 's#/home/rasp/Project/Repo/gripette#$(WORKSPACE_ROOT)#g' systemd/gripette.service \
	  | sudo tee /etc/systemd/system/gripette.service >/dev/null
	sed 's#/home/rasp/Project/Repo/gripette#$(WORKSPACE_ROOT)#g' systemd/gripette-bluetooth.service \
	  | sudo tee /etc/systemd/system/gripette-bluetooth.service >/dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable --now gripette
	sudo systemctl enable --now gripette-bluetooth
	@echo "Logs:"
	@echo "  journalctl -u gripette -f"
	@echo "  journalctl -u gripette-bluetooth -f"

check:
	@if [ ! -x $(WORKSPACE_ROOT)/.venv/bin/python ]; then \
		echo "No $(WORKSPACE_ROOT)/.venv/bin/python found. Run 'make install-rpi' first —"; \
		echo "the venv must be created with --system-site-packages so apt's"; \
		echo "picamera2 satisfies the dependency tree (otherwise uv tries to"; \
		echo "build python-prctl from PyPI, which needs libcap-dev)."; \
		exit 1; \
	fi
	$(WORKSPACE_ROOT)/.venv/bin/python scripts/check_hardware.py
