.PHONY: install-rpi check-hand write-env install-udev-oak install-systemd install-netdev install-poweroff install-ntp ensure-ble-only help

# This package is a uv workspace member; the environment uv actually uses
# lives at the workspace ROOT, not here. All venv ops target that root.
WORKSPACE_ROOT := $(abspath $(CURDIR)/../..)

help:
	@echo "Targets:"
	@echo "  install-rpi HAND=L|R  Install apt deps + udev rule, create venv, sync, verify."
	@echo "                        HAND={left,right} REQUIRED — determines proximal/distal sign."
	@echo "  install-udev-oak      Install only the OAK-D / Movidius udev rule"
	@echo "  install-systemd       Install + enable the grabette + bluetooth systemd services (BLE-only)"
	@echo "  ensure-ble-only       Force BlueZ into BLE-only mode (ControllerMode = le)"
	@echo "  install-netdev        Allow 'rasp' user to scan and connect to WiFi networks via nmcli"
	@echo "  install-poweroff      Allow 'rasp' user to power off the Pi from the web UI"
	@echo "  install-ntp           Pin NTP to a shared time service (tight multi-device start sync)"
	@echo ""
	@echo "HAND must be 'left' or 'right' — this grabette's hand. Determines the per-sensor"
	@echo "sign mapping in grabette/hardware/angle.py, written to /etc/grabette/env as"
	@echo "GRABETTE_HAND. Example:  make install-rpi HAND=right"

# One-shot bring-up for a fresh Raspberry Pi (Bookworm or Trixie).
# Handles the apt-package + correct-Python + system-site-packages combo that's
# easy to get wrong by hand and silently falls back to MockBackend on failure.
install-rpi: check-hand install-udev-oak install-ntp
	@command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
	sudo apt update
	sudo apt install -y python3-libcamera python3-picamera2 libcap-dev ffmpeg \
	  python3-dbus python3-gi git-lfs
	@# python3-dbus / python3-gi: system deps the bluetooth WiFi service imports
	@# (dbus, gi.repository.GLib); the venv reaches them via --system-site-packages.
	@# git-lfs: mesh assets in the repo are LFS-tracked. Having the binary
	@# avoids smudge-filter warnings on later `git pull`; on-device clones
	@# typically use GIT_LFS_SKIP_SMUDGE=1 so meshes still aren't fetched.
	sudo rm -rf $(WORKSPACE_ROOT)/.venv
	cd $(WORKSPACE_ROOT) && uv venv --python /usr/bin/python3 --system-site-packages \
	  && uv sync --package grabette --extra rpi --extra ui --extra hf
	@echo "--- verifying imports ---"
	@$(WORKSPACE_ROOT)/.venv/bin/python -c "import picamera2, depthai, cv2, gpiod, numpy; print('all imports OK')"
	@echo "--- pyvenv.cfg ---"
	@grep -E "home|version_info|system-site" $(WORKSPACE_ROOT)/.venv/pyvenv.cfg
	@echo
	@$(MAKE) --no-print-directory write-env HAND=$(HAND)
	@echo
	@echo "Done. Run 'uv run --package grabette python -m grabette' to start the daemon."
	@echo "Watch journalctl for 'RPi hardware detected, using RpiBackend' (NOT MockBackend)."

# Fail fast if HAND wasn't provided. Listed as a prerequisite of install-rpi
# so the rest of the recipe never runs without a valid hand.
check-hand:
	@if [ "$(HAND)" != "left" ] && [ "$(HAND)" != "right" ]; then \
		echo "ERROR: HAND must be 'left' or 'right'."; \
		echo "Usage:  make install-rpi HAND=right"; \
		exit 1; \
	fi

# Write /etc/grabette/env with GRABETTE_HAND=$(HAND), PRESERVING any existing
# GRABETTE_*_SIGN lines (operator may have overridden one for an asymmetric
# build) so a re-run of install-rpi doesn't wipe them.
write-env: check-hand
	@sudo mkdir -p /etc/grabette
	@sudo touch /etc/grabette/env
	@{ \
	  echo "# /etc/grabette/env — per-device persistent config."; \
	  echo "# Written by 'make install-rpi HAND=...'."; \
	  echo "# Sourced by grabette.service via EnvironmentFile=-/etc/grabette/env."; \
	  echo "GRABETTE_HAND=$(HAND)"; \
	  sudo grep -E '^GRABETTE_(PROXIMAL|DISTAL)_SIGN=' /etc/grabette/env 2>/dev/null || true; \
	} | sudo tee /etc/grabette/env.new >/dev/null
	@sudo mv /etc/grabette/env.new /etc/grabette/env
	@echo "Wrote /etc/grabette/env:"
	@sudo cat /etc/grabette/env | sed 's/^/  /'

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
#
# nmcli device wifi connect additionally requires:
# 3. org.freedesktop.NetworkManager.network-control  (activate/deactivate connections)
# 4. org.freedesktop.NetworkManager.settings.modify.system  (create connection profile)
POLKIT_SCAN_RULE    := /etc/polkit-1/rules.d/10-grabette-wifi-scan.rules
POLKIT_CONNECT_RULE := /etc/polkit-1/rules.d/10-grabette-wifi-connect.rules
install-netdev:
	sudo usermod -aG netdev rasp
	@if [ ! -f $(POLKIT_SCAN_RULE) ]; then \
		echo "Installing polkit rule for WiFi scan..."; \
		printf 'polkit.addRule(function(action, subject) {\n    if (action.id === "org.freedesktop.NetworkManager.wifi.scan" &&\n        subject.isInGroup("netdev")) {\n        return polkit.Result.YES;\n    }\n});\n' \
		  | sudo tee $(POLKIT_SCAN_RULE) >/dev/null; \
	else \
		echo "Polkit WiFi scan rule already present."; \
	fi
	@if [ ! -f $(POLKIT_CONNECT_RULE) ]; then \
		echo "Installing polkit rule for WiFi connect..."; \
		printf 'polkit.addRule(function(action, subject) {\n    if ((action.id === "org.freedesktop.NetworkManager.network-control" ||\n         action.id === "org.freedesktop.NetworkManager.settings.modify.system") &&\n        subject.isInGroup("netdev")) {\n        return polkit.Result.YES;\n    }\n});\n' \
		  | sudo tee $(POLKIT_CONNECT_RULE) >/dev/null; \
	else \
		echo "Polkit WiFi connect rule already present."; \
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

# Pin systemd-timesyncd to a single coordinated (anycast) time service, shared
# by every grabette, so their clocks track each other tightly — required for
# tight multi-device recording start sync (each device waits out the shared T0
# on its own clock, so any inter-device clock offset becomes a start offset).
# Idempotent. See config/timesyncd-grabette.conf for the why + the local-server
# alternative for sub-ms.
install-ntp:
	sudo mkdir -p /etc/systemd/timesyncd.conf.d
	sudo install -m 0644 -o root -g root config/timesyncd-grabette.conf \
	  /etc/systemd/timesyncd.conf.d/grabette.conf
	sudo systemctl restart systemd-timesyncd
	@echo "Installed NTP drop-in. timesync status:"
	@timedatectl timesync-status 2>/dev/null | grep -Ei 'Server|Offset' || true

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
