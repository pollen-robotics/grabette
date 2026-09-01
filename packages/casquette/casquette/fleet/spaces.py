# Mirrors grabette/spaces.py for the casquette fleet prototype. Deviations: the
# env vars carry casquette's CASQUETTE_ prefix (its Settings does), and there is
# no OAuth relay here — only the relay client consumes this. The Space NAMES are
# identical on purpose: both device types report to one fleet. Keep in sync until
# extracted to a shared package. See casquette/fleet/__init__.py.
"""Which HuggingFace Spaces this device talks to — the one place that knows.

A casquette registers with the SAME fleet Space as a grabette — the fleet is
per-operator, not per-device-type — so the URLs are built the same way and must
move together when the deployment changes.

Switching to the development deployment is one line in /etc/casquette/env:

    CASQUETTE_FLEET_ENV=test

No URL to retype. CASQUETTE_RELAY_URL still wins when it is set — that is how a
duplicated Space is targeted (see grabette's docs/source/spaces.md).

Note "" does NOT stand for "no relay" here: casquette has no OAuth module, so an
empty URL just leaves the relay client with nothing to call. Standalone is
CASQUETTE_RELAY_ENABLED=false.
"""

from __future__ import annotations

import os

PROD = "prod"
TEST = "test"
_ORG = "pollen-robotics"
# Read straight from the environment so this stays usable before any Settings
# instance exists (grabette's copy needs that for its OAuth module).
_ENV_VAR = "CASQUETTE_FLEET_ENV"


def fleet_env() -> str:
    """Which deployment this device targets: "prod" (default) or "test".

    Anything unrecognised is prod. A typo must not be the thing that quietly
    points a device at the development fleet — the failure mode of guessing wrong
    is a day of recordings landing where nobody looks for them."""
    return TEST if (os.environ.get(_ENV_VAR) or "").strip().lower() == TEST else PROD


def is_test() -> bool:
    return fleet_env() == TEST


def space_name(kind: str) -> str:
    """e.g. space_name("fleet") -> "grabette-fleet-test" on the test env."""
    return f"grabette-{kind}" + ("-test" if is_test() else "")


def space_url(kind: str) -> str:
    return f"https://{_ORG}-{space_name(kind)}.hf.space"


def fleet_url() -> str:
    """The fleet Space this device registers with.

    CASQUETTE_RELAY_URL wins when PRESENT, empty included. "" is not "no relay"
    here (see the module docstring) — standalone is CASQUETTE_RELAY_ENABLED=false."""
    override = os.environ.get("CASQUETTE_RELAY_URL")
    if override is not None:
        return override.rstrip("/")
    return space_url("fleet")


def is_overridden() -> bool:
    """Is the fleet URL something other than the one this env derives?"""
    return fleet_url() != space_url("fleet")
