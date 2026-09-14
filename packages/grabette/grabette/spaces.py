"""Which HuggingFace Spaces this device talks to — the one place that knows.

The fleet Space URL used to be a literal in three independent spots (this
package's Settings default, auth.py's OAuth relay, and casquette's mirror of the
same setting). Renaming or re-pointing a Space meant finding all three, and the
one that got missed failed in a confusing way: a device that registers with the
fleet fine but whose login bounces, because auth.py built its redirect_uri from
a different Space than the relay client polls.

Switching to the development deployment is one line in /etc/grabette/env:

    GRABETTE_FLEET_ENV=test

No URL to retype, and both the relay and the OAuth redirect move together.
GRABETTE_RELAY_URL still wins when it is set — that is how a duplicated Space is
targeted (see docs/source/spaces.md), and "" selects direct OAuth for local dev.

"" re-points OAuth and nothing else: the relay client still starts, with no URL
to call. A device meant to run standalone wants GRABETTE_RELAY_ENABLED=false,
which stops the relay client, the HF token refresh and group sync together.

One thing this cannot do for you: the fleet Space URL is also the OAuth
redirect_uri, so the test Space must be registered as one in the HF OAuth app or
login fails on a device pointed at it.
"""

from __future__ import annotations

import os

PROD = "prod"
TEST = "test"
_ORG = "pollen-robotics"
# The env var is read here rather than through Settings because auth.py needs the
# answer at import time, before any Settings instance exists.
_ENV_VAR = "GRABETTE_FLEET_ENV"


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
    """The fleet Space this device registers with and logs in through.

    GRABETTE_RELAY_URL wins when PRESENT, empty included — "" is a real setting
    (direct OAuth, see the module docstring), not a missing one. It does not stop
    the relay client; GRABETTE_RELAY_ENABLED=false does."""
    override = os.environ.get("GRABETTE_RELAY_URL")
    if override is not None:
        return override.rstrip("/")
    return space_url("fleet")


def is_overridden() -> bool:
    """Is the fleet URL something other than the one this env derives?

    Asked so the dashboard can flag ANY unusual target, not just the test
    deployment: a device pointed at a personal Space is just as easy to forget,
    and used to look exactly like production."""
    return fleet_url() != space_url("fleet")
