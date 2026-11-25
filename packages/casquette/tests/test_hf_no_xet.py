"""Uploads must not go through hf_xet on casquette.

hf_xet OOM-killed the service on a Pi Zero 2 W (416 MB) ~10 s into an upload:
the fleet saw the device go offline and the upload was lost. casquette.fleet.hf
sets HF_HUB_DISABLE_XET before huggingface_hub is imported, so uploads fall back
to streamed LFS HTTP. Run in a fresh interpreter: huggingface_hub reads the
variable once, at import.
"""

from __future__ import annotations

import os
import subprocess
import sys

_CHECK = (
    "import casquette.fleet.hf\n"
    "from huggingface_hub import constants\n"
    "from huggingface_hub.utils._runtime import is_xet_available\n"
    "print(constants.HF_HUB_DISABLE_XET, is_xet_available())\n"
)


def _run(env_extra: dict) -> str:
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_DISABLE_XET"}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", _CHECK], env=env, check=True,
                          capture_output=True, text=True).stdout.split()


def test_importing_the_hf_client_disables_xet():
    assert _run({}) == ["True", "False"]


def test_xet_can_still_be_re_enabled_from_the_environment():
    assert _run({"HF_HUB_DISABLE_XET": "0"})[0] == "False"
