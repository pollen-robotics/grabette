"""The steps must be in lerobot's registry, or a checkpoint cannot be loaded.

This exists because of a real failure: both inference scripts imported
`grabette_chunkrel.chunk_relative` (the maths) and neither imported
`chunk_relative_processor` (where the @ProcessorStepRegistry.register decorators
live). Declaring the dependency and installing the package are both irrelevant —
registration happens at IMPORT time of that specific module. lerobot then failed
with a bare KeyError listing every step except ours.
"""

import subprocess
import sys

import pytest

pytest.importorskip("lerobot")

from grabette_chunkrel.chunk_relative_processor import (  # noqa: E402
    INVERSE_STEP_NAME,
    STEP_NAME,
)


def test_importing_the_processor_registers_both_steps():
    from lerobot.processor.pipeline import ProcessorStepRegistry

    for name in (STEP_NAME, INVERSE_STEP_NAME):
        assert ProcessorStepRegistry.get(name) is not None, f"{name} not registered"


def test_importing_only_the_maths_does_not_register():
    """The trap, asserted in a fresh interpreter: if this ever starts passing
    registration by accident, the explicit imports in evaluate.py and
    smoke_generation.py look redundant and someone will delete them."""
    code = (
        "import grabette_chunkrel.chunk_relative\n"
        "from lerobot.processor.pipeline import ProcessorStepRegistry\n"
        f"print({STEP_NAME!r} in ProcessorStepRegistry.list())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False", (
        "importing the maths alone now registers the step; the explicit "
        "chunk_relative_processor imports in the inference scripts can only be "
        "removed if that becomes guaranteed"
    )


def test_the_inference_scripts_import_the_processor_module():
    """Guards the two call sites directly. A checkpoint that will not load is a
    robot session lost, so this is cheap insurance against the import being
    tidied away as unused (both carry `# noqa: F401`)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    scripts = [
        root / "integrations" / "Pi05" / "smoke_generation.py",
        root / "integrations" / "openarm" / "openarm_gripette_simu" / "examples"
        / "evaluate.py",
    ]
    for p in scripts:
        if not p.is_file():
            pytest.skip(f"{p} not present in this checkout")
        assert "grabette_chunkrel.chunk_relative_processor" in p.read_text(), (
            f"{p.name} never imports the processor module, so the checkpoint's "
            "steps will not be in lerobot's registry when it loads"
        )
