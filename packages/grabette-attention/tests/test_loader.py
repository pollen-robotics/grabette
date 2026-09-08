"""loader.py must not import torch (or lerobot) at module scope: the heavy
stack stays host-provided and deferred to inside `load_pi05`, which is what
lets `cli.py` import and parse arguments without it installed.
"""
import importlib


def test_loader_module_does_not_import_torch_at_top_level():
    module = importlib.import_module("grabette_attention.loader")
    assert not hasattr(module, "torch")
