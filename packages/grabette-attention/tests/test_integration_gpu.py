"""One end-to-end check against a real checkpoint. Skipped unless asked for.

Run on the GPU box:
    GRABETTE_ATTN_CKPT=<user>/<model>_best \
    GRABETTE_ATTN_DATASET=<user>/<dataset>_graspproj \
    uv run pytest -m gpu packages/grabette-attention/tests/test_integration_gpu.py -v

It asserts shapes and invariants, not values: the point is that the hooks fire
against the real module tree and that the masses are a proper distribution.
"""
import os

import pytest

pytestmark = pytest.mark.gpu

CKPT = os.environ.get("GRABETTE_ATTN_CKPT")
DATASET = os.environ.get("GRABETTE_ATTN_DATASET")


@pytest.mark.skipif(not CKPT or not DATASET, reason="set GRABETTE_ATTN_CKPT and _DATASET")
def test_a_real_checkpoint_yields_maps_masses_and_an_ablation():
    from grabette_attention.adapters.pi05 import Pi05Adapter
    from grabette_attention.analysis import analyse_frame
    from grabette_attention.loader import load_pi05
    from grabette_attention.sources import DatasetSource

    policy, preprocessor = load_pi05(CKPT, device="cuda", fp32=True)
    adapter = Pi05Adapter(policy, preprocessor, device="cuda", seed=0)

    source = DatasetSource(
        DATASET, episodes=[0], camera_keys=adapter.camera_keys,
        task="pick the sugar cube", selection="grasp", count=1,
    )
    obs = next(iter(source.frames()))
    analysis = analyse_frame(adapter, obs)

    # The hooks fired against the real expert layers.
    assert set(analysis.cameras) == set(adapter.camera_keys)
    # Masses are a distribution over the prefix.
    total = sum(c.mass for c in analysis.cameras.values()) + analysis.language_mass
    assert total == pytest.approx(1.0, abs=1e-5)
    # The grid lost its letterbox padding rows.
    grid = next(iter(analysis.cameras.values())).grid
    assert grid.ndim == 2 and grid.shape[0] < grid.shape[1]
    # Removing the only camera must change the commanded motion.
    assert next(iter(analysis.ablations.values())).delta_mm > 0.0
