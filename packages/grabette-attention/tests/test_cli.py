"""CLI argument handling. No checkpoint, no GPU — parsing and validation only."""
import pytest

from grabette_attention.cli import build_parser, main


def test_dataset_mode_parses_episodes_as_integers():
    args = build_parser().parse_args(
        ["--checkpoint", "user/m", "--dataset", "user/d", "--episodes", "3", "7", "11"]
    )
    assert args.episodes == [3, 7, 11]


def test_dump_obs_mode_parses_a_directory():
    args = build_parser().parse_args(
        ["--checkpoint", "user/m", "--dump-obs", "out/ep003"]
    )
    assert args.dump_obs == "out/ep003"
    assert args.dataset is None


def test_grasp_is_the_default_frame_selection():
    args = build_parser().parse_args(["--checkpoint", "user/m", "--dataset", "user/d"])
    assert args.frames == "grasp"


def test_the_last_denoising_step_is_the_default():
    args = build_parser().parse_args(["--checkpoint", "user/m", "--dataset", "user/d"])
    assert args.denoise_step == "last"


def test_fp32_is_the_default_because_the_bf16_flow_path_is_broken():
    args = build_parser().parse_args(["--checkpoint", "user/m", "--dataset", "user/d"])
    assert args.fp32 is True


def test_ablation_is_on_by_default_and_can_be_turned_off():
    parser = build_parser()
    assert parser.parse_args(["--checkpoint", "c", "--dataset", "d"]).ablate is True
    assert parser.parse_args(
        ["--checkpoint", "c", "--dataset", "d", "--no-ablation"]
    ).ablate is False


def test_explicit_frame_indices_are_accepted():
    args = build_parser().parse_args(
        ["--checkpoint", "c", "--dataset", "d", "--frames", "12", "40"]
    )
    assert args.frames == ["12", "40"]


def test_giving_neither_input_is_rejected():
    with pytest.raises(SystemExit):
        main(["--checkpoint", "user/m"])


def test_giving_both_inputs_is_rejected():
    with pytest.raises(SystemExit):
        main(["--checkpoint", "c", "--dataset", "d", "--dump-obs", "out/ep0"])


def test_a_camera_key_not_in_the_checkpoint_is_rejected(monkeypatch):
    # Finding 3: otherwise this only surfaces later as "no usable camera",
    # which names the symptom rather than the mismatched --camera-key that
    # caused it.
    import grabette_attention.adapters.pi05 as pi05_mod
    import grabette_attention.loader as loader_mod

    monkeypatch.setattr(loader_mod, "load_pi05", lambda *a, **k: (None, None, None))

    class FakeAdapter:
        camera_keys = ("observation.images.cam0",)

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(pi05_mod, "Pi05Adapter", FakeAdapter)

    with pytest.raises(SystemExit):
        main([
            "--checkpoint", "c", "--dump-obs", "out/ep0",
            "--camera-key", "observation.images.nope",
        ])


class _FakeAdapterForMain:
    """Enough of the adapter protocol for `main()` to run analyse_frame end to
    end against one camera, without a real policy."""

    patch = 14
    camera_keys = ("observation.images.cam0",)

    def __init__(self, *a, **k):
        pass

    def geometry(self, obs, camera):
        from grabette_attention.layout import LetterboxGeometry

        frame = obs.images[camera]
        return LetterboxGeometry.from_shapes(
            src_hw=(frame.shape[0], frame.shape[1]), dst_hw=(224, 224)
        )

    def layout(self, obs, *, drop_camera=None):
        from grabette_attention.layout import TokenLayout

        masked = {drop_camera} if drop_camera else set()
        return TokenLayout(
            camera_keys=self.camera_keys, tokens_per_image=256,
            grid_rows=16, grid_cols=16, language_tokens=200,
            masked_cameras=frozenset(masked),
        )

    def draw_noise(self):
        return "noise"

    def run(self, obs, noise, *, capture, drop_camera=None):
        import numpy as np

        from grabette_attention.adapters.base import RunResult

        chunk = np.zeros((50, 11), np.float32)
        captures = {}
        if capture:
            keys = 256 + 200 + 50   # tokens_per_image + language_tokens + queries
            captures = {(0, 0): np.ones((8, 50, keys), np.float32)}
        return RunResult(chunk=chunk, captures=captures)


class _FakeDatasetForMain:
    """One camera, one frame, a gripper channel that never closes -- enough
    for `DatasetSource` to run its real selection logic end to end."""

    def __init__(self, repo_id, root=None, episodes=None):
        self._episode = episodes[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        import numpy as np

        return {
            "observation.images.cam0": np.zeros((3, 4, 4), np.float32),
            "observation.state": np.zeros(2, np.float32),
            "action": np.zeros(3, np.float32),
        }


def _patch_fake_policy(monkeypatch):
    import grabette_attention.adapters.pi05 as pi05_mod
    import grabette_attention.loader as loader_mod

    monkeypatch.setattr(loader_mod, "load_pi05", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(pi05_mod, "Pi05Adapter", _FakeAdapterForMain)


def _patch_fake_dataset(monkeypatch):
    import sys
    import types

    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = _FakeDatasetForMain
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)


def test_a_missing_png_extra_still_leaves_the_summary_written(monkeypatch, tmp_path, capsys):
    # Finding 6: the plotting dependency is behind the 'png' extra and may be
    # absent. An expensive run -- checkpoint already loaded, first frame
    # already computed -- must not be lost entirely: warn once, keep going,
    # still write summary.txt.
    import grabette_attention.cli as cli_mod

    _patch_fake_policy(monkeypatch)
    _patch_fake_dataset(monkeypatch)

    def raise_import_error(*a, **k):
        raise ImportError("no module named 'matplotlib'")

    monkeypatch.setattr(cli_mod, "write_overlays", raise_import_error)

    out_dir = tmp_path / "out"
    exit_code = main([
        "--checkpoint", "c", "--dataset", "d", "--episodes", "0",
        "--out", str(out_dir),
    ])

    assert exit_code == 0
    assert (out_dir / "summary.txt").exists()
    err = capsys.readouterr().err
    assert "png" in err.lower() or "matplotlib" in err.lower()
