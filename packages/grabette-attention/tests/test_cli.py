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
