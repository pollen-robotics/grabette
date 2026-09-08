"""`grabette-attn`: offline attention maps and view ablation.

Deliberately offline. The maps are a hypothesis and the ablation millimetres are
the measurement; neither belongs on the robot's control path.
"""

import argparse
import sys
from pathlib import Path

from .analysis import analyse
from .frontends.png import write_overlays, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grabette-attn",
        description=(
            "Show where a policy's action tokens attend on each camera, and how "
            "much the commanded chunk changes when each camera is removed."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="local path or Hub repo id")
    parser.add_argument("--dataset", default=None, help="LeRobot dataset repo id")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--dump-obs", default=None, help="an episode dir from evaluate.py --dump_obs")
    parser.add_argument(
        "--camera-key",
        default="observation.images.cam0",
        help="which camera the dump_obs PNGs belong to (they carry no name)",
    )
    parser.add_argument("--task", default="", help="language prompt; required for dump_obs")
    parser.add_argument(
        "--frames",
        nargs="*",
        default="grasp",
        help="'grasp' (default), 'stride', or explicit frame indices",
    )
    parser.add_argument("--count", type=int, default=1, help="frames per episode")
    parser.add_argument("--denoise-step", default="last", help="last, first, mean, or an index")
    parser.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    parser.add_argument("--no-ablation", dest="ablate", action="store_false")
    parser.add_argument("--bf16", dest="fp32", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="attention_out")
    parser.add_argument(
        "--rerun", action="store_true", help="also log to a rerun timeline"
    )
    parser.set_defaults(ablate=True, fp32=True)
    return parser


def _frames_mode(raw):
    if isinstance(raw, str):
        return raw
    if len(raw) == 1 and raw[0] in ("grasp", "stride"):
        return raw[0]
    return [int(i) for i in raw]


def _layers(raw: str):
    if raw in ("all", "mean"):
        return raw
    return [int(i) for i in raw.split(",")]


def _denoise_step(raw: str):
    if raw in ("last", "first", "mean", "all"):
        return raw
    return int(raw)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.dataset is None) == (args.dump_obs is None):
        parser.error("give exactly one of --dataset or --dump-obs")

    from .adapters.pi05 import Pi05Adapter
    from .loader import load_pi05
    from .sources import DatasetSource, DumpObsSource

    policy, preprocessor = load_pi05(
        args.checkpoint, device=args.device, fp32=args.fp32
    )
    adapter = Pi05Adapter(
        policy, preprocessor, device=args.device, seed=args.seed
    )

    if args.dump_obs is not None:
        source = DumpObsSource(
            args.dump_obs, task=args.task, camera_key=args.camera_key
        )
        notes = {0: f"dump_obs capture {args.dump_obs}"}
    else:
        source = DatasetSource(
            args.dataset,
            episodes=args.episodes,
            camera_keys=adapter.camera_keys,
            task=args.task,
            root=args.dataset_root,
            selection=_frames_mode(args.frames),
            count=args.count,
        )
        notes = source.notes

    out_root = Path(args.out)
    recording = None
    if args.rerun:
        from .frontends.rerun_logger import open_recording

        recording = open_recording()

    analyses = []
    for obs in source.frames():
        analysis = next(
            analyse(
                adapter,
                [obs],
                denoise_step=_denoise_step(args.denoise_step),
                layers=_layers(args.layers),
                ablate=args.ablate,
                provenance={
                    "checkpoint": args.checkpoint,
                    "seed": str(args.seed),
                    "dtype": "fp32" if args.fp32 else "bf16",
                },
            )
        )
        episode_dir = out_root / f"ep{obs.episode:03d}"
        write_overlays(analysis, obs, episode_dir)
        if recording is not None:
            from .frontends.rerun_logger import log_analysis

            log_analysis(analysis, obs, recording=recording)
        analyses.append(analysis)
        print(
            f"ep{obs.episode:03d} frame {obs.frame}: "
            + "  ".join(
                f"{c.rsplit('.', 1)[-1]} mass {a.mass:.2f}"
                + (
                    f" ablate {analysis.ablations[c].delta_mm:.1f}mm"
                    if c in analysis.ablations
                    else ""
                )
                for c, a in analysis.cameras.items()
            )
        )

    summary = write_summary(analyses, out_root, notes=notes)
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
