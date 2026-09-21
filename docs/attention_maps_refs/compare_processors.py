"""What do the delta and chunk-relative postprocessors actually DO?

The comparison Steve asked for is only meaningful if both models' outputs are
the same kind of quantity. They might not be:

  chunk-relative: the chunk holds OFFSETS from the current pose, so an RMS over
                  chunk steps is a spatial displacement (tens of mm).
  delta:          if the chunk holds PER-STEP deltas, the same RMS is a
                  per-step quantity roughly 50x smaller, and comparing the two
                  in millimetres would be meaningless.

`AbsoluteActionsProcessorStep` does `action += state`, not a cumulative sum, and
it defaults to enabled=False -- so whether it converts anything at all is a
per-checkpoint fact that has to be read, not assumed.

Prints every processor step's config for both model families, plus the
relative-step mask that decides WHICH action dimensions are treated as
relative.
"""

import json

from huggingface_hub import hf_hub_download

PAIRS = {
    "delta   (pick3)":    "SteveNguyen/pick3_graspproj_pi05",
    "delta   (sugar)":    "chouziel/sugar_cup_grasproj_pi05",
    "chunkrel(pick3)":    "SteveNguyen/pick3_graspproj_chunkrel_pi05",
    "chunkrel(sugar)":    "SteveNguyen/sugar_cup_chunkrel_pi05_step20000",
}
INTERESTING = (
    "relative_actions_processor", "absolute_actions_processor",
    "grabette_chunk_relative_actions", "grabette_absolute_from_chunk_relative",
)


def main() -> None:
    for label, repo in PAIRS.items():
        print(f"\n=== {label}  {repo}")
        for which in ("policy_preprocessor.json", "policy_postprocessor.json"):
            try:
                doc = json.load(open(hf_hub_download(repo, which)))
            except Exception as exc:  # noqa: BLE001
                print(f"  {which}: unavailable ({type(exc).__name__})")
                continue
            for step in doc.get("steps", []):
                name = step.get("registry_name")
                if name not in INTERESTING:
                    continue
                cfg = step.get("config", {})
                print(f"  {which.split('_')[1][:4]:<5} {name}")
                for k, v in cfg.items():
                    s = json.dumps(v)
                    print(f"        {k} = {s[:170]}")


if __name__ == "__main__":
    main()
