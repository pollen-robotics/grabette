"""Which non-chunk-relative ('delta') pi0.5 checkpoints exist, and how big?

Metadata only — no weights fetched. Also reports each one's action layout and
training dataset, so a fair same-task comparison can be set up rather than
comparing across objects as well as across representations.
"""

import json

from huggingface_hub import HfApi, hf_hub_download

CANDIDATES = [
    "SteveNguyen/pick3_graspproj_pi05",
    "chouziel/sugar_cup_grasproj_pi05",
    "SteveNguyen/pick3_graspproj_chunkrel_pi05",          # for reference
    "SteveNguyen/sugar_cup_chunkrel_pi05_step20000",      # for reference
]


def main() -> None:
    api = HfApi()
    for repo in CANDIDATES:
        print(f"\n=== {repo}")
        try:
            info = api.model_info(repo, files_metadata=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  unavailable: {type(exc).__name__}: {exc}")
            continue
        total = sum(s.size or 0 for s in info.siblings)
        print(f"  modified {info.lastModified}   total {total/1e9:.2f} GB")
        try:
            cfg = json.load(open(hf_hub_download(repo, "config.json")))
            out = {k: v["shape"] for k, v in (cfg.get("output_features") or {}).items()}
            print(f"  output: {out}   chunk {cfg.get('chunk_size')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  config unreadable: {type(exc).__name__}")
        try:
            tc = json.load(open(hf_hub_download(repo, "train_config.json")))
            ds = (tc.get("dataset") or {}).get("repo_id")
            print(f"  trained on: {ds}   steps {tc.get('steps')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  train_config unreadable: {type(exc).__name__}")
        try:
            pre = json.load(open(hf_hub_download(repo, "policy_preprocessor.json")))
            names = [s.get("registry_name") for s in pre.get("steps", [])]
            print(f"  preprocessor: {names}")
            post = json.load(open(hf_hub_download(repo, "policy_postprocessor.json")))
            print(f"  postprocessor: "
                  f"{[s.get('registry_name') for s in post.get('steps', [])]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  processors unreadable: {type(exc).__name__}")


if __name__ == "__main__":
    main()
