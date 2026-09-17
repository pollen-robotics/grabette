"""Publish a copy of a LeRobot dataset with a corrected task string.

The prompt is NOT a training flag. `train.py` takes no --task; the string the
policy learns comes from the dataset's meta/tasks.parquet, resolved per frame
through task_index. So retraining with a correct prompt means publishing a
patched dataset first, and pointing --dataset.repo_id at that.

Patching a copy rather than the original on purpose: the existing sugar
checkpoints were trained against "pick up the sugar cup", and rewriting that
dataset in place would make their provenance a lie.

TWO files carry the string, and only one of them is load-bearing:

  meta/tasks.parquet        what training reads. dataset_reader.py does
                            `item["task"] = self._meta.tasks.iloc[idx].name`,
                            resolving the frame's task_index through this
                            table. Patching this alone changes what the policy
                            learns.
  meta/episodes/*.parquet   a per-episode `tasks` column holding the literal
                            string, 150 copies of it. NOT read during
                            training. Patched anyway: leaving a dataset whose
                            metadata contradicts itself is how a prompt ends
                            up not describing its task in the first place.

task_index values in the per-frame parquet stay valid either way — the
index -> string mapping keeps its indices; only the strings move.

    python retask_dataset.py --dry-run          # inspect, change nothing
    python retask_dataset.py                     # patch and push

Check the dry run before spending a GPU day on the result.
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

# The replacement prompt. Requirements, both from measurement:
#   1. It must describe the actual task -- grasp a cube, place it in a mug --
#      rather than name the destination, which "pick up the sugar cup" does.
#   2. It must not share vocabulary with pick3's strings, because the prompt
#      sweep showed this policy family keys on WORD PRESENCE, not syntax or
#      meaning: "pick up the cup" perturbed the sugar model less than
#      scrambling its own prompt's word order. pick3 uses
#      "pick up the red can" / "pick up the mustard bottle" /
#      "pick up the cup", so "pick", "up" and "cup" are all spent.
# "put the sugar cube in the mug" shares only "the" with any pick3 string.
NEW_TASK = "put the sugar cube in the mug"

DATASETS = {
    # chunk-relative, 8-dim -- pairs with GRABETTE_CHUNK_RELATIVE=1
    "chunkrel": ("SteveNguyen/sugar_cup_graspproj_chunkrel",
                 "SteveNguyen/sugarcube_in_mug_chunkrel"),
    # plain delta, 11-dim graspproj
    "delta": ("chouziel/grabette-sugar-cup-2008_graspproj",
              "SteveNguyen/sugarcube_in_mug_graspproj"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=sorted(DATASETS) + ["all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--task", default=NEW_TASK)
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    targets = sorted(DATASETS) if args.which == "all" else [args.which]

    for key in targets:
        src, dst = DATASETS[key]
        print(f"\n=== {key}: {src}  ->  {dst}")
        local = Path(snapshot_download(src, repo_type="dataset"))
        tasks_file = local / "meta" / "tasks.parquet"
        before = pq.read_table(tasks_file).to_pydict()
        print(f"  current tasks: {before}")

        if len(before["task"]) != 1:
            print(f"  !! {len(before['task'])} task strings — this script "
                  "assumes a single-task dataset; patch by hand")
            continue
        if before["task"][0] == args.task:
            print("  already correct, nothing to do")
            continue

        episode_files = sorted((local / "meta" / "episodes").rglob("*.parquet"))
        stale = 0
        for ep_file in episode_files:
            col = pq.read_table(ep_file).column("tasks").to_pylist()
            stale += sum(1 for entry in col if args.task not in (entry or []))
        print(f"  meta/episodes: {len(episode_files)} file(s), "
              f"{stale} episode rows still carrying the old string")

        videos = list(local.rglob("*.mp4"))
        print(f"  snapshot at {local}")
        print(f"  {len(videos)} video files present "
              f"({sum(f.stat().st_size for f in videos)/1e6:.0f} MB)")
        if not videos:
            print("  !! no videos in the snapshot — the push would publish a "
                  "dataset with no frames. Run `snapshot_download` for the "
                  "full repo first.")
            continue

        print(f"  new task: {args.task!r}")
        if args.dry_run:
            print("  dry run: not writing, not pushing")
            continue

        # Write into a staging copy so the HF cache stays pristine.
        staged = Path("/tmp") / f"retask_{key}"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(local, staged, symlinks=False)
        table = pq.read_table(staged / "meta" / "tasks.parquet")
        patched = table.set_column(
            table.column_names.index("task"), "task",
            [[args.task] * len(table)],
        )
        pq.write_table(patched, staged / "meta" / "tasks.parquet")
        print(f"  meta/tasks.parquet -> "
              f"{pq.read_table(staged / 'meta' / 'tasks.parquet').to_pydict()}")

        # The per-episode copies. Not read during training, but a dataset whose
        # metadata disagrees with itself is a trap for the next reader.
        for ep_file in sorted((staged / "meta" / "episodes").rglob("*.parquet")):
            ep_table = pq.read_table(ep_file)
            index = ep_table.column_names.index("tasks")
            # The column is list<string>: one list per episode.
            ep_patched = ep_table.set_column(
                index, "tasks", [[[args.task]] * len(ep_table)]
            )
            pq.write_table(ep_patched, ep_file)
            check = pq.read_table(ep_file).column("tasks").to_pylist()
            assert all(entry == [args.task] for entry in check), ep_file
            print(f"  {ep_file.relative_to(staged)} -> {len(check)} rows patched")

        # Public, matching the source datasets. `exist_ok` does NOT change the
        # visibility of a repo that already exists, so a repo created private
        # by an earlier run stays private — flip it in the Hub settings.
        api.create_repo(dst, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(staged), repo_id=dst, repo_type="dataset",
            commit_message=f"Sugar-cube place task, prompt {args.task!r} "
                           f"(copy of {src}; only meta/tasks.parquet differs)",
        )
        print(f"  pushed {dst}")

        # THE CODEBASE-VERSION TAG. upload_folder copies files, not refs, and
        # lerobot resolves a dataset revision through get_safe_version(), which
        # raises RevisionNotFoundError when the repo carries no version tag. An
        # untagged copy therefore fails at LeRobotDatasetMetadata.__init__ —
        # and in this huggingface_hub version that error cannot even construct
        # itself (HfHubHTTPError.__init__ missing 'response'), so the traceback
        # ends in an unrelated TypeError and says nothing about tags. A copy is
        # not a usable dataset until this runs.
        version = json.loads((staged / "meta" / "info.json").read_text())[
            "codebase_version"
        ]
        api.create_tag(dst, tag=version, repo_type="dataset", exist_ok=True)
        refs = api.list_repo_refs(dst, repo_type="dataset")
        print(f"  tagged {version} — tags now {[t.name for t in refs.tags]}")


if __name__ == "__main__":
    main()
