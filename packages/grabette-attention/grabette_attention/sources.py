"""Where frames come from: a LeRobot dataset, or a `--dump_obs` capture.

Both yield `FrameObservation` at full resolution and un-normalised, which is the
property that makes them interchangeable: the policy's own preprocessing does
the letterboxing either way.
"""

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .records import FrameObservation


class DumpObsSource:
    """Reads one episode directory written by `evaluate.py --dump_obs`.

    That writer uses `cv2.imwrite`, so the PNGs are BGR on disk and must be
    converted back to RGB here. It records one camera; the key it belongs to is
    supplied by the caller because the PNG carries no name.
    """

    def __init__(self, directory: Path | str, *, task: str, camera_key: str):
        self._dir = Path(directory)
        self._task = task
        self._camera_key = camera_key

    def _states(self) -> dict[int, np.ndarray]:
        path = self._dir / "state.jsonl"
        if not path.exists():
            return {}
        states = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            states[int(record["step"])] = np.asarray(
                record["state"], dtype=np.float32
            )
        return states

    def frames(self) -> Iterator[FrameObservation]:
        import cv2

        states = self._states()
        for path in sorted(self._dir.glob("obs_*.png")):
            index = int(path.stem.split("_")[1])
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"could not read {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield FrameObservation(
                episode=0,
                frame=index,
                images={self._camera_key: rgb},
                state=states.get(index, np.zeros(0, np.float32)),
                task=self._task,
            )


class DatasetSource:
    """Reads held-out frames from a LeRobot dataset.

    Follows the offline loading already used by
    `integrations/Pi05/smoke_generation.py`: one episode at a time, frames
    indexed directly. Camera keys come from the CHECKPOINT, not the dataset, so
    a dataset missing a view is handled by the adapter's masking rather than by
    guessing here.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        episodes: Sequence[int],
        camera_keys: Sequence[str],
        task: str | None,
        root: Path | str | None = None,
        selection: str | Sequence[int] = "grasp",
        count: int = 1,
        gripper_channel: int = -1,
    ):
        self._repo_id = repo_id
        self._episodes = list(episodes)
        self._camera_keys = list(camera_keys)
        self._task = task
        self._root = root
        self._selection = selection
        self._count = count
        self._gripper_channel = gripper_channel
        self.notes: dict[int, str] = {}
        # Per episode, which task string ended up in the observation:
        # "override" when the caller explicitly passed `task`, "dataset" when
        # none was passed and the item's own task was used instead.
        # `analyse_frame`'s provenance records this, since this policy
        # discretizes the robot state INTO the language prompt, so analysing
        # under the wrong task also corrupts the reported language mass.
        self.task_source: dict[int, str] = {}

    def frames(self) -> Iterator[FrameObservation]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        for episode in self._episodes:
            dataset = LeRobotDataset(
                self._repo_id, root=self._root, episodes=[episode]
            )
            n_frames = len(dataset)
            gripper = self._gripper_track(dataset, n_frames)
            indices, note = select_frames(
                gripper, n_frames, mode=self._selection, count=self._count
            )
            self.notes[episode] = note
            for index in indices:
                item = dataset[index]
                images = {}
                for key in self._camera_keys:
                    if key not in item:
                        continue          # absent view: the adapter masks it
                    chw = np.asarray(item[key], dtype=np.float32)
                    images[key] = (
                        np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255)
                        .astype(np.uint8)
                    )
                # An explicitly supplied task always wins: the dataset's own
                # task can be a directory slug rather than language, and this
                # policy discretizes the robot state INTO the language prompt,
                # so silently overriding a caller's `--task` would analyse the
                # policy under a prompt it was never trained on. The dataset's
                # own task is only the default when the caller supplied none;
                # if neither is available, that's an error, not an empty
                # prompt.
                item_task = item.get("task")
                if self._task is not None:
                    task, self.task_source[episode] = self._task, "override"
                elif item_task:
                    task, self.task_source[episode] = item_task, "dataset"
                else:
                    raise ValueError(
                        f"episode {episode}: no --task supplied and the "
                        "dataset item carries none"
                    )
                yield FrameObservation(
                    episode=episode,
                    frame=index,
                    images=images,
                    state=np.asarray(item["observation.state"], dtype=np.float32),
                    task=task,
                )

    def _gripper_track(self, dataset, n_frames: int) -> np.ndarray | None:
        """The gripper channel over the episode, for grasp-frame selection."""
        try:
            return np.asarray(
                [
                    float(np.asarray(dataset[i]["action"])[self._gripper_channel])
                    for i in range(n_frames)
                ],
                dtype=np.float32,
            )
        except (KeyError, IndexError, TypeError):
            return None


def select_frames(
    gripper: np.ndarray | None,
    n_frames: int,
    *,
    mode: str | Sequence[int],
    threshold: float = 0.5,
    count: int = 1,
) -> tuple[list[int], str]:
    """Choose which frames to analyse, and say how the choice was made.

    `grasp` is the default because the literature reports the failure signal
    concentrating at the grasp. When there is no gripper channel, or the episode
    never closes, this falls back to an even stride and RETURNS THAT FACT so the
    summary can print it instead of silently analysing the wrong frames.
    """
    if not isinstance(mode, str):
        indices = [int(i) for i in mode]
        for index in indices:
            if index < 0 or index >= n_frames:
                raise ValueError(
                    f"frame {index} out of range for an episode of {n_frames}"
                )
        return indices, "explicit indices"

    if mode == "stride":
        indices = _stride(n_frames, count)
        return indices, f"even stride of {len(indices)}"

    if mode != "grasp":
        raise ValueError(f"unknown frame selection {mode!r}")

    if gripper is None:
        indices = _stride(n_frames, count)
        return (
            indices,
            f"no gripper channel found; even stride of {len(indices)}",
        )
    closed = np.flatnonzero(np.asarray(gripper) >= threshold)
    if closed.size == 0:
        indices = _stride(n_frames, count)
        return (
            indices,
            f"gripper never closes (threshold {threshold}); even stride of {len(indices)}",
        )

    first = int(closed[0])
    half = count // 2
    start = max(0, min(first - half, n_frames - count))
    indices = list(range(start, min(start + count, n_frames)))
    return indices, f"gripper closure at frame {first} (threshold {threshold})"


def _stride(n_frames: int, count: int) -> list[int]:
    if count <= 0 or n_frames <= 0:
        return []
    step = max(1, n_frames // count)
    return [i * step for i in range(count) if i * step < n_frames]
