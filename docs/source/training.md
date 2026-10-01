# Training

The repository ships two policy integrations as worked examples. Both consume the LeRobot dataset built in [Data collection](./data_collection.md#build-the-lerobot-dataset), and share the same dataset preparation. Each one is a **standalone uv project** under `integrations/`, with its own environment and Python 3.12 — a plain `uv sync` inside it is correct and touches nothing else.

Before recording data for training, read the [recording guidelines](./data_collection.md#guidelines-for-recording-a-dataset): how you record shows up in the trained policy.

## DiffusionPolicy

**[integrations/DiffusionPolicy](https://github.com/pollen-robotics/grabette/tree/develop/integrations/DiffusionPolicy)**

Trains a Diffusion Policy on Grabette demonstrations with stock upstream LeRobot, and holds the dataset preparation shared with π0.5.

```bash
cd integrations/DiffusionPolicy
uv sync
./run_pipeline.sh <raw_repo_id>   # QA, cleaning, conversion and resize; prints the train command
```

The pipeline converts the absolute camera poses into camera-local delta actions plus the gripper state, rejects episodes with unrecoverable tracking loss, and keeps only the camera the policy trains on. The README then covers training, the offline checks to run before a robot session, and troubleshooting.

## Pi05

**[integrations/Pi05](https://github.com/pollen-robotics/grabette/tree/develop/integrations/Pi05)**

Fine-tunes [π0.5](https://www.physicalintelligence.company/blog/pi05), a vision-language-action model, on Grabette demonstrations, gates it offline, and runs it on the robot through a remote GPU. It is the verified VLA recipe for Grabette data.

- **Dataset:** produced by the DiffusionPolicy pipeline above, run with `--grasp-projection`.
- **Training:** an A100-80GB class GPU; an HF Jobs run costs about $30 and 12 h.
- **Inference:** about 10 GB of GPU memory, on the robot machine or remotely.

The README walks through the workflow cheapest test first, robot last: a port smoke test, a training smoke run, the full training, then generation and language gates before any robot time.
