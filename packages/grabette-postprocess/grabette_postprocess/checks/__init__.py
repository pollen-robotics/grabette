"""Episode validation checks, one module per stage of the pipeline:

- recording  : completeness/health of the raw recording (before SLAM)
- sync       : camera ↔ OAK-IMU time alignment (before SLAM)
- trajectory : SLAM odometry quality (after SLAM)

Each module is self-contained and imported on demand (they pull heavy deps —
av / cv2 / scipy), so import the submodule you need rather than the package:
``from grabette_postprocess.checks.recording import check_recording``.
"""
