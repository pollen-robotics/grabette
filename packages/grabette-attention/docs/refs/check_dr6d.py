"""Are the dr6d channels the first two columns of a rotation matrix?

The 11-dim action is dx,dy,dz + dr6d_0..5 + strategy + closure. To chain
per-step deltas into "where is the grasp point from here", the rotations have
to be composed, which means knowing what those 6 numbers are.

If they are the Zhou et al. 6D representation -- the first two COLUMNS of R,
recovered by Gram-Schmidt -- then for small per-step rotations they must sit
near the identity's first two columns, i.e. mean dr6d ~= [1,0,0, 0,1,0].
Anything else (axis-angle triples, quaternion parts, absolute rather than
delta rotations) will look obviously different.

Also checks that the recovered matrices are actually orthonormal and that
per-step rotation angles are small, since a 50 fps trajectory should not jump.
"""

import numpy as np
import pyarrow.parquet as pq

ACTIONS = (
    "/home/steve/.cache/huggingface/hub/datasets--chouziel--"
    "grabette-sugar-cup-2008_graspproj/snapshots/"
    "96f1794700bccb9c7088100caed3c974aac82ad9/data/chunk-000/file-000.parquet"
)


def six_d_to_matrix(six):
    a1, a2 = six[:3], six[3:]
    b1 = a1 / max(np.linalg.norm(a1), 1e-12)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / max(np.linalg.norm(a2p), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def main() -> None:
    tbl = pq.read_table(ACTIONS, columns=["episode_index", "action"])
    ep = np.asarray(tbl["episode_index"])
    act = np.stack([np.asarray(r) for r in tbl["action"].to_pylist()]).astype(np.float64)
    rows = act[ep == 0]
    six = rows[:, 3:9]

    print(f"episode 0: {len(rows)} frames, action dim {act.shape[1]}\n")
    print("mean dr6d over the episode:")
    print("  ", np.array2string(six.mean(axis=0), precision=4))
    print("identity's first two columns would be [1 0 0 0 1 0]")
    print("std  dr6d:")
    print("  ", np.array2string(six.std(axis=0), precision=4))

    mats = np.stack([six_d_to_matrix(s) for s in six])
    orth = np.array([
        np.abs(m @ m.T - np.eye(3)).max() for m in mats
    ])
    dets = np.array([np.linalg.det(m) for m in mats])
    angles = np.degrees(np.array([
        np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)) for m in mats
    ]))
    print(f"\nrecovered matrices: max |R Rᵀ − I| = {orth.max():.2e}, "
          f"det in [{dets.min():.4f}, {dets.max():.4f}]")
    print(f"per-step rotation angle: median {np.median(angles):.3f}° "
          f"max {angles.max():.3f}°")
    print("  small angles at 50 fps => these are DELTA rotations, as assumed")

    trans = rows[:, :3]
    print(f"\nper-step translation: median |d| "
          f"{np.median(np.linalg.norm(trans, axis=1))*1000:.3f} mm  "
          f"max {np.linalg.norm(trans, axis=1).max()*1000:.2f} mm")


if __name__ == "__main__":
    main()
