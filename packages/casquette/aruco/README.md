# Grabette ArUco cube markers

Printable ArUco cube nets for tracking a grabette with the casquette camera.
Tracking itself (board calibration + pose) lives in the `grabette-slam` repo; this
folder only holds the printable markers and their generator.

## The cube

- 55 mm cube, white, with a shallow 50 mm indent on the 4 side faces (alignment aid).
- Continuous plus-net, folded onto the cube:
  - **centre = OUT face** (55 mm, no indent) — faces the camera, carries the small ▲ "top" mark.
  - **4 arms = side faces** (50 mm tiles seated in the indents).
  - **6th face (opposite centre) = bare MOUNT** — glued to the grabette.
- Marker = **45 mm**, `DICT_4X4_50`. The 45 mm size is fixed: it is the calibration
  scale (`cube_board.S` in grabette-slam). Only the border differs (centre 5 mm, arms
  2.5 mm on the 3 free sides + 5 mm at the fold).

Marker orientation on the cube is **not** controlled precisely — board calibration
learns the true per-marker geometry from a clip. The ▲ only defines a human "up".

## ID allocation (DICT_4X4_50 → 10 cubes max)

5 ids per cube, contiguous blocks: `block = id // 5`, `face_role = id % 5`
(role 0 = OUT/reference, 1 = top, 2 = bottom, 3 = left, 4 = right).

| cube | ids   | notes                 |
|------|-------|-----------------------|
| L0   | 0–4   | left grabette, pair 0 |
| R0   | 5–9   | right grabette, pair 0|
| L1   | 10–14 | left, pair 1          |
| R1   | 15–19 | …                     |

Left/right are the same physical net, different id blocks — handedness lives in the
cube→grabette extrinsic, not the marker pattern. Multiple cubes in one view are
disambiguated by id (`id // 5`).

## Generate

```
uv run --with opencv-contrib-python --with numpy --with pillow \
  python generate_cube_net.py --base-id 0 --label L0
```

`--base-id` must be a multiple of 5 (0..45). Prints at A4, 300 DPI; **print at 100%**
and check the 100 mm scale bar.
