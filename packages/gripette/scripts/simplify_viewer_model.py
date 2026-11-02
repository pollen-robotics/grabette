# /// script
# requires-python = ">=3.10"
# dependencies = ["pymeshlab"]
# ///
"""Build the light model the dashboard's 3D viewer loads.

The CAD export in urdf/gripette_<hand>/ is about 37 MB of STL — screw threads,
circuit boards — which the viewer downloads from the Pi every time it opens.
Same recipe as packages/grabette/scripts/simplify_viewer_model.py.
This writes urdf/gripette_<hand>_viewer/: the same links and joints, only the
<visual> meshes, without the parts nobody can see from outside, and every
remaining mesh decimated. The full model stays the reference (MuJoCo, sim);
re-run this whenever it changes.

Usage (from packages/gripette):
    uv run scripts/simplify_viewer_model.py            # both hands
    uv run scripts/simplify_viewer_model.py --hand right
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pymeshlab

URDF_DIR = Path(__file__).resolve().parent.parent / "urdf"

# Hidden once the gripette is assembled: the four Phillips screws inside the
# shell, as on the grabette.
DROPPED = (
    "602_00009_phillips_flat_head",
)

# Triangle budgets. Hardware is small on screen and repeated (16 torx screws,
# 18 Feetech screws), so it gets far fewer triangles than the shells and
# fingers.
HARDWARE = ("screw", "torx", "part_1__configuration_copy_of_2x8")
HARDWARE_MAX_FACES = 300
PART_MAX_FACES = 5000


def _dropped(name: str) -> bool:
    return name.startswith(DROPPED)


def _simplify(src: Path, dst: Path) -> tuple[int, int]:
    # Quadric edge collapse from MeshLab: simpler decimators stall on the CAD
    # export (the front cover stopped at 31k of its 118k triangles).
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    # STL repeats every vertex per triangle; welding them is what lets the
    # decimation collapse edges at all.
    ms.meshing_remove_duplicate_vertices()
    before = ms.current_mesh().face_number()
    budget = (HARDWARE_MAX_FACES if any(h in src.name for h in HARDWARE)
              else PART_MAX_FACES)
    if before > budget:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=budget, preservenormal=True, qualitythr=0.3)
    ms.save_current_mesh(str(dst), binary=True)
    return before, ms.current_mesh().face_number()


def build(hand: str) -> None:
    src_dir = URDF_DIR / f"gripette_{hand}"
    out_dir = URDF_DIR / f"gripette_{hand}_viewer"
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "assets").mkdir(parents=True)

    tree = ET.parse(src_dir / "robot.urdf")
    meshes = set()
    for link in tree.getroot().iter("link"):
        # The viewer reads <visual> only; links stay even when emptied, since
        # the joints name them.
        for el in list(link):
            if el.tag == "collision":
                link.remove(el)
            elif el.tag == "visual":
                mesh = el.find("geometry/mesh")
                if mesh is None:
                    continue
                name = mesh.get("filename").rsplit("/", 1)[-1]
                if _dropped(name):
                    link.remove(el)
                else:
                    meshes.add(name)
    tree.write(out_dir / "robot.urdf", encoding="utf-8", xml_declaration=True)

    size_in = size_out = 0
    for name in sorted(meshes):
        src, dst = src_dir / "assets" / name, out_dir / "assets" / name
        before, after = _simplify(src, dst)
        size_in += src.stat().st_size
        size_out += dst.stat().st_size
        print(f"  {name[:60]:60} {before:7d} -> {after:5d} tri")
    total = sum(f.stat().st_size for f in (src_dir / "assets").iterdir())
    print(f"{hand}: {total / 1e6:.1f} MB -> {size_out / 1e6:.2f} MB "
          f"(kept parts were {size_in / 1e6:.1f} MB)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hand", choices=("right", "left"))
    args = parser.parse_args()
    for hand in [args.hand] if args.hand else ["right", "left"]:
        build(hand)


if __name__ == "__main__":
    main()
