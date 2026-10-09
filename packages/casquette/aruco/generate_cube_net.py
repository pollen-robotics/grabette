"""Parameterized plus-net (cube unfolding) for a grabette ArUco cube.

Prints 5 markers as a plus/cross so folding guarantees consistent orientation
(no arrows). Layout, with b = --base-id (a multiple of 5):

               [ b+1  TOP ]
   [ b+3 LEFT ][ b+0 OUT  ][ b+4 RIGHT ]
               [ b+2  BOT ]

  center  b+0 = outward / reference face (faces the casquette)
  arms    b+1..b+4 fold down 90 deg onto the four side faces
  the 6th face (opposite centre) stays bare = glued to the grabette (mount)

Cut on the solid outline, fold on the dashed lines. Cube face = 55 mm,
marker = 45 mm (matches cube_board.S), DICT_4X4_50, A4 @ 300 DPI.

  python generate_cube_net.py --base-id 0 --label L0
  python generate_cube_net.py --base-id 5 --label R0
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--base-id", type=int, required=True, help="first id of the 5-id block (multiple of 5)")
ap.add_argument("--label", required=True, help="cube label, e.g. L0 / R0")
ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args()
assert a.base_id % 5 == 0 and 0 <= a.base_id <= 45, "base-id must be a multiple of 5 in 0..45"

DPI = 300
PXMM = DPI / 25.4
def mm(x): return int(round(x * PXMM))

MARK = mm(45)                        # 45mm marker (= cube_board.S, calibration scale)
F = mm(55)                           # centre/OUT face: full 55mm, NO indent -> folds land on cube edges
bc, ba = (F - MARK) // 2, (mm(50) - MARK) // 2   # centre border 5mm; arm free-side border 2.5mm
Wa = MARK + 2 * ba                   # arm width  = 50mm (2.5mm each side)
La = bc + MARK + ba                  # arm length = 52.5mm (5mm root at fold + marker + 2.5mm tip)
dw = (F - Wa) // 2                   # 2.5mm arm width inset vs the 55mm centre edge
adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
b = a.base_id

W, H = mm(210), mm(297)
canvas = np.full((H, W), 255, np.uint8)

# centre (OUT) 55mm square, horizontally centred; arms attach on its edges, 5mm white at each fold
X = (W - F) // 2
Y = mm(125)

# id -> marker top-left: centre 45mm in 55mm; arms 5mm from the fold, 2.5mm on the 3 free sides
MARKERS = {
    b + 0: (X + bc, Y + bc),                   # centre / OUT / ref
    b + 1: (X + dw + ba, Y - bc - MARK),       # top arm    (marker 5mm above fold Y)
    b + 2: (X + dw + ba, Y + F + bc),          # bottom arm (marker 5mm below fold Y+F)
    b + 3: (X - bc - MARK, Y + dw + ba),       # left arm   (marker 5mm left of fold X)
    b + 4: (X + F + bc, Y + dw + ba),          # right arm  (marker 5mm right of fold X+F)
}
# markers only; NO text in the quiet zone. orientation is learned by board calibration.
for mid, (tx, ty) in MARKERS.items():
    img = cv2.aruco.generateImageMarker(adict, mid, MARK)
    canvas[ty:ty + MARK, tx:tx + MARK] = img

# single "top" mark: tiny OUTLINE triangle in the OUT face top border (at the fold line)
cx = X + F // 2
top_mark = np.array([[cx, Y + mm(0.6)], [cx - mm(1.2), Y + mm(2.6)], [cx + mm(1.2), Y + mm(2.6)]], np.int32)
cv2.polylines(canvas, [top_mark], True, 0, 1)

# cut outline (solid): clean plus. Adjacent arm edges run to their intersection (the
# armpit), one corner each, clipping the centre's 2.5mm corner tabs -> nothing fiddly to cut.
x0, x1 = X + dw, X + dw + Wa          # arm span in x (= X+dw .. X+F-dw)
y0, y1 = Y + dw, Y + dw + Wa          # arm span in y
perim = np.array([
    [x0, Y - La], [x1, Y - La], [x1, y0],                 # top arm + TR armpit
    [X + F + La, y0], [X + F + La, y1], [x1, y1],         # right arm + BR armpit
    [x1, Y + F + La], [x0, Y + F + La], [x0, y1],         # bottom arm + BL armpit
    [X - La, y1], [X - La, y0], [x0, y0],                 # left arm + TL armpit
], np.int32)
cv2.polylines(canvas, [perim], True, 0, 2)

# fold lines (dashed) = the centre square's four edges (the cube edges)
def dashed(p, q):
    p, q = np.array(p, float), np.array(q, float)
    L = np.linalg.norm(q - p); dd = (q - p) / L; step = mm(3)
    for s in range(0, int(L), 2 * step):
        s2 = min(s + step, L)
        cv2.line(canvas, tuple((p + dd * s).astype(int)), tuple((p + dd * s2).astype(int)), 0, 1)

for p, q in [((x0, Y), (x1, Y)), ((x0, Y + F), (x1, Y + F)),
             ((X, y0), (X, y1)), ((X + F, y0), (X + F, y1))]:
    dashed(p, q)

# header + mount note + label
cv2.putText(canvas, f"grabette cube net  {a.label}  ids {b}-{b+4}  DICT_4X4_50  centre 55 / arms 50 / marker 45 mm",
            (mm(12), mm(14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
cv2.putText(canvas, "cut solid, fold dashed (cube edges). centre id=OUT faces camera. opposite face (6th) = bare MOUNT.",
            (mm(12), mm(22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
cv2.putText(canvas, a.label, (mm(15), mm(50)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
cv2.putText(canvas, "triangle = TOP of OUT face", (mm(15), mm(58)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)

ry, rx = H - mm(14), (W - mm(100)) // 2
cv2.putText(canvas, "PRINT AT 100% - this line = 100 mm:", (rx, ry - mm(5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
cv2.line(canvas, (rx, ry), (rx + mm(100), ry), 0, 2)
for t in range(0, 101, 10):
    cv2.line(canvas, (rx + mm(t), ry - mm(2)), (rx + mm(t), ry + mm(2)), 0, 2)

im = Image.fromarray(canvas)
png = os.path.join(a.out, f"cube_net_{a.label}.png")
pdf = os.path.join(a.out, f"cube_net_{a.label}.pdf")
im.save(png, dpi=(DPI, DPI))
im.convert("RGB").save(pdf, "PDF", resolution=float(DPI))
print("saved:", png, "|", pdf)
