"""Regenerate grabette_grasp.xml's free-floating gripper from the grabette_right model.

Structure (re-rooted at the oak_l SLAM frame; weld stays identity so the script's
mocap/freejoint reset commands the oak_l pose directly):

  grabette_root            frame == oak_l, freejoint, welded to mocap (identity relpose)
  └─ grip_r                fixed (body/thumb side, carries the RGB camera + grip_soft_tip_r)
     ├─ gripette_cam        the reoriented RGB camera (verbatim from the model)
     └─ proximal_bend_r     proximal hinge  [-pi/2, 0]
        └─ distal_r         distal hinge    [0, pi/2]   (carries distal_soft_tip)

Every geom/inertial is copied verbatim (each stays in its own body frame); only the
body→body transforms come from MuJoCo FK at q=0. Collision is enabled ONLY on the
two soft tips. Writes grabette_grasp_v2.xml (old scene untouched until verified).
"""
import xml.etree.ElementTree as ET
import numpy as np, mujoco
from pathlib import Path

HERE = Path(__file__).parent
GR = HERE.parents[3] / "packages/grabette/urdf/grabette_right"
OLD = HERE / "grabette_grasp.xml"
OUT = HERE / "grabette_grasp_v2.xml"

m = mujoco.MjModel.from_xml_path(str(GR / "robot.xml"))
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
def q_of(R): q=np.zeros(4); mujoco.mju_mat2Quat(q,R.reshape(9)); return q
def bpose(n): i=m.body(n).id; return d.xpos[i].copy(), q_of(d.xmat[i])
def spose(n): i=m.site(n).id; return d.site_xpos[i].copy(), q_of(d.site_xmat[i])
def inv(p,q): qi=np.zeros(4); mujoco.mju_negQuat(qi,q); pi=np.zeros(3); mujoco.mju_rotVecQuat(pi,-p,qi); return pi,qi
def mul(pa,qa,pb,qb):
    pr=np.zeros(3); mujoco.mju_rotVecQuat(pr,pb,qa); pr+=pa
    qr=np.zeros(4); mujoco.mju_mulQuat(qr,qa,qb); return pr,qr
def rel(a,b): pbi,qbi=inv(*b); return mul(pbi,qbi,*a)   # pose of a in b's frame
def fmt(v): return " ".join(f"{x:.7g}" for x in v)

ROOT_to_oak = spose("oak_l")
grip_in_oak = rel(bpose("grip_r"), ROOT_to_oak)
cam_in_grip = rel(spose("camera"), bpose("grip_r"))
# MuJoCo cameras view along -z; the model's camera site has +z as optical axis.
# Apply a 180-deg-about-x correction so the rendered view faces the opening.
_corr = np.array([0.0, 1.0, 0.0, 0.0])
camq = np.zeros(4); mujoco.mju_mulQuat(camq, cam_in_grip[1], _corr)
# Camera origin is taken verbatim from the model — the camera-translation fix
# is now baked into grabette_right, so no extra forward shift here.
cam_pos = cam_in_grip[0]
prox_in_grip = rel(bpose("proximal_bend_r"), bpose("grip_r"))
dist_in_prox = rel(bpose("distal_r"), bpose("proximal_bend_r"))

grt = ET.parse(GR / "robot.xml").getroot()
def find_body(name):
    return next(b for b in grt.iter("body") if b.get("name") == name)
def inertial_xml(name):
    return ET.tostring(find_body(name).find("inertial"), encoding="unicode").strip()
SOFT = {"grip_soft_tip_r", "distal_soft_tip"}
def geoms_xml(name, indent):
    out = []
    for g in find_body(name).findall("geom"):
        if g.get("class") != "visual":
            continue
        mesh, mat = g.get("mesh"), g.get("material")
        pos, quat = g.get("pos", "0 0 0"), g.get("quat", "1 0 0 0")
        out.append(f'<geom type="mesh" class="visual" pos="{pos}" quat="{quat}" mesh="{mesh}" material="{mat}"/>')
        if mesh in SOFT:
            out.append(f'<geom type="mesh" class="collision" pos="{pos}" quat="{quat}" mesh="{mesh}" material="{mat}"/>')
    return ("\n" + indent).join(out)

def joint_xml(body_name, joint_name):
    j = find_body(body_name).find("joint")   # copy the model's ACTUAL axis + range
    return f'<joint name="{joint_name}" type="hinge" axis="{j.get("axis")}" range="{j.get("range")}"/>'

def site_xml(site_name):
    # Copy a named frame (thumb_tip / finger_tip / gripper_center) verbatim from
    # the model, keeping its local pose in its parent body. group="4" keeps the
    # marker OUT of the rendered camera images (it is a reference frame only,
    # used for measuring tip clearance / grasp geometry, not part of the obs).
    s = next(el for el in grt.iter("site") if el.get("name") == site_name)
    return (f'<site name="{site_name}" pos="{s.get("pos","0 0 0")}" '
            f'quat="{s.get("quat","1 0 0 0")}" group="4"/>')

PI2 = float(np.pi / 2)
gripper = f'''<body name="grabette_root" pos="0.44 -0.142 0.7" quat="0 1 0 0" childclass="grabette">
      <freejoint name="grabette_freejoint"/>
      <inertial pos="0 0 0" mass="1e-09" fullinertia="1e-09 1e-09 1e-09 0 0 0"/>
      <body name="grip_r" pos="{fmt(grip_in_oak[0])}" quat="{fmt(grip_in_oak[1])}">
        {inertial_xml("grip_r")}
        {geoms_xml("grip_r","        ")}
        {site_xml("thumb_tip")}
        {site_xml("gripper_center")}
        <camera name="gripette_cam" pos="{fmt(cam_pos)}" quat="{fmt(camq)}" fovy="130"/>
        <body name="proximal_bend_r" pos="{fmt(prox_in_grip[0])}" quat="{fmt(prox_in_grip[1])}">
          {joint_xml("proximal_bend_r","proximal")}
          {inertial_xml("proximal_bend_r")}
          {geoms_xml("proximal_bend_r","          ")}
          <body name="distal_r" pos="{fmt(dist_in_prox[0])}" quat="{fmt(dist_in_prox[1])}">
            {joint_xml("distal_r","distal")}
            {inertial_xml("distal_r")}
            {geoms_xml("distal_r","            ")}
            {site_xml("finger_tip")}
          </body>
        </body>
      </body>
    </body>'''

# --- splice into a copy of the old scene (keep table/cube/lights/mocap/defaults) ---
tree = ET.parse(OLD); root = tree.getroot()
root.find("compiler").set("meshdir", "../../../../packages/grabette/urdf/grabette_right/assets")
# rebuild <asset>: keep the wood texture + wood/red_cube materials, swap in grabette_right meshes+mats
old_asset = root.find("asset"); src_asset = grt.find("asset")
keep = [el for el in old_asset if (el.tag == "texture") or (el.tag == "material" and not el.get("name", "").endswith("_material"))]
used = {g.get("mesh") for nm in ("grip_r","proximal_bend_r","distal_r") for g in find_body(nm).findall("geom") if g.get("class")=="visual"}
meshes = [el for el in src_asset if el.tag == "mesh" and Path(el.get("file")).stem in used]
mats = [el for el in src_asset if el.tag == "material" and el.get("name","").rsplit("_material",1)[0] in used]
for el in list(old_asset): old_asset.remove(el)
for el in keep + meshes + mats: old_asset.append(el)
# swap the gripper body in worldbody
wb = root.find("worldbody")
# point the mocap (and thus the welded gripper) down by default, so the scene
# opens grasp-ready instead of staring at the sky
for b in wb.iter("body"):
    if b.get("name") == "grabette_mocap":
        b.set("quat", "0 1 0 0")
        g = b.find("geom")
        if g is not None:          # make the marker bigger + opaque so it's visible
            g.set("size", "0.015"); g.set("rgba", "1 0.25 0 1")
for i, b in enumerate(list(wb)):
    if b.tag == "body" and b.get("name") == "grabette_root":
        wb.remove(b); wb.insert(i, ET.fromstring(gripper)); break
ET.indent(tree, space="  ")
tree.write(OUT, encoding="unicode", xml_declaration=False)
print("wrote", OUT)
print("grip_r in oak_l :", fmt(grip_in_oak[0]), "|", fmt(grip_in_oak[1]))
print("cam in grip_r   :", fmt(cam_in_grip[0]), "|", fmt(cam_in_grip[1]))
print("meshes:", len(meshes), "mats:", len(mats), "kept:", [e.get('name') for e in keep])
