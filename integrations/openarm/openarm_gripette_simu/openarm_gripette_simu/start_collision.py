"""Geometric start-pose collision check for the free-floating Grabette gripper.

The free-floating grasp scene enables *physics* collision only on the two soft
finger tips, so the gripper body and fingers can visually clip the table top
without ever generating a MuJoCo contact. To reject physically-implausible
start poses (the gripper sitting inside the table), we therefore test the actual
gripper mesh vertices against the table box geometrically rather than relying on
contact detection.

This is a START-pose guard only: it is cheap (one ``mj_kinematics`` call per
query) and is used to rejection-sample home poses in the data-collection loop.
"""
from pathlib import Path

import numpy as np
import mujoco

# Bodies that make up the free-floating gripper in grabette_grasp.xml.
GRIPPER_BODIES = ("grip_r", "proximal_bend_r", "distal_r")


class StartCollisionChecker:
    """Tests whether a commanded gripper-root pose puts any gripper mesh vertex
    inside the table box. Loads the free-floating scene once and caches each
    gripper mesh's local vertices, so each query is a single kinematics pass.
    """

    def __init__(self, scene_xml: str | Path,
                 gripper_bodies: tuple[str, ...] = GRIPPER_BODIES,
                 table_geom: str = "tabletop",
                 margin: float = 0.002,
                 tip_sites: tuple[str, ...] = ("finger_tip", "thumb_tip"),
                 tip_clearance: float = 0.01):
        self._m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self._d = mujoco.MjData(self._m)
        self._fj = self._m.joint("grabette_freejoint").qposadr[0]
        self._margin = float(margin)
        self._tip_clearance = float(tip_clearance)
        # Tip frames (if present): reject starts where a fingertip is below the
        # table top by less than tip_clearance. The mesh-vs-footprint test below
        # misses a tip hanging just past the table EDGE (no box under it), which
        # still reads as a "low" start; this catches those too.
        self._tip_ids = [self._m.site(n).id for n in tip_sites
                         if mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_SITE, n) >= 0]

        # Table box footprint + top surface (world frame; static geometry, so
        # read once after a forward pass).
        mujoco.mj_forward(self._m, self._d)
        tg = self._m.geom(table_geom).id
        self._tc = self._d.geom_xpos[tg].copy()      # box center (world)
        self._ts = self._m.geom_size[tg].copy()      # box half-sizes

        # Cache each gripper mesh geom's local vertices.
        bids = {self._m.body(b).id for b in gripper_bodies}
        self._geoms: list[tuple[int, np.ndarray]] = []
        for g in range(self._m.ngeom):
            if (self._m.geom(g).bodyid[0] in bids
                    and self._m.geom(g).type[0] == mujoco.mjtGeom.mjGEOM_MESH):
                mid = self._m.geom_dataid[g]
                v0 = self._m.mesh_vertadr[mid]
                nv = self._m.mesh_vertnum[mid]
                verts = self._m.mesh_vert[v0:v0 + nv].reshape(-1, 3).copy()
                self._geoms.append((g, verts))

    def collides(self, home_xyz: np.ndarray, home_quat: np.ndarray) -> bool:
        """True if the gripper at (home_xyz, home_quat) penetrates the table OR
        sits too low (a fingertip within ``tip_clearance`` of / below the table
        top). Rejecting the latter keeps the recorded start clearly above the
        table, not skimming it.

        A vertex counts as penetrating when it lies within the table's xy
        footprint and below the table top by more than ``margin`` — i.e. the
        gripper would have to pass through the table surface to be there.
        """
        d = self._d
        d.qpos[self._fj:self._fj + 3] = home_xyz
        d.qpos[self._fj + 3:self._fj + 7] = home_quat
        mujoco.mj_kinematics(self._m, d)

        tc, ts = self._tc, self._ts
        table_top = tc[2] + ts[2]
        # Fingertip clearance: reject if any tip is below table_top + clearance.
        for sid in self._tip_ids:
            if d.site_xpos[sid][2] < table_top + self._tip_clearance:
                return True

        top = table_top - self._margin
        for g, verts in self._geoms:
            R = d.geom_xmat[g].reshape(3, 3)
            w = verts @ R.T + d.geom_xpos[g]
            inside_xy = ((np.abs(w[:, 0] - tc[0]) <= ts[0])
                         & (np.abs(w[:, 1] - tc[1]) <= ts[1]))
            if np.any(inside_xy & (w[:, 2] < top)):
                return True
        return False
