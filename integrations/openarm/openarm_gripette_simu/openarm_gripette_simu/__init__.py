"""OpenArm-Gripette MuJoCo simulation package.

Re-exports are lazy (PEP 562): the real-hardware package (openarm_gripette)
imports only Kinematics / proto / rotation from here, and importing those must
not drag in the simulation stack (cv2, MuJoCo) — it isn't needed, and may not
be importable, on robot deployments.
"""

import importlib

# Public name -> submodule that defines it.
_EXPORTS = {
    "Simulation": ".simulation",
    "Kinematics": ".kinematics",
    "FisheyeCamera": ".camera",
    "SimulationServer": ".server",
    "IKFeasibilityChecker": ".ik_feasibility",
    "PoseFeasibility": ".ik_feasibility",
    "TrajectoryFeasibility": ".ik_feasibility",
    "RejectionSamplingStats": ".ik_feasibility",
    "DRConfig": ".domain_randomization",
    "randomize_scene": ".domain_randomization",
    "StartCollisionChecker": ".start_collision",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
