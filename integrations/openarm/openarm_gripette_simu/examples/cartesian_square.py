"""Move the arm end-effector along a square trajectory in Cartesian space.

The camera frame traces a 20cm x 20cm square in the YZ plane while
maintaining a fixed orientation.
"""

import time
import numpy as np
import cv2
from openarm_gripette_simu import Simulation, Kinematics

# Render camera at ~30fps, not every physics step
CAMERA_FPS = 30


def interpolate_waypoints(waypoints: list[np.ndarray], n_steps: int) -> list[np.ndarray]:
    """Linearly interpolate joint-space waypoints."""
    trajectory = []
    for i in range(len(waypoints)):
        q_start = waypoints[i]
        q_end = waypoints[(i + 1) % len(waypoints)]
        for t in np.linspace(0, 1, n_steps, endpoint=False):
            trajectory.append(q_start + t * (q_end - q_start))
    return trajectory


def main():
    sim = Simulation()
    kin = Kinematics()

    q_start = np.array([0.0, 0.0, 0.0, 1.57079632679, 0.0, 0.0, 0.0])
    T_center = kin.forward(q_start)
    center_pos = T_center[:3, 3].copy()
    print(f"Square center: {center_pos}")

    half = 0.10
    square_offsets = [
        np.array([0, half, half]),
        np.array([0, half, -half]),
        np.array([0, -half, -half]),
        np.array([0, -half, half]),
    ]

    # Solve IK for each corner (keep orientation fixed)
    corner_joints = []
    q_seed = q_start.copy()
    for i, offset in enumerate(square_offsets):
        T_target = T_center.copy()
        T_target[:3, 3] = center_pos + offset
        q = kin.inverse(T_target, current_joint_positions=q_seed)
        corner_joints.append(q)
        q_seed = q
        T_check = kin.forward(q)
        err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
        print(f"  corner {i}: err={err * 1000:.1f}mm")

    # Build smooth trajectory: interpolate between corners
    steps_per_edge = 200
    trajectory = interpolate_waypoints(corner_joints, steps_per_edge)
    print(f"\nTrajectory: {len(trajectory)} steps, {len(trajectory) * sim.model.opt.timestep:.1f}s")

    # First move from neutral to the start of the square
    viewer = sim.launch_passive_viewer()
    sim.set_arm_commands(corner_joints[0])
    for _ in range(1000):
        sim.step()
    viewer.sync()

    # Run the square trajectory in a loop
    print("Running square trajectory... (press 'q' in camera window to quit)")
    dt = sim.model.opt.timestep
    cam_interval = max(1, int(1.0 / (CAMERA_FPS * dt)))
    step = 0
    t_wall = time.perf_counter()

    while viewer.is_running():
        q_cmd = trajectory[step % len(trajectory)]
        sim.set_arm_commands(q_cmd)
        sim.step()
        step += 1

        # Display at ~30fps: render camera + sync viewer + sleep to real time
        if step % cam_interval == 0:
            viewer.sync()
            img = sim.render_camera()
            cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # Sleep to match real time
            t_target = t_wall + step * dt
            t_now = time.perf_counter()
            if t_target > t_now:
                time.sleep(t_target - t_now)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
