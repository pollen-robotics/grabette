"""Demo: table scene with a red cube.

Moves the arm to look at the table, then slowly approaches the cube.
Shows how to load a custom scene.
"""

import time
from pathlib import Path
import numpy as np
import cv2
from openarm_gripette_simu import Simulation, Kinematics

SCENE = Path(__file__).parent.parent / "scenes" / "table_red_cube.xml"
CAMERA_FPS = 30


def main():
    sim = Simulation(scene_xml=SCENE)
    kin = Kinematics()
    viewer = sim.launch_passive_viewer()

    dt = sim.model.opt.timestep
    cam_interval = max(1, int(1.0 / (CAMERA_FPS * dt)))

    # Start: arm pointing forward, elbow bent
    # q_start = np.array([0.3, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0])
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    T_start = kin.forward(q_start)
    print(f"Start camera position: {T_start[:3, 3]}")

    # Move to start pose
    sim.set_arm_commands(q_start)
    for _ in range(1000):
        sim.step()


    #init position
    q_start = np.array([1.0, 0.0, 0.0, -2.44, 0.0, 0.0, 0.0])
    T_start = kin.forward(q_start)
    # print(f"Start camera position: {T_start[:3, 3]}")

    # Move to start pose
    sim.set_arm_commands(q_start)
    for _ in range(1000):
        sim.step()



    viewer.sync()

    # Target: camera above the cube, looking down
    cube_pos = sim.data.body("red_cube").xpos.copy()
    print(f"Red cube position: {cube_pos}")

    T_target = T_start.copy()
    T_target[0, 3] = cube_pos[0]
    T_target[2, 3] = cube_pos[2] + 0.15

    target_joints = kin.inverse(T_target, current_joint_positions=q_start)
    T_check = kin.forward(target_joints)
    err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
    print(f"IK target above cube — error: {err * 1000:.1f}mm")

    # Interpolate from start to target
    n_steps = 500
    print("\nMoving above the cube... (press 'q' in camera window to quit)")
    t_wall = time.perf_counter()
    step = 0

    for i in range(n_steps):
        t = i / n_steps
        sim.set_arm_commands(q_start + t * (target_joints - q_start))
        sim.step()
        step += 1

        if step % cam_interval == 0:
            viewer.sync()
            img = sim.render_camera()
            cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
            t_target = t_wall + step * dt
            t_now = time.perf_counter()
            if t_target > t_now:
                time.sleep(t_target - t_now)

    # Hold position
    print("Above the cube. Close viewer to exit.")
    while viewer.is_running():
        sim.step()
        step += 1
        if step % cam_interval == 0:
            viewer.sync()
            img = sim.render_camera()
            cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            t_target = t_wall + step * dt
            t_now = time.perf_counter()
            if t_target > t_now:
                time.sleep(t_target - t_now)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
