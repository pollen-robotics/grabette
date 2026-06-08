"""Basic joint-space control: move each joint one by one.

Demonstrates direct joint commanding without IK.
"""

import time
import numpy as np
import cv2
from openarm_gripette_simu import Simulation
from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES

CAMERA_FPS = 30


def main():
    sim = Simulation()
    viewer = sim.launch_passive_viewer()

    dt = sim.model.opt.timestep
    cam_interval = max(1, int(1.0 / (CAMERA_FPS * dt)))

    q_cmd = np.zeros(7)
    joint_targets = [
        ("r_arm_pitch", -0.8),
        ("r_arm_roll", -0.5),
        ("r_arm_yaw", 0.5),
        ("r_elbow", -1.2),
        ("r_wrist_yaw", 0.5),
        ("r_wrist_roll", 0.3),
        ("r_wrist_pitch", -0.5),
    ]

    print("Moving each joint in sequence...")
    t_wall = time.perf_counter()
    step = 0

    for joint_name, target in joint_targets:
        idx = ARM_JOINT_NAMES.index(joint_name)
        q_cmd[idx] = target
        sim.set_arm_commands(q_cmd)
        print(f"  {joint_name} -> {target:.1f} rad")

        for _ in range(500):
            sim.step()
            step += 1
            if step % cam_interval == 0:
                viewer.sync()
                img = sim.render_camera()
                cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                t_target = t_wall + step * dt
                t_now = time.perf_counter()
                if t_target > t_now:
                    time.sleep(t_target - t_now)

    print("\nAll joints moved. Close the viewer to exit.")
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
