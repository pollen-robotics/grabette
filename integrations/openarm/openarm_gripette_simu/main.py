"""Demo: OpenArm + Gripette simulation with joint and Cartesian control."""

import time
import numpy as np
import cv2
from openarm_gripette_simu import Simulation, Kinematics

CAMERA_FPS = 30


def main():
    sim = Simulation()
    kin = Kinematics()

    # Get the camera pose at neutral (all joints at 0)
    neutral_joints = np.zeros(7)
    T_neutral = kin.forward(neutral_joints)
    print("Camera pose at neutral position:")
    print(f"  position: {T_neutral[:3, 3]}")

    # Define a Cartesian target: shift camera 5cm in x
    T_target = T_neutral.copy()
    T_target[0, 3] += 0.05

    # Solve IK from neutral
    target_joints = kin.inverse(T_target, current_joint_positions=neutral_joints)
    print(f"\nIK solution (rad): {target_joints}")

    # Verify FK roundtrip
    T_check = kin.forward(target_joints)
    pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
    print(f"FK verification — position error: {pos_err * 1000:.2f} mm")

    # Launch the passive viewer and send arm commands
    viewer = sim.launch_passive_viewer()
    sim.set_arm_commands(target_joints)
    print("\nMoving arm to Cartesian target... (press 'q' in camera window to quit)")

    dt = sim.model.opt.timestep
    cam_interval = max(1, int(1.0 / (CAMERA_FPS * dt)))
    t_wall = time.perf_counter()
    step = 0

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
