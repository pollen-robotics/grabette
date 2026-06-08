"""Demonstrate gripper open/close while moving the arm.

Moves the arm to a pose, then cycles the gripper open and closed.
"""

import time
import numpy as np
import cv2
from openarm_gripette_simu import Simulation
from openarm_gripette_simu.simulation import ACTUATOR_NAMES

CAMERA_FPS = 30


def main():
    sim = Simulation()
    viewer = sim.launch_passive_viewer()

    dt = sim.model.opt.timestep
    cam_interval = max(1, int(1.0 / (CAMERA_FPS * dt)))

    # Move arm to a bent pose
    q_arm = np.array([-0.5, -0.3, 0.0, -1.0, 0.0, 0.0, 0.0])
    sim.set_arm_commands(q_arm)
    print("Moving arm to start pose...")
    for _ in range(1000):
        sim.step()
    viewer.sync()

    proximal_idx = ACTUATOR_NAMES.index("proximal")
    distal_idx = ACTUATOR_NAMES.index("distal")

    print("Cycling gripper... (press 'q' in camera window to quit)")
    t_wall = time.perf_counter()
    step = 0
    n_cycles = 0

    while viewer.is_running():
        # Close then open
        for t in np.linspace(0, 2 * np.pi, 600):
            phase = (1 - np.cos(t)) / 2  # 0 → 1 → 0 smoothly
            sim.data.ctrl[proximal_idx] = -1.57 * phase
            sim.data.ctrl[distal_idx] = -2.09 * phase
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

        n_cycles += 1
        print(f"  cycle {n_cycles}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
