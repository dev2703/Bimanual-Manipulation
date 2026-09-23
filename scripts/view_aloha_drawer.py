"""Watch physical drawer-handle opening in the MuJoCo viewer.

On macOS, run with mjpython so Cocoa owns the main thread.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.sim.aloha_env import CONTROL_HZ, AlohaTableSettingEnv

SCENE = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_drawer_v2.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100_000)
    args = parser.parse_args()
    env = AlohaTableSettingEnv(SCENE)
    env.reset(seed=args.seed, randomize_objects=True)
    audience_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "audience_cam")
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = audience_id
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[5] = 1

            def sync_viewer() -> None:
                viewer.sync()
                time.sleep(1.0 / CONTROL_HZ)

            env.after_step = sync_viewer
            result = run_drawer_open(env)
            env.after_step = None
            print(
                f"drawer open success={result.success} "
                f"final_opening={result.final_opening:.3f} "
                f"handle_contact_steps={result.handle_contact_steps}"
            )
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / CONTROL_HZ)
    finally:
        env.close()


if __name__ == "__main__":
    main()
