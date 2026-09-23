"""Open the presentation-ready ALOHA task scene without a policy."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from bimanual.sim.aloha_env import AlohaTableSettingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100_000)
    args = parser.parse_args()
    env = AlohaTableSettingEnv()
    env.reset(seed=args.seed, randomize_objects=True)
    audience_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "audience_cam")
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = audience_id
            # Hide Menagerie's capture-rig extrusions for a clean audience
            # composition. Robot visuals (group 2) and dinner dressing
            # (group 5) remain visible.
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[5] = 1
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 30)
    finally:
        env.close()


if __name__ == "__main__":
    main()
