"""Run an ALOHA ACT checkpoint in the interactive MuJoCo viewer.

macOS requires mjpython so Cocoa owns the main thread.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from bimanual.policy.act_runner import load_act_bundle
from bimanual.policy.aloha_act_runner import run_aloha_act_episode
from bimanual.sim.aloha_env import CONTROL_HZ, AlohaPhysicalEnv, AlohaTableSettingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--task", choices=["block_lift", "mug_pick_place"], default="block_lift")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-policy-steps", type=int, default=None)
    args = parser.parse_args()

    policy, preprocessor, postprocessor = load_act_bundle(args.checkpoint, args.device)
    if args.task == "mug_pick_place":
        env = AlohaTableSettingEnv()
        env.reset(seed=args.seed, randomize_objects=True)
    else:
        env = AlohaPhysicalEnv()
        env.reset(seed=args.seed, randomize_block=True)
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

            result = run_aloha_act_episode(
                env,
                policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=args.device,
                max_policy_steps=args.max_policy_steps or (240 if args.task == "mug_pick_place" else 160),
                task=args.task,
                after_control_step=sync_viewer,
            )
            print(result)
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 30)
    finally:
        env.close()


if __name__ == "__main__":
    main()
