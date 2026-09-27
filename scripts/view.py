"""Watch one scripted skill, or a checkpoint, in the MuJoCo viewer.

On macOS run with mjpython so Cocoa owns the main thread:
    mjpython scripts/view.py --skill mug_pick_place --seed 100000
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from bimanual.policy.aloha_act_runner import run_aloha_act_episode
from bimanual.policy.registry import load_policy
from bimanual.sim.aloha_env import CONTROL_HZ
from bimanual.skills.registry import get_skill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", default="mug_pick_place")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy", choices=("act", "smolvla", "pi05", "compact"), default="act")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    skill = get_skill(args.skill)
    env = skill.make_env()
    if skill.randomize == "block":
        env.reset(seed=args.seed, randomize_block=True)
    else:
        env.reset(seed=args.seed, randomize_objects=True)
    audience_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "audience_cam")
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            if audience_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = audience_id
            viewer.opt.geomgroup[5] = 1

            def sync_viewer() -> None:
                viewer.sync()
                time.sleep(1.0 / CONTROL_HZ)

            env.after_step = sync_viewer
            if args.checkpoint:
                policy, preprocessor, postprocessor, instruction = load_policy(
                    args.policy, args.checkpoint, args.device, prefix=skill.default_prefix, skill=skill.name,
                )
                result = run_aloha_act_episode(
                    env, policy, preprocessor=preprocessor, postprocessor=postprocessor,
                    device=args.device, max_policy_steps=skill.max_policy_steps,
                    skill=skill, instruction=instruction,
                )
                print(f"{skill.name} policy success={result.success}")
            else:
                result = skill.run(env)
                print(f"{skill.name} expert success={result.success}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
