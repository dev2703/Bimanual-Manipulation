"""Watch a scripted skill, a policy checkpoint, or just the scene in the MuJoCo viewer.

On macOS run with mjpython so Cocoa owns the main thread:
    mjpython scripts/view.py --skill mug_pick_place --seed 100000
    mjpython scripts/view.py --skill mug_pick_place --checkpoint <pretrained_model dir>
    mjpython scripts/view.py --skill mug_pick_place --scene-only
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from bimanual.evaluation.skill_gate import reset_for_skill
from bimanual.policy.aloha_act_runner import run_aloha_act_episode
from bimanual.policy.registry import load_policy
from bimanual.sim.aloha_env import CONTROL_HZ
from bimanual.skills.registry import SKILLS, get_skill


def _configure_camera(env, viewer) -> None:
    audience_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "audience_cam")
    if audience_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = audience_id
    # Hide Menagerie's capture-rig extrusions; keep robot visuals and dinner dressing.
    viewer.opt.geomgroup[1] = 0
    viewer.opt.geomgroup[5] = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skill", default="mug_pick_place", choices=sorted(SKILLS))
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--checkpoint", help="run this policy checkpoint instead of the scripted expert")
    parser.add_argument("--policy", choices=("act", "smolvla", "pi05", "compact"), default="act")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--scene-only", action="store_true", help="open the scene without acting")
    args = parser.parse_args()
    skill = get_skill(args.skill)
    env = skill.make_env()
    reset_for_skill(skill, env, args.seed, jitter=0.015)
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            _configure_camera(env, viewer)

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
            elif not args.scene_only:
                result = skill.run(env)
                print(f"{skill.name} expert success={result.success}")
            env.after_step = None
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / CONTROL_HZ)
    finally:
        env.close()


if __name__ == "__main__":
    main()
