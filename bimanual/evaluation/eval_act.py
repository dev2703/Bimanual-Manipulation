"""Phase 4 gate measurement: ACT closed-loop success on pick-plate at L0
and L1 (docs/phases.md).

Read this before trusting any number it prints: as of this writing,
BimanualTableEnv only attaches a held object via the SCRIPTED
env.grasp()/env.release() calls made by experts/executor.py based on
phase labels (see env.py's `_held` docstring). ACT's action space is
pure joint targets with no such call and no automatic contact-based
attach -- confirmed empirically (docs/decisions.md Phase 4 notes) by
replaying the exact expert joint trajectory used to generate the
training data WITHOUT calling env.grasp(): the object's z position does
not change at all (contact friction alone provides ~0 holding force).
This means ACT closed-loop success is expected to be near 0% regardless
of imitation quality, until the environment's grasp mechanism is either
made contact/proximity-based (policy-agnostic) or ACT's action space is
extended with an explicit discrete grasp signal wired to env.grasp().
Run this script anyway to get the actual measured number rather than
assume it -- but a low score here is diagnostic of the environment gap
above, not necessarily of ACT's trajectory-following quality.
"""

from __future__ import annotations

import argparse

from bimanual.logging_utils import get_logger
from bimanual.policy.act_runner import load_act_bundle, run_closed_loop_episode
from bimanual.sim.env import BimanualTableEnv, ResetOptions
from bimanual.sim.randomization import sample_scene_config

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a lerobot ACT checkpoint dir (pretrained_model)")
    parser.add_argument("--level", default="L1", choices=["L0", "L1"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    policy, preprocessor, postprocessor = load_act_bundle(args.checkpoint, args.device)

    successes = 0
    for seed in range(args.seeds):
        cfg = sample_scene_config(args.level, seed=seed)
        env = BimanualTableEnv(ResetOptions(scene_cfg=cfg))
        ok = run_closed_loop_episode(
            env, policy, obj="plate", region="plate_region",
            max_steps=args.max_steps, device=args.device,
            preprocessor=preprocessor, postprocessor=postprocessor,
        )
        successes += int(ok)
        log.info("seed %d: %s", seed, "SUCCESS" if ok else "fail")

    log.info("ACT closed-loop success at %s: %d/%d (%.1f%%)", args.level, successes, args.seeds, 100 * successes / args.seeds)


if __name__ == "__main__":
    main()
