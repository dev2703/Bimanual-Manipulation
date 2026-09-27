"""Closed-loop evaluation for one policy and one skill.

    python scripts/eval_policy.py --policy act --skill mug_pick_place \
        --checkpoint outputs/act_aloha_mug_full/checkpoints/020000/pretrained_model \
        --prefixes 8 20 --episodes 10 --seed-offset 200000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimanual.policy.registry import load_policy
from bimanual.policy.aloha_act_runner import run_aloha_act_episode
from bimanual.skills.registry import get_skill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("act", "smolvla", "pi05", "compact"), required=True)
    parser.add_argument("--skill", default="mug_pick_place")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=200_000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    skill = get_skill(args.skill)
    prefixes = args.prefixes or [skill.default_prefix if args.policy != "compact" else 1]
    summary = {
        "policy": args.policy,
        "skill": skill.name,
        "seed_offset": args.seed_offset,
        "episodes_per_prefix": args.episodes,
        "prefixes": {},
    }
    rows = []
    for prefix in prefixes:
        policy, preprocessor, postprocessor, instruction = load_policy(
            args.policy, args.checkpoint, args.device, prefix=None if args.policy == "compact" else prefix,
        )
        successes = 0
        for index in range(args.episodes):
            seed = args.seed_offset + index
            env = skill.make_env()
            if skill.randomize == "block":
                env.reset(seed=seed, randomize_block=True)
            else:
                env.reset(seed=seed, randomize_objects=True)
            try:
                result = run_aloha_act_episode(
                    env, policy, preprocessor=preprocessor, postprocessor=postprocessor,
                    device=args.device, max_policy_steps=skill.max_policy_steps,
                    task=skill.name, skill=skill,
                    instruction=instruction if args.policy != "act" else None,
                )
            finally:
                env.close()
            row = {"prefix": prefix, "seed": seed, **result.__dict__}
            rows.append(row)
            successes += int(result.success)
            print(json.dumps(row, sort_keys=True), flush=True)
        summary["prefixes"][str(prefix)] = {"successes": successes, "success_rate": successes / args.episodes}
    output = Path(args.output or f"outputs/gates/eval_{args.policy}_{skill.name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
