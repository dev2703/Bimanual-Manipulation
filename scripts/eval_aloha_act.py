"""Evaluate an ACT checkpoint on held-out physical ALOHA skill scenes."""

from __future__ import annotations

import argparse
import json

from bimanual.policy.act_runner import load_act_bundle
from bimanual.policy.aloha_act_runner import run_aloha_act_episode
from bimanual.sim.aloha_env import AlohaPhysicalEnv, AlohaTableSettingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--task", choices=["block_lift", "mug_pick_place"], default="block_lift")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--success-threshold", type=float, default=0.5)
    args = parser.parse_args()

    policy, preprocessor, postprocessor = load_act_bundle(args.checkpoint, args.device)
    results = []
    for index in range(args.episodes):
        seed = args.seed_offset + index
        if args.task == "mug_pick_place":
            env = AlohaTableSettingEnv()
            env.reset(seed=seed, randomize_objects=True)
        else:
            env = AlohaPhysicalEnv()
            env.reset(seed=seed, randomize_block=True)
        try:
            result = run_aloha_act_episode(
                env,
                policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=args.device,
                max_policy_steps=240 if args.task == "mug_pick_place" else 160,
                task=args.task,
            )
        finally:
            env.close()
        row = {"seed": seed, **result.__dict__}
        results.append(row)
        print(json.dumps(row, sort_keys=True))

    successes = sum(item["success"] for item in results)
    rate = successes / len(results)
    print(json.dumps({"episodes": len(results), "successes": successes, "success_rate": rate}, sort_keys=True))
    if rate < args.success_threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
