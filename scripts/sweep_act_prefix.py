"""Phase 9 Q7: closed-loop success of one ACT checkpoint across execution prefixes.

The chunk length is fixed by training, so this sweeps only how many actions of
each predicted chunk are executed before re-planning. Every prefix sees the
same held-out seeds; chunk-length variation needs separately trained
checkpoints passed as additional runs of this script.
"""

from __future__ import annotations

import argparse
import json

from bimanual.policy.act_runner import load_act_bundle
from bimanual.policy.aloha_act_runner import run_aloha_act_episode
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prefixes", type=int, nargs="+", default=[1, 4, 8, 12, 20])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--max-policy-steps", type=int, default=240)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="outputs/act_prefix_sweep.json")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    policy, preprocessor, postprocessor = load_act_bundle(args.checkpoint, args.device)
    chunk_size = policy.config.chunk_size
    if policy.config.temporal_ensemble_coeff is not None:
        parser.error("checkpoint uses temporal ensembling; execution prefix does not apply")
    invalid = [p for p in args.prefixes if not 1 <= p <= chunk_size]
    if invalid:
        parser.error(f"prefixes {invalid} outside 1..{chunk_size}")

    summary: dict = {
        "metric": "closed-loop mug_pick_place success by ACT execution prefix",
        "checkpoint": args.checkpoint,
        "chunk_size": chunk_size,
        "trained_prefix": policy.config.n_action_steps,
        "episodes_per_prefix": args.episodes,
        "seed_offset": args.seed_offset,
        "prefixes": {},
    }
    rows = []
    for prefix in args.prefixes:
        policy.config.n_action_steps = prefix
        successes = 0
        for index in range(args.episodes):
            seed = args.seed_offset + index
            env = AlohaTableSettingEnv()
            env.reset(seed=seed, randomize_objects=True)
            try:
                result = run_aloha_act_episode(
                    env, policy, preprocessor=preprocessor, postprocessor=postprocessor,
                    device=args.device, max_policy_steps=args.max_policy_steps, task="mug_pick_place",
                )
            finally:
                env.close()
            row = {"prefix": prefix, "seed": seed, **result.__dict__}
            rows.append(row)
            successes += int(result.success)
            print(json.dumps(row, sort_keys=True), flush=True)
        summary["prefixes"][str(prefix)] = {"successes": successes, "success_rate": successes / args.episodes}
        print(json.dumps({"prefix": prefix, **summary["prefixes"][str(prefix)]}), flush=True)

    with open(args.output, "w") as handle:
        json.dump({"summary": summary, "episodes": rows}, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
