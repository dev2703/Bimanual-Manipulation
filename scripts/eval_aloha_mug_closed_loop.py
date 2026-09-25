"""Closed-loop MuJoCo task-success comparison: SmolVLA vs. compact-160M on the
dinner-mug task. The saved held-out metrics only cover open-loop first-action
error on 80 expert frames (rollout_success_evaluated: false in
heldout_metrics.json); this measures actual rollout success in sim.
"""

from __future__ import annotations

import argparse
import json

from bimanual.policy.compact_vla_runner import load_compact_bundle, run_compact_mug_episode
from bimanual.policy.pi05_runner import load_pi05_bundle, run_pi05_mug_episode
from bimanual.policy.smolvla.runner import load_smolvla_bundle, run_smolvla_mug_episode
from bimanual.sim.aloha_env import AlohaTableSettingEnv

POLICIES = ("smolvla", "compact_160m", "pi05")


def run_policy(name: str, checkpoint: str, episodes: int, seed_offset: int, device: str, max_policy_steps: int) -> list[dict]:
    if name == "smolvla":
        policy, preprocessor, postprocessor = load_smolvla_bundle(checkpoint, device)
    elif name == "compact_160m":
        policy, preprocessor, postprocessor = load_compact_bundle(checkpoint, device)
    else:
        policy, preprocessor, postprocessor = load_pi05_bundle(checkpoint, device)

    results = []
    for index in range(episodes):
        seed = seed_offset + index
        env = AlohaTableSettingEnv()
        env.reset(seed=seed, randomize_objects=True)
        try:
            if name == "smolvla":
                result = run_smolvla_mug_episode(
                    env, policy, preprocessor, postprocessor,
                    device=device, max_policy_steps=max_policy_steps,
                )
            elif name == "compact_160m":
                result = run_compact_mug_episode(
                    env, policy, preprocessor,
                    device=device, max_policy_steps=max_policy_steps,
                )
            else:
                result = run_pi05_mug_episode(
                    env, policy, preprocessor, postprocessor,
                    device=device, max_policy_steps=max_policy_steps,
                )
        finally:
            env.close()
        row = {"policy": name, "seed": seed, **result.__dict__}
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smolvla-checkpoint", required=True)
    parser.add_argument("--compact-checkpoint", required=True)
    parser.add_argument("--pi05-checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=200_000)
    parser.add_argument("--max-policy-steps", type=int, default=240)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="outputs/aloha_mug_closed_loop.json")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    checkpoints = {
        "smolvla": args.smolvla_checkpoint,
        "compact_160m": args.compact_checkpoint,
        "pi05": args.pi05_checkpoint,
    }
    policies = POLICIES if args.pi05_checkpoint else POLICIES[:-1]
    summary = {"metric": "closed-loop MuJoCo task success", "episodes_per_policy": args.episodes}
    all_rows = []
    for name in policies:
        rows = run_policy(name, checkpoints[name], args.episodes, args.seed_offset, args.device, args.max_policy_steps)
        all_rows.extend(rows)
        successes = sum(row["success"] for row in rows)
        summary[name] = {"successes": successes, "success_rate": successes / len(rows)}

    with open(args.output, "w") as handle:
        json.dump({"summary": summary, "episodes": all_rows}, handle, indent=2)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
