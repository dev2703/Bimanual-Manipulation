from __future__ import annotations

import argparse

from bimanual.experts.aloha_block import run_block_grasp_lift
from bimanual.sim.aloha_env import AlohaPhysicalEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()

    successes = 0
    failures = []
    for seed in range(args.seeds):
        env = AlohaPhysicalEnv()
        env.reset(seed=seed, randomize_block=True)
        try:
            result = run_block_grasp_lift(env)
        except Exception as error:
            failures.append((seed, env.oracle_state()["task_block_pos"].tolist(), str(error)))
            print(f"seed={seed} success=False error={error}")
            continue
        successes += int(result.success)
        if not result.success:
            failures.append((seed, result.initial_position.tolist(), result.final_position.tolist()))
        print(
            f"seed={seed} arm={result.arm} success={result.success} peak={result.peak_height:.3f} "
            f"retained={result.retained_height:.3f}"
        )
    rate = successes / args.seeds
    print(f"ALOHA physical block gate: {successes}/{args.seeds} ({rate:.1%})")
    if failures:
        print(f"failures={failures}")
    raise SystemExit(0 if rate >= 0.90 else 1)


if __name__ == "__main__":
    main()
