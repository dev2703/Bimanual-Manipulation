"""Run the physical dinner-table mug skill gate over randomized scenes."""

from __future__ import annotations

import argparse

from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--jitter", type=float, default=0.015)
    args = parser.parse_args()
    successes = 0
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv()
        try:
            env.reset(seed=seed, randomize_objects=True, position_jitter=args.jitter)
            result = run_mug_pick_place(env)
            successes += int(result.success)
            print(
                f"seed={seed} success={result.success} "
                f"xy_error={((result.final_position[:2] - result.target_position[:2])**2).sum()**0.5:.3f} "
                f"upright={result.final_upright_cosine:.3f}"
            )
        except Exception as exc:
            print(f"seed={seed} success=False error={exc}")
        finally:
            env.close()
    rate = successes / args.episodes
    print(f"physical mug pick-place: {successes}/{args.episodes} = {rate:.1%}")
    if rate < args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
