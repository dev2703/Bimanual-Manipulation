"""Measure physical bottle grasp/lift across randomized dinner scenes."""

from __future__ import annotations

import argparse

from bimanual.experts.aloha_bottle import run_bottle_grasp_lift
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
            result = run_bottle_grasp_lift(env)
            successes += int(result.success)
            print(f"seed={seed} success={result.success} retained_height={result.final_position[2]:.3f}")
        except Exception as exc:
            print(f"seed={seed} success=False error={exc}")
        finally:
            env.close()
    rate = successes / args.episodes
    print(f"physical bottle grasp-lift: {successes}/{args.episodes} = {rate:.1%}")
    if rate < args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
