"""Run the held-out contact plate-placement gate in the isolated scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.sim.aloha_env import AlohaTableSettingEnv

SCENE = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_plate_v2.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--jitter", type=float, default=0.015)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    successes = 0
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(SCENE)
        try:
            env.reset(seed=seed, randomize_objects=True, position_jitter=args.jitter)
            result = run_plate_pick_place(env)
            successes += int(result.success)
            error = np.linalg.norm(result.final_position[:2] - result.target_position[:2])
            print(f"seed={seed} success={result.success} xy_error={error:.3f} "
                  f"upright={result.final_upright_cosine:.3f}")
        except Exception as exc:
            print(f"seed={seed} success=False error={exc}")
        finally:
            env.close()
    rate = successes / args.episodes
    print(f"physical plate placement: {successes}/{args.episodes} = {rate:.1%}")
    if rate < args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
