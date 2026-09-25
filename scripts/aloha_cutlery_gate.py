"""Held-out physical gate for fork and spoon placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimanual.experts.aloha_cutlery import run_fork_place, run_spoon_place
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    args = parser.parse_args()
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_combined_v2.xml"
    successes = 0
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(scene)
        try:
            env.reset(seed=seed, randomize_objects=True)
            drawer = run_drawer_open(env)
            fork = run_fork_place(env)
            spoon = run_spoon_place(env)
        finally:
            env.close()
        success = drawer.success and fork.success and spoon.success
        successes += success
        print(json.dumps({"seed": seed, "drawer": drawer.success, "fork": fork.success,
                          "spoon": spoon.success, "success": success}), flush=True)
    rate = successes / args.episodes
    print(json.dumps({"episodes": args.episodes, "successes": successes, "success_rate": rate}))
    if rate < 0.9:
        raise SystemExit("cutlery gate is below 90%")


if __name__ == "__main__":
    main()
