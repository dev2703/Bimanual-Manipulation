"""Held-out contact handoff trials for the ALOHA baton scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimanual.experts.aloha_handoff import run_baton_handoff
from bimanual.sim.aloha_env import AlohaPhysicalEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    args = parser.parse_args()
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_handoff_baton.xml"
    successes = 0
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaPhysicalEnv(scene)
        try:
            env.reset(seed=seed, randomize_block=True)
            result = run_baton_handoff(env)
        finally:
            env.close()
        successes += result.success
        print(json.dumps({"seed": seed, **{k: v for k, v in result.__dict__.items() if k != "record"}}, sort_keys=True))
    rate = successes / args.episodes
    print(json.dumps({"episodes": args.episodes, "successes": successes, "success_rate": rate}, sort_keys=True))
    if rate < 0.9:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
