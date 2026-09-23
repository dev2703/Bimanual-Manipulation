"""Held-out physical drawer-handle contact gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.sim.aloha_env import AlohaTableSettingEnv

SCENE = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_drawer_v2.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    successes = 0
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(SCENE)
        try:
            env.reset(seed=seed, randomize_objects=True)
            result = run_drawer_open(env)
            successes += int(result.success)
            print(
                f"seed={seed} success={result.success} "
                f"peak={result.peak_opening:.3f} final={result.final_opening:.3f} "
                f"handle_contacts={result.handle_contact_steps}"
            )
        except Exception as exc:
            print(f"seed={seed} success=False error={exc}")
        finally:
            env.close()
    rate = successes / args.episodes
    print(f"physical drawer-handle opening: {successes}/{args.episodes} = {rate:.1%}")
    if rate < args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
