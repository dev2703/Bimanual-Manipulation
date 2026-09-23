"""Run physical core dinner skills from matched starts in one scene revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimanual.experts.aloha_bottle import run_bottle_grasp_lift
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.sim.aloha_env import AlohaTableSettingEnv

EXPERTS = {
    "mug": run_mug_pick_place,
    "plate": run_plate_pick_place,
    "drawer": run_drawer_open,
    "bottle_lift": run_bottle_grasp_lift,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--skills", nargs="+", choices=tuple(EXPERTS), default=list(EXPERTS))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_combined_v2.xml"
    totals = {name: 0 for name in args.skills}
    for index in range(args.episodes):
        seed = args.seed_offset + index
        for name in args.skills:
            env = AlohaTableSettingEnv(scene)
            try:
                env.reset(seed=seed, randomize_objects=True)
                result = EXPERTS[name](env)
            finally:
                env.close()
            totals[name] += bool(result.success)
            print(json.dumps({"seed": seed, "skill": name, "success": result.success}), flush=True)
    rates = {name: totals[name] / args.episodes for name in totals}
    print(json.dumps({"episodes_per_skill": args.episodes, "successes": totals,
                      "success_rates": rates}, sort_keys=True))
    if any(rate < 0.9 for rate in rates.values()):
        raise SystemExit("combined scene core-skill gate is below 90%")


if __name__ == "__main__":
    main()
