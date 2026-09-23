"""Compare stale versus replanned physical plate grasps after object movement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--shift", type=float, default=0.035)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.shift <= 0.06:
        parser.error("episodes must be positive and shift must be in (0, 0.06]")
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_plate_v2.xml"
    rows = []
    for index in range(args.episodes):
        seed = args.seed_offset + index
        angle = np.random.default_rng(seed).uniform(-np.pi, np.pi)
        shift = args.shift * np.array([np.cos(angle), np.sin(angle)])
        for replan in (False, True):
            env = AlohaTableSettingEnv(scene)
            try:
                env.reset(seed=seed, randomize_objects=True)
                result = run_plate_pick_place(
                    env, perturbation_xy=shift, replan_after_perturb=replan,
                )
            finally:
                env.close()
            row = {"seed": seed, "replanned": replan, "success": result.success,
                   "shift_xy": shift.tolist()}
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    rates = {str(replan): sum(r["success"] for r in rows if r["replanned"] == replan) / args.episodes
             for replan in (False, True)}
    print(json.dumps({"episodes": args.episodes, "success_rate_by_replanned": rates}, sort_keys=True))


if __name__ == "__main__":
    main()
