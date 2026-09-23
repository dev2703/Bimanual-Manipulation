"""Composed four-core-skill sequence in the combined physical dinner scene.

This is not the full dinner-table task: cutlery, handoff, and pouring are not
executed. The bottle primitive ends in a held lift rather than a pour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bimanual.experts.aloha_bottle import run_bottle_grasp_lift
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.sim.aloha_env import AlohaTableSettingEnv

SEQUENCE = (
    ("drawer", run_drawer_open),
    ("plate", run_plate_pick_place),
    ("mug", run_mug_pick_place),
    ("bottle_lift", run_bottle_grasp_lift),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    args = parser.parse_args()
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_combined_v2.xml"
    success_count = 0
    stage_counts = {name: 0 for name, _ in SEQUENCE}
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(scene)
        stages = {}
        try:
            env.reset(seed=seed, randomize_objects=True)
            for name, expert in SEQUENCE:
                try:
                    stages[name] = bool(expert(env).success)
                except RuntimeError:
                    stages[name] = False
                    break
            state = env.oracle_state()
            plate_site = env.data.site_xpos[env.model.site("plate_region").id]
            mug_site = env.data.site_xpos[env.model.site("mug_region").id]
            retained = bool(
                float(state["drawer_opening"][0]) > 0.11
                and np.linalg.norm(state["plate_pos"][:2] - plate_site[:2]) < 0.04
                and np.linalg.norm(state["mug_pos"][:2] - mug_site[:2]) < 0.045
            )
        finally:
            env.close()
        for name in stages:
            stage_counts[name] += stages[name]
        success = len(stages) == len(SEQUENCE) and all(stages.values()) and retained
        success_count += success
        print(json.dumps({"seed": seed, "stages": stages, "retained": retained,
                          "success": success}, sort_keys=True), flush=True)
    summary = {"episodes": args.episodes, "successes": success_count,
               "success_rate": success_count / args.episodes, "stage_successes": stage_counts}
    print(json.dumps(summary, sort_keys=True))
    if summary["success_rate"] < 0.70:
        raise SystemExit("four-core-skill composition gate is below 70%")


if __name__ == "__main__":
    main()
