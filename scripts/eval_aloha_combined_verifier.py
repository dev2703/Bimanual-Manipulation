"""RGB-verifier-driven drawer, plate, and mug sequence in one physical scene.

The expert may use simulator pose for scripted control. Goal transitions use
only the three-camera verifier ensemble. Checkpoints trained on isolated scene
variants are evaluated as cross-scene transfer, not assumed compatible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bimanual.evaluation.aloha_executor import run_verified_goals
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.perception.aloha_verifier import AlohaRGBVerifier
from bimanual.perception.verifier import PredicateVerifier
from bimanual.sim.aloha_env import AlohaTableSettingEnv

GOALS = ("drawer_open", "plate_in_region", "mug_in_region")
EXPERTS = {"drawer_open": run_drawer_open,
           "plate_in_region": run_plate_pick_place,
           "mug_in_region": run_mug_pick_place}


def load_ensemble(paths: dict[str, Path]) -> tuple[dict[str, AlohaRGBVerifier], dict[str, str]]:
    verifiers, training_scenes = {}, {}
    for goal, path in paths.items():
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle["predicate"] != goal:
            raise ValueError(f"{path} predicts {bundle['predicate']}, expected {goal}")
        model = PredicateVerifier((goal,))
        model.load_state_dict(bundle["model"])
        verifiers[goal] = AlohaRGBVerifier(model, goal)
        training_scenes[goal] = bundle["scene"]
    return verifiers, training_scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drawer", type=Path, required=True)
    parser.add_argument("--plate", type=Path, required=True)
    parser.add_argument("--mug", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    args = parser.parse_args()
    paths = dict(zip(GOALS, (args.drawer, args.plate, args.mug), strict=True))
    verifiers, training_scenes = load_ensemble(paths)
    print(json.dumps({"evaluation_scene": "task_table_setting_combined_v2.xml",
                      "training_scenes": training_scenes}, sort_keys=True))
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_combined_v2.xml"
    successes = matches = 0
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(scene)
        physical = {}

        def execute_skill(scene_env, goal, _prompt):
            physical[goal] = bool(EXPERTS[goal](scene_env).success)

        def verify_rgb(frames):
            return {goal: adapter(frames)[goal] for goal, adapter in verifiers.items()}

        try:
            env.reset(seed=seed, randomize_objects=True)
            memory, attempts = run_verified_goals(
                env, list(GOALS), execute_skill, verify_rgb,
                instruction="Set the dinner table", max_attempts=1,
            )
            oracle = env.oracle_state()
            retained = bool(float(oracle["drawer_opening"][0]) > 0.11)
            for obj, tol in (("plate", 0.04), ("mug", 0.045)):
                target = env.data.site_xpos[env.model.site(f"{obj}_region").id]
                retained &= bool(np.linalg.norm(oracle[f"{obj}_pos"][:2] - target[:2]) < tol)
        finally:
            env.close()
        rgb = {row.goal: row.verified for row in attempts}
        matched = all(rgb[goal] == physical.get(goal, False) for goal in GOALS)
        success = all(rgb.values()) and all(physical.values()) and retained and len(memory.state.completed) == 3
        matches += matched
        successes += success
        print(json.dumps({"seed": seed, "physical": physical, "rgb": rgb,
                          "retained": retained, "match": matched, "success": success}, sort_keys=True), flush=True)
    summary = {"episodes": args.episodes, "matches": matches, "successes": successes,
               "success_rate": successes / args.episodes}
    print(json.dumps(summary, sort_keys=True))
    if successes < 0.9 * args.episodes or matches < 0.95 * args.episodes:
        raise SystemExit("combined RGB-verifier executor gate remains open")


if __name__ == "__main__":
    main()
