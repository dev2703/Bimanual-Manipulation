"""Physical mug expert controlled by RGB verifier and task memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bimanual.evaluation.aloha_executor import run_verified_goals
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.perception.aloha_verifier import AlohaRGBVerifier, GOAL_BY_SKILL
from bimanual.perception.verifier import PredicateVerifier
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--skill", choices=["mug_pick_place", "plate_pick_place", "drawer_open"],
                        default="mug_pick_place")
    args = parser.parse_args()
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    object_name = {"mug_pick_place": "mug", "plate_pick_place": "plate", "drawer_open": "drawer"}[args.skill]
    scene_name = {"mug": "task_table_setting.xml", "plate": "task_table_setting_plate_v2.xml",
                  "drawer": "task_table_setting_drawer_v2.xml"}[object_name]
    predicate = GOAL_BY_SKILL[args.skill]
    if bundle["predicate"] != predicate or bundle["scene"] != scene_name:
        raise ValueError("checkpoint does not match the selected physical scene")
    model = PredicateVerifier((predicate,))
    model.load_state_dict(bundle["model"])
    verifier = AlohaRGBVerifier(model, predicate)
    expert = {"mug": run_mug_pick_place, "plate": run_plate_pick_place,
              "drawer": run_drawer_open}[object_name]
    scene = Path(__file__).parents[1] / "assets/robots/aloha" / scene_name
    matches = successes = 0
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(scene)
        physical_success = False

        def execute_skill(scene, goal, prompt):
            nonlocal physical_success
            assert goal == predicate and f"Now: {predicate.replace('_', ' ')}" in prompt
            physical_success = expert(scene).success

        try:
            env.reset(seed=seed, randomize_objects=True)
            memory, attempts = run_verified_goals(
                env, [predicate], execute_skill, verifier,
                instruction="Set the dinner table", max_attempts=1,
            )
        finally:
            env.close()
        verified = attempts[0].verified
        matches += verified == physical_success
        successes += physical_success and verified and memory.state.completed == [predicate]
        print(json.dumps({"seed": seed, "physical_success": physical_success,
                          "rgb_verified": verified}, sort_keys=True))
    summary = {"episodes": args.episodes, "expert_and_verifier_success": successes,
               "verifier_matches_physical": matches}
    print(json.dumps(summary, sort_keys=True))
    if successes != args.episodes or matches != args.episodes:
        raise SystemExit("verified executor did not match physical mug expert")


if __name__ == "__main__":
    main()
