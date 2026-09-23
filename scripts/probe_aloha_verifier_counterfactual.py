"""Move a placed mug while the robot pose stays fixed; query RGB verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.perception.aloha_verifier import AlohaRGBVerifier, GOAL_BY_SKILL
from bimanual.perception.verifier import PredicateVerifier
from bimanual.sim.aloha_env import AlohaTableSettingEnv
from bimanual.sim.perturbation import close_drawer_exogenously, move_ungrasped_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
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
        raise ValueError("checkpoint must match the selected dinner skill and scene")
    model = PredicateVerifier((predicate,))
    model.load_state_dict(bundle["model"])
    verifier = AlohaRGBVerifier(model, predicate)
    scene = Path(__file__).parents[1] / "assets/robots/aloha" / scene_name
    expert = {"mug": run_mug_pick_place, "plate": run_plate_pick_place,
              "drawer": run_drawer_open}[object_name]
    passed = 0
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(scene)
        try:
            env.reset(seed=seed, randomize_objects=True)
            result = expert(env)
            if not result.success:
                raise RuntimeError(f"{object_name} expert failed on seed {seed}")
            before = verifier(env.render())[predicate]
            state = env.state_vector().copy()
            if object_name == "drawer":
                close_drawer_exogenously(env)
            else:
                move_ungrasped_object(env, object_name, np.array([0.06, 0.0]))
            after = verifier(env.render())[predicate]
            np.testing.assert_array_equal(env.state_vector(), state)
            if object_name == "drawer":
                distance = abs(float(env.oracle_state()["drawer_opening"][0]))
            else:
                target = env.data.site_xpos[env.model.site(f"{object_name}_region").id]
                distance = float(np.linalg.norm(env.oracle_state()[f"{object_name}_pos"][:2] - target[:2]))
        finally:
            env.close()
        if object_name == "drawer":
            if distance >= 0.01:
                raise RuntimeError("counterfactual drawer did not close")
        elif distance <= (0.045 if object_name == "mug" else 0.04):
            raise RuntimeError("6 cm displacement did not leave the target region")
        passed += before and not after
        print(json.dumps({"seed": seed, "before": before, "after": after,
                          "counterfactual_distance": distance}, sort_keys=True))
    print(json.dumps({"episodes": args.episodes, "correct_flip": passed}, sort_keys=True))
    if passed != args.episodes:
        raise SystemExit("RGB verifier failed the moved-object counterfactual")


if __name__ == "__main__":
    main()
