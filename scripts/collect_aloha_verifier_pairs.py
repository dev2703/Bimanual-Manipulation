"""Collect matched RGB positives/negatives at identical robot poses.

Only the manipulated object's pose or drawer slide is changed after the
physical expert has released it. Robot pose, cameras, lighting, and scene seed
are identical within each pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.sim.aloha_env import AlohaTableSettingEnv
from bimanual.sim.perturbation import close_drawer_exogenously, move_ungrasped_object

CAMERAS = ("overhead_cam", "wrist_cam_left", "wrist_cam_right")


def collect(episodes: int, seed_offset: int, output: Path,
            skill: str = "mug_pick_place") -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if skill not in {"mug_pick_place", "plate_pick_place", "drawer_open"}:
        raise ValueError(f"unsupported counterfactual skill {skill!r}")
    object_name = {"mug_pick_place": "mug", "plate_pick_place": "plate", "drawer_open": "drawer"}[skill]
    scene_name = {"mug": "task_table_setting.xml", "plate": "task_table_setting_plate_v2.xml",
                  "drawer": "task_table_setting_drawer_v2.xml"}[object_name]
    scene = Path(__file__).parents[1] / "assets/robots/aloha" / scene_name
    expert = {"mug": run_mug_pick_place, "plate": run_plate_pick_place,
              "drawer": run_drawer_open}[object_name]
    images, labels, seeds = [], [], []
    for index in range(episodes):
        seed = seed_offset + index
        env = AlohaTableSettingEnv(scene)
        try:
            env.reset(seed=seed, randomize_objects=True)
            result = expert(env)
            if not result.success:
                raise RuntimeError(f"physical {object_name} expert failed seed={seed}")
            before = env.render()
            state = env.state_vector().copy()
            if object_name == "drawer":
                close_drawer_exogenously(env)
                if abs(float(env.oracle_state()["drawer_opening"][0])) > 0.01:
                    raise RuntimeError("counterfactual drawer remained open")
            else:
                direction = 1.0 if seed % 2 else -1.0
                move_ungrasped_object(env, object_name, np.array([direction * 0.06, 0.0]))
                target = env.data.site_xpos[env.model.site(f"{object_name}_region").id]
                radius = 0.045 if object_name == "mug" else 0.04
                if np.linalg.norm(env.oracle_state()[f"{object_name}_pos"][:2] - target[:2]) <= radius:
                    raise RuntimeError(f"counterfactual {object_name} remained in target region")
            np.testing.assert_array_equal(env.state_vector(), state)
            after = env.render()
            images.extend((np.stack([before[c] for c in CAMERAS]),
                           np.stack([after[c] for c in CAMERAS])))
            labels.extend((1, 0))
            seeds.extend((seed, seed))
        finally:
            env.close()
        print(json.dumps({"seed": seed, "pair": index + 1}, sort_keys=True), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, images=np.stack(images), labels=np.asarray(labels, dtype=np.int64),
                        seeds=np.asarray(seeds, dtype=np.int64), skill=np.asarray(skill))
    return {"episodes": episodes, "frames": len(labels), "output": str(output), "skill": skill}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--skill", choices=["mug_pick_place", "plate_pick_place", "drawer_open"],
                        default="mug_pick_place")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.episodes, args.seed_offset, args.output, args.skill), sort_keys=True))


if __name__ == "__main__":
    main()
