"""Replay a recorded dataset's 10 Hz actions through the 30 Hz controller.

    python -m bimanual.data.replay outputs/aloha_plate_train --skill plate_pick_place

Each episode is reset from its recorded seed, actions are held for three
control steps, and success is judged by the skill's registry check, the same
check that scores experts and policies. Recovery episodes re-apply their
recorded exogenous shift at the recorded event frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from bimanual.data.record import objects_from_features
from bimanual.data.scene_artifacts import assert_scene_compatible
from bimanual.evaluation.skill_gate import reset_for_skill
from bimanual.sim.aloha_env import CONTROL_HZ, TABLE_SETTING_OBJECTS
from bimanual.sim.perturbation import move_ungrasped_object
from bimanual.skills.registry import SKILLS, get_skill

CONTROL_STEPS_PER_FRAME = CONTROL_HZ // 10


@dataclass(frozen=True)
class ReplayResult:
    episode_index: int
    seed: int
    target_name: str
    frames: int
    max_target_error: float
    mean_target_error: float
    mean_joint_max_error: float
    final_success: bool


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _target_error(env, rows: dict[str, list], index: int, target: str, objects: tuple[str, ...]) -> float:
    """Distance between the replayed and recorded tracked quantity at one frame."""
    if target == "drawer":
        return abs(float(env.oracle_state()["drawer_opening"][0]) - _scalar(rows["privileged.drawer_opening"][index]))
    offset = 3 * objects.index(target)
    recorded = np.asarray(rows["privileged.object_positions"][index][offset:offset + 3])
    return float(np.linalg.norm(env.oracle_state()[f"{target}_pos"] - recorded))


def replay_episode(rows: dict[str, list], episode_index: int, skill: str = "mug_pick_place",
                   objects: tuple[str, ...] = tuple(TABLE_SETTING_OBJECTS)) -> ReplayResult:
    spec = get_skill(skill)
    seeds = {int(_scalar(seed)) for seed in rows["privileged.scene_seed"]}
    if len(seeds) != 1:
        raise ValueError(f"episode {episode_index} has inconsistent scene seeds: {seeds}")
    seed = seeds.pop()
    target = spec.object_key[:-len("_pos")] if spec.object_key else "drawer"
    env = spec.make_env()
    try:
        reset_for_skill(spec, env, seed, jitter=0.015)
        state = spec.begin(env)
        target_errors, joint_errors, events = [], [], 0
        for index, (action, joints) in enumerate(zip(rows["action"], rows["observation.state"], strict=True)):
            if spec.recovery and _scalar(rows["privileged.failure_event"][index]) > 0.5:
                move_ungrasped_object(env, target, np.asarray(rows["privileged.perturbation_xy"][index], dtype=np.float64))
                events += 1
            target_errors.append(_target_error(env, rows, index, target, objects))
            joint_errors.append(float(np.max(np.abs(env.state_vector() - np.asarray(joints)))))
            for _ in range(CONTROL_STEPS_PER_FRAME):
                env.step(np.asarray(action, dtype=np.float64))
                spec.update(env, state)
        if spec.recovery and events != 1:
            raise ValueError(f"recovery episode {episode_index} must contain exactly one perturbation event")
        return ReplayResult(
            episode_index, seed, target, len(target_errors), max(target_errors),
            float(np.mean(target_errors)), float(np.mean(joint_errors)), spec.succeeded(env, state),
        )
    finally:
        env.close()


def replay_dataset(root: str | Path, skill: str = "mug_pick_place",
                   max_episodes: int | None = None) -> list[ReplayResult]:
    root = Path(root)
    spec = get_skill(skill)
    assert_scene_compatible(root, spec.scene_path())
    features = json.loads((root / "meta/info.json").read_text())["features"]
    objects = objects_from_features(features)
    columns = ["episode_index", "privileged.scene_seed", "observation.state", "action",
               "privileged.object_positions"]
    columns += [name for name in ("privileged.drawer_opening", "privileged.failure_event",
                                  "privileged.perturbation_xy") if name in features]
    data = ds.dataset(root / "data", format="parquet").to_table(columns=columns).to_pydict()
    episodes: dict[int, dict[str, list]] = {}
    for index, episode in enumerate(data.pop("episode_index")):
        rows = episodes.setdefault(int(episode), {key: [] for key in data})
        for key, column in data.items():
            rows[key].append(column[index])
    selected = sorted(episodes)[:max_episodes]
    return [replay_episode(episodes[index], index, skill, objects) for index in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--skill", choices=sorted(SKILLS), default="mug_pick_place")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-target-error", type=float, default=0.02)
    args = parser.parse_args()
    results = replay_dataset(args.root, args.skill, args.max_episodes)
    for result in results:
        print(result)
    if not results or any(not r.final_success or r.max_target_error > args.max_target_error for r in results):
        raise SystemExit(1)
    print(f"replay passed: {len(results)}/{len(results)} episodes")


if __name__ == "__main__":
    main()
