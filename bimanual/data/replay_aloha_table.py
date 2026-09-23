"""Replay recorded 10 Hz dinner actions through the 30 Hz ALOHA controller."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from bimanual.data.scene_artifacts import assert_scene_compatible
from bimanual.evaluation.aloha_contacts import right_gripper_touches_drawer_handle
from bimanual.evaluation.aloha_predicates import drawer_opened, mug_placed, plate_placed
from bimanual.sim.aloha_env import TABLE_SETTING_OBJECTS, AlohaTableSettingEnv

SCENES = {
    "mug_pick_place": "task_table_setting.xml",
    "plate_pick_place": "task_table_setting_plate_v2.xml",
    "drawer_open": "task_table_setting_drawer_v2.xml",
}


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


def _replay_drawer_episode(rows: dict[str, list], episode_index: int, seed: int) -> ReplayResult:
    scene = Path(__file__).parents[2] / "assets/robots/aloha" / SCENES["drawer_open"]
    env = AlohaTableSettingEnv(scene)
    try:
        env.reset(seed=seed, randomize_objects=True)
        initial = float(env.oracle_state()["drawer_opening"][0])
        peak = initial
        target_errors = []
        joint_errors = []
        handle_contact_steps = 0
        for action, state, opening, phase in zip(
            rows["action"], rows["observation.state"],
            rows["privileged.drawer_opening"], rows["privileged.phase"],
            strict=True,
        ):
            current = float(env.oracle_state()["drawer_opening"][0])
            target_errors.append(abs(current - float(np.asarray(opening).reshape(-1)[0])))
            joint_errors.append(float(np.max(np.abs(env.state_vector() - np.asarray(state)))))
            peak = max(peak, current)
            for _ in range(3):
                env.step(np.asarray(action, dtype=np.float64))
                if phase == "PULL":
                    handle_contact_steps += int(right_gripper_touches_drawer_handle(env.model, env.data))
        final = float(env.oracle_state()["drawer_opening"][0])
        peak = max(peak, final)
        success = drawer_opened(
            initial, peak, final,
            handle_contact_steps=handle_contact_steps,
            gripper_opening=float(env.state_vector()[13]),
        )
        return ReplayResult(
            episode_index, seed, "drawer", len(target_errors), max(target_errors),
            float(np.mean(target_errors)), float(np.mean(joint_errors)), success,
        )
    finally:
        env.close()


def replay_episode(
    rows: dict[str, list], episode_index: int, task: str = "mug_pick_place",
) -> ReplayResult:
    if task not in SCENES:
        raise ValueError(f"unsupported replay task: {task}")
    seeds = set(int(seed) for seed in rows["privileged.scene_seed"])
    if len(seeds) != 1:
        raise ValueError(f"episode {episode_index} has inconsistent scene seeds: {seeds}")
    seed = seeds.pop()
    if task == "drawer_open":
        return _replay_drawer_episode(rows, episode_index, seed)
    object_name = "mug" if task == "mug_pick_place" else "plate"
    scene = Path(__file__).parents[2] / "assets/robots/aloha" / SCENES[task]
    object_offset = 3 * list(TABLE_SETTING_OBJECTS).index(object_name)
    env = AlohaTableSettingEnv(scene)
    try:
        env.reset(seed=seed, randomize_objects=True)
        initial = env.oracle_state()[f"{object_name}_pos"].copy()
        target = env.data.site_xpos[env.model.site(f"{object_name}_region").id].copy()
        peak = float(initial[2])
        carried = False
        object_errors = []
        joint_errors = []
        for action, state, objects in zip(
            rows["action"], rows["observation.state"], rows["privileged.object_positions"],
            strict=True,
        ):
            position = env.oracle_state()[f"{object_name}_pos"]
            object_errors.append(float(np.linalg.norm(position - np.asarray(objects[object_offset:object_offset + 3]))))
            joint_errors.append(float(np.max(np.abs(env.state_vector() - np.asarray(state)))))
            peak = max(peak, float(position[2]))
            carried |= bool(
                position[2] > initial[2] + (0.08 if object_name == "mug" else 0.06)
                and np.linalg.norm(position[:2] - target[:2]) < (0.05 if object_name == "mug" else 0.04)
            )
            for _ in range(3):
                env.step(np.asarray(action, dtype=np.float64))
        final = env.oracle_state()[f"{object_name}_pos"]
        predicate = mug_placed if object_name == "mug" else plate_placed
        success = predicate(
            initial, final, target, peak_height=peak, carried_to_target=carried,
            upright_cosine=env.object_upright_cosine(object_name),
            gripper_opening=float(env.state_vector()[13]),
        )
        return ReplayResult(
            episode_index, seed, object_name, len(object_errors), max(object_errors),
            float(np.mean(object_errors)), float(np.mean(joint_errors)), success,
        )
    finally:
        env.close()


def replay_dataset(
    root: str | Path, max_episodes: int | None = None, task: str = "mug_pick_place",
) -> list[ReplayResult]:
    if task not in SCENES:
        raise ValueError(f"unsupported replay task: {task}")
    scene = Path(__file__).parents[2] / "assets/robots/aloha" / SCENES[task]
    assert_scene_compatible(root, scene)
    columns = [
        "episode_index", "privileged.scene_seed", "observation.state", "action",
        "privileged.object_positions",
    ]
    if task == "drawer_open":
        columns.extend(["privileged.drawer_opening", "privileged.phase"])
    table = ds.dataset(Path(root) / "data", format="parquet").to_table(columns=columns)
    data = table.to_pydict()
    episodes: dict[int, dict[str, list]] = {}
    for index, episode in enumerate(data.pop("episode_index")):
        rows = episodes.setdefault(int(episode), {key: [] for key in data})
        for key, column in data.items():
            rows[key].append(column[index])
    selected = sorted(episodes)
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return [replay_episode(episodes[index], index, task) for index in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--skill", choices=tuple(SCENES), default="mug_pick_place")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--max-target-error", "--max-object-error", "--max-mug-error",
        dest="max_target_error", type=float, default=0.02,
    )
    args = parser.parse_args()
    results = replay_dataset(args.root, args.max_episodes, args.skill)
    for result in results:
        print(result)
    if not results or any(
        not result.final_success or result.max_target_error > args.max_target_error
        for result in results
    ):
        raise SystemExit(1)
    print(f"replay passed: {len(results)}/{len(results)} episodes")


if __name__ == "__main__":
    main()
