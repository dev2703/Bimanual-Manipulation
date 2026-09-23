"""Replay recorded 10 Hz dinner actions through the 30 Hz ALOHA controller."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from bimanual.evaluation.aloha_predicates import mug_placed
from bimanual.sim.aloha_env import TABLE_SETTING_OBJECTS, AlohaTableSettingEnv


@dataclass(frozen=True)
class ReplayResult:
    episode_index: int
    seed: int
    frames: int
    max_mug_position_error: float
    mean_mug_position_error: float
    mean_joint_max_error: float
    final_success: bool


def replay_episode(rows: dict[str, list], episode_index: int) -> ReplayResult:
    seeds = set(int(seed) for seed in rows["privileged.scene_seed"])
    if len(seeds) != 1:
        raise ValueError(f"episode {episode_index} has inconsistent scene seeds: {seeds}")
    seed = seeds.pop()
    mug_offset = 3 * list(TABLE_SETTING_OBJECTS).index("mug")
    env = AlohaTableSettingEnv()
    try:
        env.reset(seed=seed, randomize_objects=True)
        initial = env.oracle_state()["mug_pos"].copy()
        target = env.data.site_xpos[env.model.site("mug_region").id].copy()
        peak = float(initial[2])
        carried = False
        mug_errors = []
        joint_errors = []
        for action, state, objects in zip(
            rows["action"], rows["observation.state"], rows["privileged.object_positions"],
            strict=True,
        ):
            position = env.oracle_state()["mug_pos"]
            mug_errors.append(float(np.linalg.norm(position - np.asarray(objects[mug_offset:mug_offset + 3]))))
            joint_errors.append(float(np.max(np.abs(env.state_vector() - np.asarray(state)))))
            peak = max(peak, float(position[2]))
            carried |= bool(
                position[2] > initial[2] + 0.08
                and np.linalg.norm(position[:2] - target[:2]) < 0.05
            )
            for _ in range(3):
                env.step(np.asarray(action, dtype=np.float64))
        final = env.oracle_state()["mug_pos"]
        success = mug_placed(
            initial, final, target, peak_height=peak, carried_to_target=carried,
            upright_cosine=env.mug_upright_cosine(),
            gripper_opening=float(env.state_vector()[13]),
        )
        return ReplayResult(
            episode_index, seed, len(mug_errors), max(mug_errors),
            float(np.mean(mug_errors)), float(np.mean(joint_errors)), success,
        )
    finally:
        env.close()


def replay_dataset(root: str | Path, max_episodes: int | None = None) -> list[ReplayResult]:
    table = ds.dataset(Path(root) / "data", format="parquet").to_table(columns=[
        "episode_index", "privileged.scene_seed", "observation.state", "action",
        "privileged.object_positions",
    ])
    data = table.to_pydict()
    episodes: dict[int, dict[str, list]] = {}
    for index, episode in enumerate(data.pop("episode_index")):
        rows = episodes.setdefault(int(episode), {key: [] for key in data})
        for key, column in data.items():
            rows[key].append(column[index])
    selected = sorted(episodes)
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return [replay_episode(episodes[index], index) for index in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-mug-error", type=float, default=0.02)
    args = parser.parse_args()
    results = replay_dataset(args.root, args.max_episodes)
    for result in results:
        print(result)
    if not results or any(
        not result.final_success or result.max_mug_position_error > args.max_mug_error
        for result in results
    ):
        raise SystemExit(1)
    print(f"replay passed: {len(results)}/{len(results)} episodes")


if __name__ == "__main__":
    main()
