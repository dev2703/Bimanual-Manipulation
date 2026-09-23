"""Fail-fast audits for synchronized robot-learning datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class DatasetAudit:
    episodes: int
    frames: int
    manifest_entries: int
    unique_seeds: int
    min_episode_frames: int
    max_episode_frames: int
    mean_dt: float
    max_dt_error: float
    action_state_mae: float
    exact_action_state_fraction: float


def audit_dataset(root: str | Path, *, expected_fps: int = 10) -> DatasetAudit:
    """Validate metadata, manifests, timing, dimensions, and non-identity actions.

    This intentionally fails before training. It catches the two expensive
    historical failures: action labels copied from state and unsynchronized
    simulator timestamps.
    """
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    manifests = json.loads((root / "episode_manifests.json").read_text())
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"no parquet data under {root}")
    table = pq.read_table(parquet_files)
    columns = table.to_pydict()

    episodes = np.asarray(columns["episode_index"], dtype=np.int64)
    state = np.stack(columns["observation.state"]).astype(np.float64)
    action = np.stack(columns["action"]).astype(np.float64)
    sim_time = np.asarray(columns["privileged.sim_time"], dtype=np.float64).reshape(-1)
    seeds = np.asarray(columns["privileged.scene_seed"], dtype=np.int64).reshape(-1)

    if state.shape != action.shape:
        raise ValueError(f"state/action shape mismatch: {state.shape} versus {action.shape}")
    if len(episodes) != int(info["total_frames"]):
        raise ValueError("metadata frame count does not match parquet")
    episode_ids, lengths = np.unique(episodes, return_counts=True)
    if len(episode_ids) != int(info["total_episodes"]):
        raise ValueError("metadata episode count does not match parquet")
    if len(manifests) != len(episode_ids):
        raise ValueError("one episode manifest is required per episode")
    manifest_seeds = [int(item["seed"]) for item in manifests]
    if len(set(manifest_seeds)) != len(manifest_seeds):
        raise ValueError("episode manifest seeds must be unique")
    if set(manifest_seeds) != set(seeds.tolist()):
        raise ValueError("manifest seeds do not match frame seeds")

    same_episode = episodes[1:] == episodes[:-1]
    dt = np.diff(sim_time)[same_episode]
    target_dt = 1.0 / expected_fps
    max_dt_error = float(np.max(np.abs(dt - target_dt)))
    if max_dt_error > 1e-4:
        raise ValueError(f"simulator timing is not {expected_fps} Hz (max error {max_dt_error:g}s)")

    abs_error = np.abs(action - state)
    action_state_mae = float(abs_error.mean())
    exact_fraction = float(np.mean(action == state))
    if action_state_mae < 1e-4 or exact_fraction > 0.99:
        raise ValueError("actions appear to be copied from observation.state")

    return DatasetAudit(
        episodes=len(episode_ids),
        frames=len(episodes),
        manifest_entries=len(manifests),
        unique_seeds=len(set(seeds.tolist())),
        min_episode_frames=int(lengths.min()),
        max_episode_frames=int(lengths.max()),
        mean_dt=float(dt.mean()),
        max_dt_error=max_dt_error,
        action_state_mae=action_state_mae,
        exact_action_state_fraction=exact_fraction,
    )


def assert_disjoint_splits(*roots: str | Path) -> None:
    seen: set[int] = set()
    for root in roots:
        manifests = json.loads((Path(root) / "episode_manifests.json").read_text())
        seeds = {int(item["seed"]) for item in manifests}
        overlap = seen & seeds
        if overlap:
            raise ValueError(f"scene seed leakage across splits: {sorted(overlap)[:5]}")
        seen.update(seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(asdict(audit_dataset(args.root, expected_fps=args.fps)), indent=2))


if __name__ == "__main__":
    main()

