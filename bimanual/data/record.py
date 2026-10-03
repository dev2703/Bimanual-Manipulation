"""Record LeRobot datasets from any registered skill's scripted expert.

    python -m bimanual.data.record --skill plate_pick_place --split train --episodes 50
    python -m bimanual.data.record --skill plate_pick_place --split train --episodes 50 --resume

Each frame stores the 14-D `observation.state` and the 37-D
`observation.cooperative_state`, so one dataset serves both sides of the
state ablation. Train, validation and test seeds are disjoint by offset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from bimanual.control.aloha_context import COOPERATIVE_STATE_NAMES
from bimanual.data.audit import audit_dataset
from bimanual.data.scene_artifacts import scene_artifact_hashes
from bimanual.evaluation.skill_gate import reset_for_skill
from bimanual.policy.types import EpisodeManifest
from bimanual.sim.aloha_env import ALOHA_BIMANUAL
from bimanual.skills.registry import SKILLS, Skill, recovery_shift

FPS = 10
SPLIT_OFFSETS = {"train": 0, "val": 100_000, "test": 200_000}
IMAGE_MAP = {
    "overhead_cam": "observation.images.global",
    "wrist_cam_left": "observation.images.left_wrist",
    "wrist_cam_right": "observation.images.right_wrist",
}
RECOVERY_FEATURES = {
    "privileged.failure_event": {"dtype": "float32", "shape": (1,), "names": None},
    "privileged.recovery_active": {"dtype": "float32", "shape": (1,), "names": None},
    "privileged.recovery_outcome": {"dtype": "float32", "shape": (1,), "names": None},
    "privileged.perturbation_xy": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
}


def scene_objects(oracle: dict) -> tuple[str, ...]:
    """Free objects in recording order, taken from the scene's oracle state."""
    return tuple(key[:-len("_pos")] for key in oracle if key.endswith("_pos"))


def objects_from_features(features: dict) -> tuple[str, ...]:
    names = features["privileged.object_positions"]["names"]
    return tuple(dict.fromkeys(name.rsplit(".", 1)[0] for name in names))


def make_features(objects: tuple[str, ...], drawer: bool = True, recovery: bool = False) -> dict:
    state_names = list(ALOHA_BIMANUAL.state_names)
    features = {
        **{key: {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
           for key in IMAGE_MAP.values()},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": state_names},
        "observation.cooperative_state": {"dtype": "float32", "shape": (len(COOPERATIVE_STATE_NAMES),),
                                          "names": list(COOPERATIVE_STATE_NAMES)},
        "observation.velocity": {"dtype": "float32", "shape": (14,), "names": state_names},
        "action": {"dtype": "float32", "shape": (14,), "names": list(ALOHA_BIMANUAL.action_names)},
        "privileged.object_positions": {"dtype": "float32", "shape": (3 * len(objects),),
                                        "names": [f"{name}.{axis}" for name in objects for axis in "xyz"]},
        "privileged.phase": {"dtype": "string", "shape": (1,), "names": None},
        "privileged.arm": {"dtype": "string", "shape": (1,), "names": None},
        "privileged.sim_time": {"dtype": "float32", "shape": (1,), "names": None},
        "privileged.scene_seed": {"dtype": "int64", "shape": (1,), "names": None},
    }
    if drawer:
        features["privileged.drawer_opening"] = {"dtype": "float32", "shape": (1,), "names": None}
    if recovery:
        features.update(RECOVERY_FEATURES)
    return features


def write_episode(dataset, record: list[dict], task: str, seed: int, objects: tuple[str, ...],
                  perturbation_xy: np.ndarray | None = None) -> None:
    """Append one expert episode. A perturbation marks a recovery episode."""
    recovery = perturbation_xy is not None
    if recovery and np.asarray(perturbation_xy).shape != (2,):
        raise ValueError("recovery episodes require the exact exogenous XY shift")
    event_recorded = False
    for tick in record:
        observation, oracle = tick["observation"], tick["oracle"]
        frame = {destination: tick["frames"][source] for source, destination in IMAGE_MAP.items()}
        frame.update({
            "observation.state": observation["observation.state"],
            "observation.cooperative_state": np.asarray(tick["cooperative_state"], dtype=np.float32),
            "observation.velocity": observation["observation.velocity"],
            "action": tick["action"],
            "privileged.object_positions": np.concatenate(
                [oracle[f"{name}_pos"] for name in objects]).astype(np.float32),
            "privileged.phase": tick["phase"],
            "privileged.arm": tick["arm"],
            "privileged.sim_time": np.array([tick["timestamp"]], dtype=np.float32),
            "privileged.scene_seed": np.array([seed], dtype=np.int64),
            "task": task,
        })
        if "drawer_opening" in oracle:
            frame["privileged.drawer_opening"] = oracle["drawer_opening"].astype(np.float32)
        if recovery:
            active = tick["phase"] != "APPROACH"
            event = active and not event_recorded
            event_recorded |= event
            frame.update({
                "privileged.failure_event": np.array([float(event)], dtype=np.float32),
                "privileged.recovery_active": np.array([float(active)], dtype=np.float32),
                "privileged.recovery_outcome": np.array([1.0], dtype=np.float32),
                "privileged.perturbation_xy": np.asarray(perturbation_xy, dtype=np.float32),
            })
        dataset.add_frame(frame)
    dataset.save_episode()


def _manifest(skill: Skill, seed: int, split: str, hashes: dict[str, str]) -> dict:
    return EpisodeManifest(
        environment="aloha_table_setting" if skill.env_kind == "table" else "aloha_physical",
        embodiment=ALOHA_BIMANUAL.name,
        model_revision=ALOHA_BIMANUAL.model_revision,
        seed=seed,
        split=split,
        source_expert=skill.source_expert,
        artifact_hashes=hashes,
    ).to_dict()


def _recover_manifests(root: Path, completed: int, offset: int, skill: Skill,
                       split: str, hashes: dict[str, str]) -> list[dict]:
    """Verify saved episodes before appending to an interrupted collection."""
    if completed == 0:
        path = root / "episode_manifests.json"
        if path.exists() and json.loads(path.read_text()):
            raise ValueError("episode manifests exist but dataset metadata has no episodes")
        return []
    try:
        table = ds.dataset(root / "data", format="parquet").to_table(
            columns=["episode_index", "privileged.scene_seed"])
    except pa.ArrowInvalid as exc:
        raise ValueError("dataset Parquet is incomplete; resume requires a finalized episode checkpoint") from exc
    episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    seeds = np.asarray(table["privileged.scene_seed"].to_pylist(), dtype=np.int64).reshape(-1)
    if set(episodes.tolist()) != set(range(completed)):
        raise ValueError("saved episode indices do not match dataset metadata")
    for index in range(completed):
        if set(seeds[episodes == index].tolist()) != {offset + index}:
            raise ValueError(f"saved episode {index} has an unexpected scene seed")
    expected = [_manifest(skill, offset + index, split, hashes) for index in range(completed)]
    path = root / "episode_manifests.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if len(recorded) > completed or recorded != expected[:len(recorded)]:
            raise ValueError("existing episode manifests disagree with this skill, split, or scene")
    return expected


def _save_manifests(root: Path, manifests: list[dict]) -> None:
    temporary = root / "episode_manifests.json.tmp"
    temporary.write_text(json.dumps(manifests, indent=2) + "\n")
    temporary.replace(root / "episode_manifests.json")


def _compatible_features(actual: dict, expected: dict) -> bool:
    """LeRobot adds index/timestamp features to the declared robot schema."""
    return all(
        name in actual
        and actual[name].get("dtype") == feature["dtype"]
        and list(actual[name].get("shape", ())) == list(feature["shape"])
        and actual[name].get("names") == feature["names"]
        for name, feature in expected.items()
    )


def _scene_schema(skill: Skill) -> tuple[tuple[str, ...], bool]:
    env = skill.make_env()
    try:
        oracle = env.oracle_state()
        return scene_objects(oracle), "drawer_opening" in oracle
    finally:
        env.close()


def record_dataset(skill: Skill, split: str, episodes: int, root: Path, repo_id: str,
                   resume: bool = False, overwrite: bool = False, checkpoint_every: int = 1) -> dict:
    """Record `episodes` expert demonstrations and return the audit report."""
    if resume and not root.exists():
        raise FileNotFoundError(f"cannot resume missing dataset root: {root}")
    if root.exists() and not resume:
        if not overwrite:
            raise FileExistsError(f"dataset root already exists: {root}; pass --overwrite to replace it")
        shutil.rmtree(root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    hashes = scene_artifact_hashes(skill.scene_path())
    offset = SPLIT_OFFSETS[split]
    objects, drawer = _scene_schema(skill)
    features = make_features(objects, drawer, skill.recovery)
    if resume:
        start = int(json.loads((root / "meta/info.json").read_text())["total_episodes"])
        if start > episodes:
            raise ValueError(f"dataset already has {start} episodes, more than requested {episodes}")
        manifests = _recover_manifests(root, start, offset, skill, split, hashes)
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=root)
        if dataset.meta.fps != FPS or not _compatible_features(dataset.meta.features, features):
            raise ValueError("existing dataset timing or feature schema does not match")
    else:
        dataset = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features, root=root,
                                        robot_type=ALOHA_BIMANUAL.name, use_videos=True)
        start, manifests = 0, []
    _save_manifests(root, manifests)
    for index in range(start, episodes):
        seed = offset + index
        env = skill.make_env()
        try:
            reset_for_skill(skill, env, seed, jitter=0.015)
            result = skill.run(env, record_frames=True)
        finally:
            env.close()
        if not result.success:
            raise RuntimeError(f"{skill.name} expert failed seed={seed}: {result}")
        write_episode(dataset, result.record, skill.instruction_for(seed), seed, objects,
                      perturbation_xy=recovery_shift(seed) if skill.recovery else None)
        manifests.append(_manifest(skill, seed, split, hashes))
        _save_manifests(root, manifests)
        print(f"episode={index + 1}/{episodes} seed={seed} frames={len(result.record)}", flush=True)
        if (index + 1) % checkpoint_every == 0 and index + 1 < episodes:
            dataset.finalize()
            dataset = LeRobotDataset.resume(repo_id=repo_id, root=root)
    dataset.finalize()
    return audit_dataset(root).__dict__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skill", choices=sorted(SKILLS), default="mug_pick_place")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--split", choices=sorted(SPLIT_OFFSETS), default="train")
    parser.add_argument("--root", type=Path, help="default: outputs/aloha_<bucket>_<split>")
    parser.add_argument("--repo-id", help="default: local/aloha-dinner-<bucket>")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if args.episodes < 1 or args.checkpoint_every < 1:
        parser.error("--episodes and --checkpoint-every must be positive")
    skill = SKILLS[args.skill]
    report = record_dataset(
        skill, args.split, args.episodes,
        args.root or Path(f"outputs/aloha_{skill.bucket}_{args.split}"),
        args.repo_id or f"local/aloha-dinner-{skill.bucket}",
        resume=args.resume, overwrite=args.overwrite, checkpoint_every=args.checkpoint_every,
    )
    print(f"audit={json.dumps(report, sort_keys=True)}")


if __name__ == "__main__":
    main()
