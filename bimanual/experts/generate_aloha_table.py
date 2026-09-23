"""Generate synchronized ALOHA dinner-table demonstrations by skill bucket."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from bimanual.data.audit import audit_dataset
from bimanual.data.scene_artifacts import scene_artifact_hashes
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.experts.generate_aloha import IMAGE_MAP
from bimanual.policy.types import EpisodeManifest
from bimanual.sim.aloha_env import ALOHA_BIMANUAL, TABLE_SETTING_OBJECTS, AlohaTableSettingEnv

OBJECT_NAMES = tuple(TABLE_SETTING_OBJECTS)


@dataclass(frozen=True)
class TableSkill:
    bucket: str
    scene_name: str
    instruction: str
    source_expert: str
    run: Callable


SKILLS = {
    "mug_pick_place": TableSkill(
        "mug", "task_table_setting.xml",
        "Set the dinner table: place the blue mug to the right of the plate.",
        "aloha_contact_mug_v1", run_mug_pick_place,
    ),
    "plate_pick_place": TableSkill(
        "plate", "task_table_setting_plate_v2.xml",
        "Set the dinner table: place the serving plate in the centre.",
        "aloha_contact_plate_v2", run_plate_pick_place,
    ),
    "plate_recovery": TableSkill(
        "plate_recovery", "task_table_setting_plate_v2.xml",
        "Set the dinner table: place the serving plate in the centre.",
        "aloha_contact_plate_moved_recovery_v1", run_plate_pick_place,
    ),
    "drawer_open": TableSkill(
        "drawer", "task_table_setting_drawer_v2.xml",
        "Open the top drawer of the dinner cabinet.",
        "aloha_contact_drawer_v2", run_drawer_open,
    ),
}


def _manifest(skill: TableSkill, seed: int, split: str, hashes: dict[str, str]) -> dict:
    return EpisodeManifest(
        environment="aloha_table_setting",
        embodiment=ALOHA_BIMANUAL.name,
        model_revision=ALOHA_BIMANUAL.model_revision,
        seed=seed,
        split=split,
        source_expert=skill.source_expert,
        artifact_hashes=hashes,
    ).to_dict()


def _recover_manifests(
    root: Path, completed: int, offset: int, skill: TableSkill,
    split: str, hashes: dict[str, str],
) -> list[dict]:
    """Verify saved episodes before appending to an interrupted collection."""
    if completed == 0:
        path = root / "episode_manifests.json"
        if path.exists() and json.loads(path.read_text()):
            raise ValueError("episode manifests exist but dataset metadata has no episodes")
        return []
    try:
        table = ds.dataset(root / "data", format="parquet").to_table(
            columns=["episode_index", "privileged.scene_seed"],
        )
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
    destination = root / "episode_manifests.json"
    temporary = root / "episode_manifests.json.tmp"
    temporary.write_text(json.dumps(manifests, indent=2) + "\n")
    temporary.replace(destination)


def _compatible_features(actual: dict, expected: dict) -> bool:
    """LeRobot adds index/timestamp features to the declared robot schema."""
    return all(
        name in actual
        and actual[name].get("dtype") == feature["dtype"]
        and list(actual[name].get("shape", ())) == list(feature["shape"])
        and actual[name].get("names") == feature["names"]
        for name, feature in expected.items()
    )


def make_features(recovery: bool = False) -> dict:
    names = list(ALOHA_BIMANUAL.state_names)
    features = {
        **{
            key: {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
            for key in IMAGE_MAP.values()
        },
        "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
        "observation.velocity": {"dtype": "float32", "shape": (14,), "names": names},
        "action": {"dtype": "float32", "shape": (14,), "names": list(ALOHA_BIMANUAL.action_names)},
        "privileged.object_positions": {
            "dtype": "float32", "shape": (len(OBJECT_NAMES) * 3,),
            "names": [f"{name}.{axis}" for name in OBJECT_NAMES for axis in "xyz"],
        },
        "privileged.drawer_opening": {"dtype": "float32", "shape": (1,), "names": None},
        "privileged.phase": {"dtype": "string", "shape": (1,), "names": None},
        "privileged.arm": {"dtype": "string", "shape": (1,), "names": None},
        "privileged.sim_time": {"dtype": "float32", "shape": (1,), "names": None},
        "privileged.scene_seed": {"dtype": "int64", "shape": (1,), "names": None},
    }
    if recovery:
        features.update({
            "privileged.failure_event": {"dtype": "float32", "shape": (1,), "names": None},
            "privileged.recovery_active": {"dtype": "float32", "shape": (1,), "names": None},
            "privileged.recovery_outcome": {"dtype": "float32", "shape": (1,), "names": None},
            "privileged.perturbation_xy": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        })
    return features


def write_episode(dataset, record: list[dict], task: str, seed: int,
                  recovery: bool = False, perturbation_xy: np.ndarray | None = None) -> None:
    if recovery and (perturbation_xy is None or np.asarray(perturbation_xy).shape != (2,)):
        raise ValueError("recovery episodes require the exact exogenous XY shift")
    event_recorded = False
    for tick in record:
        observation, oracle = tick["observation"], tick["oracle"]
        frame = {destination: tick["frames"][source] for source, destination in IMAGE_MAP.items()}
        frame.update({
            "observation.state": observation["observation.state"],
            "observation.velocity": observation["observation.velocity"],
            "action": tick["action"],
            "privileged.object_positions": np.concatenate(
                [oracle[f"{name}_pos"] for name in OBJECT_NAMES]
            ).astype(np.float32),
            "privileged.drawer_opening": oracle["drawer_opening"].astype(np.float32),
            "privileged.phase": tick["phase"],
            "privileged.arm": tick["arm"],
            "privileged.sim_time": np.array([tick["timestamp"]], dtype=np.float32),
            "privileged.scene_seed": np.array([seed], dtype=np.int64),
            "task": task,
        })
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=tuple(SKILLS), default="mug_pick_place")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--root")
    parser.add_argument("--repo-id")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    skill = SKILLS[args.skill]
    recovery = args.skill == "plate_recovery"
    root = Path(args.root or f"outputs/aloha_{skill.bucket}_{args.split}")
    repo_id = args.repo_id or f"local/aloha-dinner-{skill.bucket}"
    if args.resume and not root.exists():
        raise FileNotFoundError(f"cannot resume missing dataset root: {root}")
    if root.exists() and not args.resume:
        if not args.overwrite:
            raise FileExistsError(f"dataset root already exists: {root}; pass --overwrite to replace it")
        shutil.rmtree(root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    scene = Path(__file__).parents[2] / "assets" / "robots" / "aloha" / skill.scene_name
    artifact_hashes = scene_artifact_hashes(scene)
    offset = {"train": 0, "val": 100_000, "test": 200_000}[args.split]
    if args.resume:
        info = json.loads((root / "meta/info.json").read_text())
        start_index = int(info["total_episodes"])
        if start_index > args.episodes:
            raise ValueError(f"dataset already has {start_index} episodes, more than requested {args.episodes}")
        manifests = _recover_manifests(root, start_index, offset, skill, args.split, artifact_hashes)
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=root)
        if dataset.meta.fps != 10 or not _compatible_features(dataset.meta.features, make_features(recovery)):
            raise ValueError("existing dataset timing or feature schema does not match")
        _save_manifests(root, manifests)
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=10,
            features=make_features(recovery),
            root=root,
            robot_type=ALOHA_BIMANUAL.name,
            use_videos=True,
        )
        start_index = 0
        manifests = []
        _save_manifests(root, manifests)
    for index in range(start_index, args.episodes):
        seed = offset + index
        env = AlohaTableSettingEnv(scene)
        try:
            env.reset(seed=seed, randomize_objects=True)
            if recovery:
                rng = np.random.default_rng(seed)
                angle = rng.uniform(-np.pi, np.pi)
                shift = 0.035 * np.array([np.cos(angle), np.sin(angle)])
                result = run_plate_pick_place(
                    env, record_frames=True, perturbation_xy=shift,
                    replan_after_perturb=True,
                )
            else:
                result = skill.run(env, record_frames=True)
        finally:
            env.close()
        if not result.success:
            raise RuntimeError(f"{skill.bucket} expert failed seed={seed}: {result}")
        write_episode(dataset, result.record, skill.instruction, seed,
                      recovery=recovery, perturbation_xy=shift if recovery else None)
        manifests.append(_manifest(skill, seed, args.split, artifact_hashes))
        _save_manifests(root, manifests)
        print(f"episode={index+1}/{args.episodes} seed={seed} frames={len(result.record)}")
        if (index + 1) % args.checkpoint_every == 0 and index + 1 < args.episodes:
            dataset.finalize()
            dataset = LeRobotDataset.resume(repo_id=repo_id, root=root)
    dataset.finalize()
    print(f"audit={json.dumps(audit_dataset(root).__dict__, sort_keys=True)}")


if __name__ == "__main__":
    main()
