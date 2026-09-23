"""Generate synchronized ALOHA dinner-table demonstrations by skill bucket."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from bimanual.data.audit import audit_dataset
from bimanual.data.scene_artifacts import scene_artifact_hashes
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.generate_aloha import IMAGE_MAP
from bimanual.policy.types import EpisodeManifest
from bimanual.sim.aloha_env import ALOHA_BIMANUAL, TABLE_SETTING_OBJECTS, AlohaTableSettingEnv

OBJECT_NAMES = tuple(TABLE_SETTING_OBJECTS)


def make_features() -> dict:
    names = list(ALOHA_BIMANUAL.state_names)
    return {
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


def write_episode(dataset, record: list[dict], task: str, seed: int) -> None:
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
        dataset.add_frame(frame)
    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=["mug_pick_place"], default="mug_pick_place")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--root", default="outputs/aloha_mug_train")
    parser.add_argument("--repo-id", default="local/aloha-dinner-mug")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"dataset root already exists: {root}; pass --overwrite to replace it")
        shutil.rmtree(root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=10,
        features=make_features(),
        root=root,
        robot_type=ALOHA_BIMANUAL.name,
        use_videos=True,
    )
    scene = Path(__file__).parents[2] / "assets" / "robots" / "aloha" / "task_table_setting.xml"
    artifact_hashes = scene_artifact_hashes(scene)
    offset = {"train": 0, "val": 100_000, "test": 200_000}[args.split]
    manifests = []
    for index in range(args.episodes):
        seed = offset + index
        env = AlohaTableSettingEnv()
        try:
            env.reset(seed=seed, randomize_objects=True)
            result = run_mug_pick_place(env, record_frames=True)
        finally:
            env.close()
        if not result.success:
            raise RuntimeError(f"mug expert failed seed={seed}: {result}")
        task = "Set the dinner table: place the blue mug to the right of the plate."
        write_episode(dataset, result.record, task, seed)
        manifests.append(EpisodeManifest(
            environment="aloha_table_setting",
            embodiment=ALOHA_BIMANUAL.name,
            model_revision=ALOHA_BIMANUAL.model_revision,
            seed=seed,
            split=args.split,
            source_expert="aloha_contact_mug_v1",
            artifact_hashes=artifact_hashes,
        ).to_dict())
        print(f"episode={index+1}/{args.episodes} seed={seed} frames={len(result.record)}")
    dataset.finalize()
    (root / "episode_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")
    print(f"audit={json.dumps(audit_dataset(root).__dict__, sort_keys=True)}")


if __name__ == "__main__":
    main()
