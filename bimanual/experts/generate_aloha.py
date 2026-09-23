"""Generate synchronized contact-only ALOHA block demonstrations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from bimanual.experts.aloha_block import run_block_grasp_lift
from bimanual.data.audit import audit_dataset
from bimanual.policy.types import EpisodeManifest, file_sha256
from bimanual.sim.aloha_env import ALOHA_BIMANUAL, AlohaPhysicalEnv

IMAGE_MAP = {
    "overhead_cam": "observation.images.global",
    "wrist_cam_left": "observation.images.left_wrist",
    "wrist_cam_right": "observation.images.right_wrist",
}


def make_features() -> dict:
    return {
        **{
            key: {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
            for key in IMAGE_MAP.values()
        },
        "observation.state": {"dtype": "float32", "shape": (14,), "names": list(ALOHA_BIMANUAL.state_names)},
        "observation.velocity": {"dtype": "float32", "shape": (14,), "names": list(ALOHA_BIMANUAL.state_names)},
        "action": {"dtype": "float32", "shape": (14,), "names": list(ALOHA_BIMANUAL.action_names)},
        "privileged.block_position": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "privileged.phase": {"dtype": "string", "shape": (1,), "names": None},
        "privileged.arm": {"dtype": "string", "shape": (1,), "names": None},
        "privileged.sim_time": {"dtype": "float32", "shape": (1,), "names": None},
        "privileged.scene_seed": {"dtype": "int64", "shape": (1,), "names": None},
    }


def write_episode(dataset, record: list[dict], task: str, seed: int) -> None:
    for tick in record:
        observation = tick["observation"]
        frame = {
            destination: tick["frames"][source]
            for source, destination in IMAGE_MAP.items()
        }
        frame.update(
            {
                "observation.state": observation["observation.state"],
                "observation.velocity": observation["observation.velocity"],
                "action": tick["action"],
                "privileged.block_position": tick["oracle"]["task_block_pos"].astype(np.float32),
                "privileged.phase": tick["phase"],
                "privileged.arm": tick["arm"],
                "privileged.sim_time": np.array([tick["timestamp"]], dtype=np.float32),
                "privileged.scene_seed": np.array([seed], dtype=np.int64),
                "task": task,
            }
        )
        dataset.add_frame(frame)
    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--root", default="outputs/aloha_block_train")
    parser.add_argument("--repo-id", default="local/aloha-physical-block")
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
    offset = {"train": 0, "val": 100_000, "test": 200_000}[args.split]
    scene_path = Path(__file__).parents[2] / "assets" / "robots" / "aloha" / "task_block.xml"
    artifact_hashes = {"task_block.xml": f"sha256:{file_sha256(scene_path)}"}
    manifests = []
    for episode in range(args.episodes):
        seed = offset + episode
        env = AlohaPhysicalEnv()
        env.reset(seed=seed, randomize_block=True)
        result = run_block_grasp_lift(env, record_frames=True)
        env.close()
        if not result.success:
            raise RuntimeError(f"physical expert failed at seed {seed}: {result}")
        task = f"Lift the block with the {result.arm} arm and hold it securely."
        write_episode(dataset, result.record, task, seed)
        manifests.append(
            EpisodeManifest(
                environment="aloha_physical_block",
                embodiment=ALOHA_BIMANUAL.name,
                model_revision=ALOHA_BIMANUAL.model_revision,
                seed=seed,
                split=args.split,
                source_expert="aloha_contact_block_v1",
                artifact_hashes=artifact_hashes,
            ).to_dict()
        )
        print(f"episode={episode + 1}/{args.episodes} seed={seed} frames={len(result.record)} arm={result.arm}")
    # LeRobot's streaming writer leaves the parquet footer open until finalize.
    # Auditing before this call can misdiagnose a healthy run as corrupt.
    dataset.finalize()
    (root / "episode_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")
    report = audit_dataset(root)
    print(f"audit={json.dumps(report.__dict__, sort_keys=True)}")


if __name__ == "__main__":
    main()
