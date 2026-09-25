"""Merge the mug/plate/drawer/block scripted-expert datasets into one
multi-task LeRobotDataset so lerobot-train and train_compact_vla.py can
train on all four skills from a single --dataset.root.

Only the features common to every source dataset (images, state, velocity,
action, task text) are kept; per-skill privileged.* columns are dropped.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from bimanual.sim.aloha_env import ALOHA_BIMANUAL

CAMERA_KEYS = tuple(
    key for key in (
        "observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist",
    )
)


def _to_hwc_uint8(image) -> np.ndarray:
    array = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    if array.dtype != np.uint8:
        array = (array * 255.0).round().clip(0, 255).astype(np.uint8)
    return array

CAMERA_FEATURES = {
    "observation.images.global": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.left_wrist": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.right_wrist": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
}


def make_features() -> dict:
    names = list(ALOHA_BIMANUAL.state_names)
    return {
        **CAMERA_FEATURES,
        "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
        "observation.velocity": {"dtype": "float32", "shape": (14,), "names": names},
        "action": {"dtype": "float32", "shape": (14,), "names": list(ALOHA_BIMANUAL.action_names)},
    }


def merge(sources: list[Path], root: Path, repo_id: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=10, features=make_features(), root=root,
        robot_type=ALOHA_BIMANUAL.name, use_videos=True,
    )
    for source in sources:
        source_ds = LeRobotDataset(f"local/{source.name}", root=source, video_backend="pyav")
        episode_boundaries = source_ds.meta.episodes
        for episode_index in range(source_ds.meta.total_episodes):
            start = episode_boundaries[episode_index]["dataset_from_index"]
            end = episode_boundaries[episode_index]["dataset_to_index"]
            for frame_index in range(start, end):
                sample = source_ds[frame_index]
                frame = {key: sample[key] for key in make_features() if key not in CAMERA_KEYS}
                for key in CAMERA_KEYS:
                    frame[key] = _to_hwc_uint8(sample[key])
                frame["task"] = sample["task"]
                dataset.add_frame(frame)
            dataset.save_episode()
        print(f"merged {source_ds.meta.total_episodes} episodes from {source}", flush=True)
    dataset.finalize()
    print(f"merged dataset ready at {root}: {dataset.meta.total_episodes} episodes, "
          f"{dataset.meta.total_frames} frames", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--root")
    parser.add_argument("--repo-id", default="local/aloha-multitask")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root or f"outputs/aloha_multitask_{args.split}")
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"dataset root already exists: {root}; pass --overwrite to replace it")
        shutil.rmtree(root)
    sources = [Path(f"outputs/aloha_{bucket}_{args.split}") for bucket in ("mug", "plate", "drawer", "block")]
    missing = [s for s in sources if not s.exists()]
    if missing:
        raise FileNotFoundError(f"missing source datasets: {missing}")
    merge(sources, root, args.repo_id)


if __name__ == "__main__":
    main()
