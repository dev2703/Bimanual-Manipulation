"""Offline inference on LeRobot's public ALOHA transfer-cube dataset.

The published ACT checkpoint predates LeRobot's processor JSON files. Its
normalization buffers are embedded in model.safetensors, so this script reads
those buffers explicitly rather than using dinner-task statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors import safe_open


def legacy_stats(checkpoint: Path, device: str) -> dict[str, torch.Tensor]:
    names = {
        "image_mean": "normalize_inputs.buffer_observation_images_top.mean",
        "image_std": "normalize_inputs.buffer_observation_images_top.std",
        "state_mean": "normalize_inputs.buffer_observation_state.mean",
        "state_std": "normalize_inputs.buffer_observation_state.std",
        "action_mean": "unnormalize_outputs.buffer_action.mean",
        "action_std": "unnormalize_outputs.buffer_action.std",
    }
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        return {name: weights.get_tensor(key).to(device) for name, key in names.items()}


def run(dataset_root: Path, checkpoint: Path, frames: list[int], device: str) -> list[dict]:
    dataset = LeRobotDataset(
        "lerobot/aloha_sim_transfer_cube_human", root=dataset_root,
        episodes=[0], video_backend="pyav",
    )
    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    stats = legacy_stats(checkpoint, device)
    results = []
    for frame_index in frames:
        example = dataset[frame_index]
        image = example["observation.images.top"].to(device).unsqueeze(0)
        state = example["observation.state"].to(device).unsqueeze(0)
        observation = {
            "observation.images.top": (image - stats["image_mean"]) / stats["image_std"],
            "observation.state": (state - stats["state_mean"]) / stats["state_std"],
        }
        policy.reset()  # independent observation-to-first-action checks
        with torch.no_grad():
            normalized_action = policy.select_action(observation)
        predicted = normalized_action * stats["action_std"] + stats["action_mean"]
        policy.reset()
        with torch.no_grad():
            blank_action = policy.select_action({
                **observation,
                "observation.images.top": (torch.zeros_like(image) - stats["image_mean"]) / stats["image_std"],
            })
        blank_predicted = blank_action * stats["action_std"] + stats["action_mean"]
        target = example["action"].to(device).unsqueeze(0)
        if not torch.isfinite(predicted).all():
            raise ValueError(f"non-finite public ACT action at frame {frame_index}")
        row = {
            "frame": frame_index,
            "timestamp_s": float(example["timestamp"]),
            "action_mae": float((predicted - target).abs().mean()),
            "identity_action_mae": float((state - target).abs().mean()),
            "blank_image_action_delta": float((predicted - blank_predicted).abs().mean()),
            "predicted_action": np.round(predicted[0].cpu().numpy(), 4).tolist(),
            "dataset_action": np.round(target[0].cpu().numpy(), 4).tolist(),
        }
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 50, 100, 150, 200])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    results = run(args.dataset_root, args.checkpoint, args.frames, args.device)
    for result in results:
        print(json.dumps(result, sort_keys=True))
    print(json.dumps({"frames": len(results), "mean_action_mae": float(np.mean([r["action_mae"] for r in results]))}))


if __name__ == "__main__":
    main()
