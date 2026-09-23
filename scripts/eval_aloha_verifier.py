"""Evaluate a saved ALOHA RGB verifier over every held-out video frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bimanual.perception.aloha_verifier import GOAL_BY_SKILL, binary_agreement
from bimanual.perception.verifier import PredicateVerifier
from bimanual.training.train_aloha_verifier import LabeledFrames, SCENE_BY_SKILL, recorded_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--skill", choices=tuple(GOAL_BY_SKILL), required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shuffle-images", action="store_true",
                        help="negative control: shuffle images while keeping oracle labels fixed")
    args = parser.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    predicate = GOAL_BY_SKILL[args.skill]
    if bundle["predicate"] != predicate or bundle["scene"] != SCENE_BY_SKILL[args.skill]:
        raise ValueError("checkpoint predicate or scene does not match requested evaluation")
    model = PredicateVerifier((predicate,)).to(args.device).eval()
    model.load_state_dict(bundle["model"])
    bucket = args.skill.split("_")[0]
    dataset = LeRobotDataset(f"local/aloha-dinner-{bucket}", root=args.val_root, video_backend="pyav")
    labels = recorded_labels(args.val_root, args.skill)
    indices = np.arange(len(labels))
    frame_indices = np.random.default_rng(0).permutation(indices) if args.shuffle_images else indices
    loader = DataLoader(LabeledFrames(dataset, labels, indices, frame_indices), batch_size=16)
    predicted = []
    with torch.no_grad():
        for global_img, left_img, right_img, _ in loader:
            logits = model(global_img.to(args.device), left_img.to(args.device), right_img.to(args.device))
            predicted.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().reshape(-1))
    metrics = binary_agreement(np.asarray(predicted), labels)
    metrics.update({"frames": len(labels), "episodes": int(dataset.meta.total_episodes),
                    "skill": args.skill, "predicate": predicate,
                    "shuffle_images": args.shuffle_images})
    print(json.dumps(metrics, sort_keys=True))
    if not args.shuffle_images and min(metrics["positive_recall"], metrics["negative_recall"]) < 0.95:
        raise SystemExit("verifier class recall is below 95%")


if __name__ == "__main__":
    main()
