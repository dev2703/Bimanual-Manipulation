"""Train and evaluate one RGB predicate on separate ALOHA scene splits.

Example:
  HF_HOME=outputs/.cache/hf .venv/bin/python -m bimanual.training.train_aloha_verifier \
    --skill mug_pick_place --train-root outputs/aloha_mug_train \
    --val-root outputs/aloha_mug_val --output outputs/aloha_mug_verifier.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from bimanual.perception.aloha_verifier import (
    GOAL_BY_SKILL, balanced_indices, binary_agreement, labels_from_recorded_state,
)
from bimanual.perception.verifier import PredicateVerifier

SCENE_BY_SKILL = {
    "mug_pick_place": "task_table_setting.xml",
    "plate_pick_place": "task_table_setting_plate_v2.xml",
    "drawer_open": "task_table_setting_drawer_v2.xml",
}
CAMERAS = ("global", "left_wrist", "right_wrist")


def _target_from_scene(skill: str) -> np.ndarray | None:
    if skill == "drawer_open":
        return None
    scene = Path(__file__).parents[2] / "assets/robots/aloha" / SCENE_BY_SKILL[skill]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = "mug_region" if skill == "mug_pick_place" else "plate_region"
    return data.site_xpos[model.site(site).id].copy()


def recorded_labels(root: Path, skill: str) -> np.ndarray:
    table = ds.dataset(root / "data", format="parquet").to_table(
        columns=["privileged.object_positions", "privileged.drawer_opening"],
    )
    positions = np.asarray(table["privileged.object_positions"].to_pylist(), dtype=np.float64)
    opening = np.asarray(table["privileged.drawer_opening"].to_pylist(), dtype=np.float64)
    return labels_from_recorded_state(skill, positions, opening, _target_from_scene(skill))


class LabeledFrames(Dataset):
    def __init__(self, dataset, labels: np.ndarray, indices: np.ndarray,
                 frame_indices: np.ndarray | None = None) -> None:
        if len(dataset) != len(labels):
            raise ValueError("video dataset and privileged labels have different lengths")
        self.dataset, self.labels, self.indices = dataset, labels, indices
        self.frame_indices = indices if frame_indices is None else np.asarray(frame_indices)
        if self.frame_indices.shape != self.indices.shape:
            raise ValueError("frame and label index arrays must have the same shape")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        row = self.dataset[int(self.frame_indices[position])]
        views = [torch.nn.functional.interpolate(
            row[f"observation.images.{camera}"].unsqueeze(0), size=(96, 96), mode="bilinear",
            align_corners=False,
        ).squeeze(0) for camera in CAMERAS]
        return (*views, torch.tensor([float(self.labels[index])], dtype=torch.float32))


class CounterfactualPairs(Dataset):
    """Matched scene/robot images differing only in mug position."""

    def __init__(self, path: Path) -> None:
        with np.load(path) as data:
            self.images = data["images"]
            self.labels = data["labels"]
            self.seeds = data["seeds"]
            self.skill = str(data["skill"]) if "skill" in data else "mug_pick_place"
        if self.images.shape[1:] != (3, 256, 256, 3) or len(self.images) != len(self.labels):
            raise ValueError("counterfactual archive has an unexpected image or label shape")
        if len(self.images) % 2 or not np.all(self.labels[0::2] == 1) or not np.all(self.labels[1::2] == 0):
            raise ValueError("counterfactual frames must be positive/negative pairs")
        if not np.array_equal(self.seeds[0::2], self.seeds[1::2]):
            raise ValueError("each counterfactual pair must share a scene seed")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        views = [torch.nn.functional.interpolate(
            torch.from_numpy(self.images[index, camera]).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            size=(96, 96), mode="bilinear", align_corners=False,
        ).squeeze(0) for camera in range(3)]
        return (*views, torch.tensor([float(self.labels[index])], dtype=torch.float32))


def train_and_evaluate(
    skill: str, train_root: Path, val_root: Path, output: Path,
    *, epochs: int = 5, max_per_class: int = 1000, device: str = "cpu",
    train_counterfactual: Path | None = None, val_counterfactual: Path | None = None,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    predicate = GOAL_BY_SKILL[skill]
    bucket = skill.split("_")[0]
    train = LeRobotDataset(f"local/aloha-dinner-{bucket}", root=train_root, video_backend="pyav")
    val = LeRobotDataset(f"local/aloha-dinner-{bucket}", root=val_root, video_backend="pyav")
    train_labels = recorded_labels(train_root, skill)
    val_labels = recorded_labels(val_root, skill)
    train_indices = balanced_indices(train_labels, max_per_class=max_per_class, seed=1)
    val_indices = balanced_indices(val_labels, max_per_class=max_per_class, seed=2)
    training: Dataset = LabeledFrames(train, train_labels, train_indices)
    if (train_counterfactual is None) != (val_counterfactual is None):
        raise ValueError("provide both train and validation counterfactual archives")
    if train_counterfactual is not None:
        pairs = CounterfactualPairs(train_counterfactual)
        if pairs.skill != skill:
            raise ValueError("training counterfactual skill does not match dataset")
        training = ConcatDataset([training, pairs, pairs, pairs, pairs])
    loader = DataLoader(training, batch_size=16, shuffle=True)
    validation = DataLoader(LabeledFrames(val, val_labels, val_indices), batch_size=16)
    model = PredicateVerifier((predicate,)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(epochs):
        model.train()
        for global_img, left_img, right_img, label in loader:
            logits = model(global_img.to(device), left_img.to(device), right_img.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for global_img, left_img, right_img, label in validation:
            logits = model(global_img.to(device), left_img.to(device), right_img.to(device))
            predictions.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().reshape(-1))
            labels.extend(label.numpy().reshape(-1))
    metrics = binary_agreement(np.asarray(predictions), np.asarray(labels))
    if val_counterfactual is not None:
        pair_predictions, pair_labels = [], []
        val_pairs = CounterfactualPairs(val_counterfactual)
        if val_pairs.skill != skill:
            raise ValueError("validation counterfactual skill does not match dataset")
        pair_loader = DataLoader(val_pairs, batch_size=16)
        with torch.no_grad():
            for global_img, left_img, right_img, label in pair_loader:
                logits = model(global_img.to(device), left_img.to(device), right_img.to(device))
                pair_predictions.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().reshape(-1))
                pair_labels.extend(label.numpy().reshape(-1))
        metrics["counterfactual"] = binary_agreement(
            np.asarray(pair_predictions), np.asarray(pair_labels), min_support=10,
        )
    metrics.update({"skill": skill, "predicate": predicate, "train_frames": len(train_indices),
                    "val_frames": len(val_indices), "epochs": epochs})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "predicate": predicate,
                "scene": SCENE_BY_SKILL[skill], "metrics": metrics}, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=tuple(GOAL_BY_SKILL), required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-counterfactual", type=Path)
    parser.add_argument("--val-counterfactual", type=Path)
    args = parser.parse_args()
    metrics = train_and_evaluate(args.skill, args.train_root, args.val_root, args.output,
                                 epochs=args.epochs, max_per_class=args.max_per_class,
                                 device=args.device,
                                 train_counterfactual=args.train_counterfactual,
                                 val_counterfactual=args.val_counterfactual)
    print(json.dumps(metrics, sort_keys=True))
    if min(metrics["positive_recall"], metrics["negative_recall"]) < 0.95:
        raise SystemExit("verifier class recall is below the 95% phase gate")
    if "counterfactual" in metrics and min(
        metrics["counterfactual"]["positive_recall"],
        metrics["counterfactual"]["negative_recall"],
    ) < 0.95:
        raise SystemExit("counterfactual class recall is below 95%")


if __name__ == "__main__":
    main()
