"""Trains perception/verifier.py's PredicateVerifier on a LeRobotDataset
written by experts/generate.py (Phase 5, decisions A2/A3).

Labels are computed on the fly from the dataset's privileged.* fields via
bimanual.sim.oracle_predicates -- the dataset stores object poses/drawer
opening, not predicate booleans directly, so this script is the one place
simulator ground truth is allowed to touch verifier supervision (never at
inference; see evaluation/executor.py's oracle_verify_fn docstring for the
inference-time distinction).

CAUTION (read before trusting a reported accuracy number): a dataset
generated from a single scripted skill (e.g. --skill plate only) makes
every predicate EXCEPT that skill's own object-in-region label constant
across the whole dataset (drawer never touched, other four objects never
moved). A verifier can hit near-100% "agreement" on those by predicting a
constant, which is not verifier skill -- it's a degenerate label
distribution. Only trust per-predicate agreement numbers from a dataset
that actually exercises that predicate's both classes (see
`label_balance` printed at the end of a run).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from bimanual.logging_utils import get_logger
from bimanual.perception.verifier import PREDICATE_NAMES, PredicateVerifier
from bimanual.sim.oracle_predicates import CLOSED_DRAWER_HANDLE_Y, DRAWER_OPEN_THRESHOLD
from bimanual.sim.scene_builder import DRAWER_OPEN_DIST, OBJECT_SPEC, REGIONS, TABLE_HEIGHT, resting_center_z

log = get_logger(__name__)

OBJECT_ORDER = ("plate", "mug", "bottle", "fork", "spoon")
_REGION_NAME_BY_OBJ = {
    "plate": "plate_region",
    "mug": "mug_region",
    "bottle": "bottle_region",
    "fork": "left_cutlery_region",
    "spoon": "right_cutlery_region",
}


def compute_labels(batch: dict) -> torch.Tensor:
    """batch["privileged.object_poses"]: (B, 5, 3); batch["privileged.
    drawer_opening"]: (B, 1). Returns (B, len(PREDICATE_NAMES)) float
    labels in {0, 1}, in the same order as PREDICATE_NAMES."""
    object_poses = batch["privileged.object_poses"].numpy()  # (B, 5, 3)
    drawer_opening = batch["privileged.drawer_opening"].numpy().reshape(-1)  # (B,)

    B = object_poses.shape[0]
    labels = np.zeros((B, len(PREDICATE_NAMES)), dtype=np.float32)

    # drawer_open: qpos-only fallback form of oracle_predicates.drawer_open
    # (dataset doesn't store drawer_handle_pos, only the joint value).
    if "privileged.drawer_handle_y" in batch:
        handle_y = batch["privileged.drawer_handle_y"].numpy().reshape(-1)
        drawer_open_label = (CLOSED_DRAWER_HANDLE_Y - handle_y) >= DRAWER_OPEN_THRESHOLD * DRAWER_OPEN_DIST
    else:
        # Compatibility path for legacy datasets without absolute handle pose.
        drawer_open_label = (-drawer_opening) >= DRAWER_OPEN_THRESHOLD * DRAWER_OPEN_DIST
    labels[:, PREDICATE_NAMES.index("drawer_open")] = drawer_open_label.astype(np.float32)

    for i, obj in enumerate(OBJECT_ORDER):
        rx, ry, radius = REGIONS[_REGION_NAME_BY_OBJ[obj]]
        pos = object_poses[:, i, :]
        xy_dist = np.hypot(pos[:, 0] - rx, pos[:, 1] - ry)
        if obj in ("fork", "spoon"):
            expected_z = TABLE_HEIGHT + OBJECT_SPEC[obj]["half_height"]
        else:
            expected_z = resting_center_z(obj)
        on_table = np.abs(pos[:, 2] - expected_z) < 0.03
        in_region_label = (xy_dist <= radius) & on_table
        labels[:, PREDICATE_NAMES.index(f"{obj}_in_region")] = in_region_label.astype(np.float32)

    return torch.from_numpy(labels)


def train(
    dataset,
    validation_dataset=None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[PredicateVerifier, dict[str, float]]:
    model = PredicateVerifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            global_img = batch["observation.images.global"].to(device)
            left_img = batch["observation.images.left_wrist"].to(device)
            right_img = batch["observation.images.right_wrist"].to(device)
            labels = compute_labels(batch).to(device)

            logits = model(global_img, left_img, right_img)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        log.info("epoch %d: mean loss %.4f", epoch, total_loss / max(n_batches, 1))

    agreement = evaluate_agreement(model, validation_dataset or dataset, device=device)
    return model, agreement


@torch.no_grad()
def evaluate_agreement(model: PredicateVerifier, dataset, device: str = "cpu", batch_size: int = 16) -> dict[str, float]:
    """Per-predicate verifier-vs-oracle agreement (A3's reported metric),
    plus the label balance so a 99% number on a constant label isn't
    mistaken for verifier skill."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = torch.zeros(len(PREDICATE_NAMES))
    positive = torch.zeros(len(PREDICATE_NAMES))
    total = 0

    for batch in loader:
        global_img = batch["observation.images.global"].to(device)
        left_img = batch["observation.images.left_wrist"].to(device)
        right_img = batch["observation.images.right_wrist"].to(device)
        labels = compute_labels(batch).to(device)

        logits = model(global_img, left_img, right_img)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum(dim=0).cpu()
        positive += labels.sum(dim=0).cpu()
        total += labels.shape[0]

    agreement = {name: float(correct[i] / total) for i, name in enumerate(PREDICATE_NAMES)}
    label_balance = {name: float(positive[i] / total) for i, name in enumerate(PREDICATE_NAMES)}
    log.info("verifier-vs-oracle agreement: %s", agreement)
    log.info("label_balance (fraction positive): %s", label_balance)
    return agreement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/dataset_pick_place")
    parser.add_argument("--repo-id", default="local/bimanual-pick-place")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-root", help="separate held-out dataset root")
    parser.add_argument("--val-repo-id", default="local/bimanual-pick-place-val")
    parser.add_argument("--allow-train-eval", action="store_true")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    validation_dataset = None
    if args.val_root:
        validation_dataset = LeRobotDataset(repo_id=args.val_repo_id, root=args.val_root)
    elif not args.allow_train_eval:
        raise RuntimeError("provide --val-root for held-out evaluation, or --allow-train-eval for a smoke test")
    train(dataset, validation_dataset=validation_dataset, epochs=args.epochs)


if __name__ == "__main__":
    main()
