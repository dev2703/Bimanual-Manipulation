"""RGB goal verification for the isolated physical ALOHA skill scenes.

The geometric functions here create *training labels* from recorded simulator
state. At inference the adapter accepts only RGB frames and a trained CNN.
Its predicate is visual goal occupancy, while the stricter physical success
predicates also require contact, carry, release, and retention history.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from bimanual.perception.verifier import PredicateVerifier, preprocess_frame

OBJECT_ORDER = ("plate", "mug", "bottle", "fork", "spoon")
GOAL_BY_SKILL = {
    "mug_pick_place": "mug_in_region",
    "plate_pick_place": "plate_in_region",
    "drawer_open": "drawer_open",
}


def labels_from_recorded_state(
    skill: str,
    object_positions: np.ndarray,
    drawer_opening: np.ndarray,
    target: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-frame binary labels from the recorded privileged columns."""
    if skill not in GOAL_BY_SKILL:
        raise ValueError(f"unknown skill {skill!r}")
    positions = np.asarray(object_positions, dtype=np.float64)
    openings = np.asarray(drawer_opening, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 15 or len(positions) != len(openings):
        raise ValueError("expected matching (N,15) object positions and N drawer openings")
    if skill == "drawer_open":
        return (openings > 0.11).astype(np.int64)
    if target is None or np.asarray(target).shape != (3,):
        raise ValueError("object placement labels require a 3-D target")
    obj = "mug" if skill == "mug_pick_place" else "plate"
    index = OBJECT_ORDER.index(obj)
    pos = positions[:, index * 3 : index * 3 + 3]
    radius = 0.045 if obj == "mug" else 0.04
    height = 0.029 if obj == "mug" else 0.011
    tolerance = 0.012 if obj == "mug" else 0.01
    return ((np.linalg.norm(pos[:, :2] - target[:2], axis=1) < radius)
            & (np.abs(pos[:, 2] - height) < tolerance)).astype(np.int64)


def balanced_indices(labels: np.ndarray, *, max_per_class: int, seed: int = 0) -> np.ndarray:
    """Draw equal positive/negative frame counts; reject degenerate data."""
    labels = np.asarray(labels)
    if max_per_class < 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("binary labels and a positive class cap are required")
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive):
        raise ValueError(f"both classes required; negatives={len(negative)} positives={len(positive)}")
    count = min(max_per_class, len(negative), len(positive))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate((rng.choice(negative, count, replace=False),
                                   rng.choice(positive, count, replace=False))))


def binary_agreement(predicted: np.ndarray, labels: np.ndarray, *, min_support: int = 20) -> dict[str, float | int]:
    """Report both class recalls, preventing majority-class gate inflation."""
    predicted = np.asarray(predicted).astype(bool)
    labels = np.asarray(labels).astype(bool)
    if predicted.shape != labels.shape:
        raise ValueError("prediction and label shapes differ")
    positive, negative = int(labels.sum()), int((~labels).sum())
    if min(positive, negative) < min_support:
        raise ValueError(f"held-out class support too small: positive={positive}, negative={negative}")
    return {
        "agreement": float(np.mean(predicted == labels)),
        "positive_recall": float(np.mean(predicted[labels])),
        "negative_recall": float(np.mean(~predicted[~labels])),
        "positive_support": positive,
        "negative_support": negative,
    }


class AlohaRGBVerifier:
    """Inference-only wrapper: three policy cameras in, one predicate out."""

    def __init__(self, model: PredicateVerifier, predicate: str, device: str = "cpu") -> None:
        if model.predicate_names != (predicate,):
            raise ValueError("checkpoint predicate does not match verifier contract")
        self.model = model.to(device).eval()
        self.predicate = predicate
        self.device = device

    @torch.no_grad()
    def __call__(self, frames: Mapping[str, np.ndarray]) -> dict[str, bool]:
        views = [torch.nn.functional.interpolate(
                     preprocess_frame(frames[name]).unsqueeze(0), size=(96, 96),
                     mode="bilinear", align_corners=False,
                 ).squeeze(0).to(self.device)
                 for name in ("overhead_cam", "wrist_cam_left", "wrist_cam_right")]
        return self.model.predict(*views)
