"""RGB predicate verifier (Phase 5, decisions A2/A3, docs/plan_review.md).

A small CNN over the three 256x256 camera views predicts a fixed set of
symbolic predicates, so the executor (evaluation/executor.py) never calls
oracle_state() at inference. Supervised during training on privileged.*
labels computed from the simulator (bimanual/training/train_verifier.py);
this module only defines the architecture and inference-time
preprocessing.

Reported metric (A3): verifier-vs-oracle agreement, per predicate, on
held-out seeds -- NOT training loss. See docs/decisions.md for the actual
measured number once training has been run at scale; the dataset
generated so far (2 episodes, plate skill only) is too small and too
uniform (e.g. the drawer is never touched) to produce a meaningful
agreement estimate, so no number is claimed here.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

OBJECTS = ("plate", "mug", "bottle", "fork", "spoon")
PREDICATE_NAMES: tuple[str, ...] = ("drawer_open",) + tuple(f"{obj}_in_region" for obj in OBJECTS)


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) render -> float32 (3, H, W) tensor in [0, 1],
    matching LeRobotDataset's own image convention (CHW, float)."""
    assert frame.dtype == np.uint8, frame.dtype
    return torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredicateVerifier(nn.Module):
    """Three camera views (each 3xHxW, channel-concatenated) -> one logit
    per predicate in `predicate_names`."""

    def __init__(self, predicate_names: tuple[str, ...] = PREDICATE_NAMES) -> None:
        super().__init__()
        self.predicate_names = predicate_names
        self.backbone = nn.Sequential(
            _ConvBlock(9, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, len(predicate_names))

    def forward(self, global_img: torch.Tensor, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        """Each arg is (B, 3, H, W). Returns (B, n_predicates) logits."""
        x = torch.cat([global_img, left_img, right_img], dim=1)
        feat = self.backbone(x).flatten(1)
        return self.head(feat)

    @torch.no_grad()
    def predict(
        self, global_img: torch.Tensor, left_img: torch.Tensor, right_img: torch.Tensor, threshold: float = 0.5
    ) -> dict[str, bool]:
        """Single-example convenience wrapper: each arg is (3, H, W)."""
        logits = self.forward(global_img.unsqueeze(0), left_img.unsqueeze(0), right_img.unsqueeze(0))
        probs = torch.sigmoid(logits)[0]
        return {name: bool(p > threshold) for name, p in zip(self.predicate_names, probs)}
