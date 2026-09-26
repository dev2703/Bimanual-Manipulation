"""Auxiliary supervision for the compact VLA (Phase 9, question Q5).

Targets are derived from the recorded `privileged.*` columns, which are
training labels only; they never enter the policy input. Heads without a
recorded source (relation, event) stay unsupervised.

Grounding is coarse table-plane localisation rather than pixel keypoints: the
overhead camera's sensor-size intrinsics do not map cleanly onto the 256x256
policy render, while table-plane positions are exact and still require the
model to localise objects from RGB.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bimanual.policy.types import PolicyOutput

PHASES = (
    "APPROACH", "PRE_GRASP", "GRASP", "LIFT", "HOLD", "TRANSPORT",
    "PLACE", "PULL", "RELEASE", "SETTLE", "RETRACT",
)
PHASE_INDEX = {name: index for index, name in enumerate(PHASES)}
HOLDING_PHASES = frozenset({"LIFT", "HOLD", "TRANSPORT", "PLACE", "PULL"})
ARMS = ("left", "right")

OBJECT_ORDER = ("plate", "mug", "bottle", "fork", "spoon")
GROUNDING_OBJECTS = ("plate", "mug", "bottle", "fork")
TABLE_X = (-0.5, 0.5)
TABLE_Y = (-0.25, 0.5)

REQUIRED_COLUMNS = (
    "privileged.phase", "privileged.arm", "privileged.object_positions",
    "frame_index", "episode_index",
)
HEADS = ("phase", "progress", "holding", "grounding")


@dataclass(frozen=True)
class AuxWeights:
    phase: float = 1.0
    progress: float = 1.0
    holding: float = 1.0
    grounding: float = 1.0

    def scaled(self, factor: float) -> "AuxWeights":
        return AuxWeights(*(factor * getattr(self, name) for name in HEADS))


def _strings(values) -> list[str]:
    if isinstance(values, str):
        return [values]
    return [value[0] if isinstance(value, (list, tuple)) else str(value) for value in values]


def aux_targets(batch: dict, episode_lengths: dict[int, int], device: str | torch.device) -> dict[str, torch.Tensor]:
    """Build per-frame targets from one LeRobot batch."""
    missing = [key for key in REQUIRED_COLUMNS if key not in batch]
    if missing:
        raise KeyError(f"auxiliary supervision needs dataset columns {missing}")
    phases = _strings(batch["privileged.phase"])
    arms = _strings(batch["privileged.arm"])
    unknown = sorted({p for p in phases if p not in PHASE_INDEX})
    if unknown:
        raise ValueError(f"unknown expert phases {unknown}; extend PHASES")

    phase = torch.tensor([PHASE_INDEX[p] for p in phases], dtype=torch.long, device=device)

    frame = batch["frame_index"].reshape(-1).to(torch.float32)
    episodes = batch["episode_index"].reshape(-1).tolist()
    lengths = torch.tensor([episode_lengths[int(e)] for e in episodes], dtype=torch.float32)
    progress = (frame / (lengths - 1).clamp_min(1)).clamp(0, 1).unsqueeze(-1).to(device)

    holding = torch.zeros(len(phases), len(ARMS), device=device)
    for row, (p, arm) in enumerate(zip(phases, arms)):
        if p in HOLDING_PHASES and arm in ARMS:
            holding[row, ARMS.index(arm)] = 1.0

    positions = batch["privileged.object_positions"].reshape(len(phases), len(OBJECT_ORDER), 3)
    indices = [OBJECT_ORDER.index(name) for name in GROUNDING_OBJECTS]
    xy = positions[:, indices, :2].to(torch.float32)
    lo = torch.tensor([TABLE_X[0], TABLE_Y[0]])
    hi = torch.tensor([TABLE_X[1], TABLE_Y[1]])
    grounding = ((xy - lo) / (hi - lo)).clamp(0, 1).to(device)

    return {"phase": phase, "progress": progress, "holding": holding, "grounding": grounding}


def auxiliary_loss(
    output: PolicyOutput, targets: dict[str, torch.Tensor], weights: AuxWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of head losses plus unweighted per-head values for logging."""
    losses = {
        "phase": nn.functional.cross_entropy(output.phase_logits, targets["phase"]),
        "progress": nn.functional.mse_loss(output.progress, targets["progress"]),
        "holding": nn.functional.binary_cross_entropy_with_logits(output.holding_logits, targets["holding"]),
        "grounding": nn.functional.mse_loss(output.grounding, targets["grounding"]),
    }
    total = sum(getattr(weights, name) * value for name, value in losses.items())
    return total, {name: float(value.detach()) for name, value in losses.items()}


@torch.no_grad()
def aux_metrics(output: PolicyOutput, targets: dict[str, torch.Tensor]) -> dict[str, float]:
    """Interpretable held-out numbers: accuracies and mean absolute errors."""
    table_scale = torch.tensor([TABLE_X[1] - TABLE_X[0], TABLE_Y[1] - TABLE_Y[0]], device=output.grounding.device)
    return {
        "phase_accuracy": float((output.phase_logits.argmax(-1) == targets["phase"]).float().mean()),
        "progress_mae": float((output.progress - targets["progress"]).abs().mean()),
        "holding_accuracy": float(((output.holding_logits > 0).float() == targets["holding"]).float().mean()),
        "grounding_mae_m": float(((output.grounding - targets["grounding"]).abs() * table_scale).mean()),
    }
