"""Embodiment-neutral contracts shared by learned policy runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EmbodimentSpec:
    name: str
    model_revision: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    camera_names: tuple[str, ...]
    control_hz: int = 30
    policy_hz: int = 10
    gripper_semantics: str = "absolute_joint_position"

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    def validate(self, state: np.ndarray, action: np.ndarray | None = None) -> None:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"{self.name} expects state dim {self.state_dim}, got {state.shape[-1]}")
        if action is not None and action.shape[-1] != self.action_dim:
            raise ValueError(f"{self.name} expects action dim {self.action_dim}, got {action.shape[-1]}")


SO101_STATE_NAMES = tuple(
    f"{arm}.{joint}"
    for arm in ("left", "right")
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
)

SO101_BIMANUAL = EmbodimentSpec(
    name="bimanual_so101",
    model_revision="legacy-v1",
    state_names=SO101_STATE_NAMES,
    action_names=SO101_STATE_NAMES,
    camera_names=("global", "left_wrist", "right_wrist"),
)


@dataclass
class PolicyObservation:
    images: dict[str, Any]
    state: Any
    language: Any
    memory: Any | None = None
    recent_actions: Any | None = None
    inter_arm_context: Any | None = None
    timestamp: float | None = None


@dataclass
class PolicyOutput:
    actions: Any
    phase_logits: Any | None = None
    progress: Any | None = None
    grounding: Any | None = None
    relation_logits: Any | None = None
    holding_logits: Any | None = None
    event_logits: Any | None = None


@dataclass(frozen=True)
class EpisodeManifest:
    environment: str
    embodiment: str
    model_revision: str
    seed: int
    split: str
    source_expert: str
    physics_hz: int = 600
    control_hz: int = 30
    policy_hz: int = 10
    artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
