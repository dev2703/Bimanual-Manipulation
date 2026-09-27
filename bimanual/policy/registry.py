"""Load a saved policy and the processors its rollout expects."""

from __future__ import annotations

from typing import Any

from bimanual.policy.lerobot_io import load_act_bundle
from bimanual.skills.registry import get_skill


def _apply_prefix(policy: Any, prefix: int | None) -> None:
    if prefix is None:
        return
    chunk = getattr(policy.config, "chunk_size", None)
    if chunk is not None and not 1 <= prefix <= chunk:
        raise ValueError(f"prefix {prefix} outside 1..{chunk}")
    if hasattr(policy, "config") and hasattr(policy.config, "n_action_steps"):
        policy.config.n_action_steps = prefix


def load_policy(
    kind: str, checkpoint: str, device: str = "mps", prefix: int | None = None, skill: str = "mug_pick_place",
) -> tuple[Any, Any, Any, str | None]:
    """Return policy, preprocessor, postprocessor, and the instruction to feed.

    ACT dinner checkpoints were trained without language, so the instruction is None.
    """
    if kind == "act":
        policy, preprocessor, postprocessor = load_act_bundle(checkpoint, device)
        _apply_prefix(policy, prefix)
        return policy, preprocessor, postprocessor, None
    if kind == "smolvla":
        from bimanual.policy.smolvla.runner import load_smolvla_bundle
        policy, preprocessor, postprocessor = load_smolvla_bundle(checkpoint, device, prefix=prefix)
        return policy, preprocessor, postprocessor, get_skill(skill).instruction
    if kind == "pi05":
        from bimanual.policy.pi05_runner import load_pi05_bundle
        policy, preprocessor, postprocessor = load_pi05_bundle(checkpoint, device)
        _apply_prefix(policy, prefix)
        return policy, preprocessor, postprocessor, get_skill(skill).instruction
    if kind == "compact":
        from bimanual.policy.compact_vla_runner import load_compact_bundle
        policy, preprocessor, _post = load_compact_bundle(checkpoint, device)
        return policy, preprocessor, None, get_skill(skill).instruction
    raise ValueError(f"unknown policy kind {kind!r}")
