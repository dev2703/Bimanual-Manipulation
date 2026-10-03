"""Load a saved policy and the processors its rollout expects."""

from __future__ import annotations

from typing import Any

from bimanual.policy.lerobot_io import load_act_bundle

POLICY_KINDS = ("act", "smolvla", "pi05", "compact")


def uses_language(kind: str) -> bool:
    """ACT dinner checkpoints were trained without language; the VLAs need an instruction."""
    return kind != "act"


def _apply_prefix(policy: Any, prefix: int | None) -> None:
    if prefix is None:
        return
    chunk = getattr(policy.config, "chunk_size", None)
    if chunk is not None and not 1 <= prefix <= chunk:
        raise ValueError(f"prefix {prefix} outside 1..{chunk}")
    if hasattr(policy, "config") and hasattr(policy.config, "n_action_steps"):
        policy.config.n_action_steps = prefix


def load_policy(kind: str, checkpoint: str, device: str = "mps",
                prefix: int | None = None) -> tuple[Any, Any, Any]:
    """Return policy, preprocessor and postprocessor (which may be None)."""
    if kind == "act":
        policy, preprocessor, postprocessor = load_act_bundle(checkpoint, device)
        _apply_prefix(policy, prefix)
        return policy, preprocessor, postprocessor
    if kind == "smolvla":
        from bimanual.policy.smolvla.runner import load_smolvla_bundle
        return load_smolvla_bundle(checkpoint, device, prefix=prefix)
    if kind == "pi05":
        from bimanual.policy.pi05_runner import load_pi05_bundle
        policy, preprocessor, postprocessor = load_pi05_bundle(checkpoint, device)
        _apply_prefix(policy, prefix)
        return policy, preprocessor, postprocessor
    if kind == "compact":
        from bimanual.policy.compact_vla_runner import load_compact_bundle
        policy, preprocessor, _post = load_compact_bundle(checkpoint, device)
        return policy, preprocessor, None
    raise ValueError(f"unknown policy kind {kind!r}")
