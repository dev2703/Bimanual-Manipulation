"""Closed-loop ACT policy execution against BimanualTableEnv (Phase 4
gate: "ACT >=50% on pick-plate at L0 and L1 with closed-loop execution
in env.py", docs/phases.md).

Converts env.render()/proprio() into the batch dict ACTPolicy.
select_action() expects (matching generate.py's dataset schema exactly:
observation.state is the 12-D raw joint-qpos-plus-gripper vector, NOT a
normalized fraction), runs the policy at CONTROL_HZ, and writes its
12-D output straight into the arm/gripper actuators, leaving drawer_act
at 0 (ACT was never trained on drawer actuation for the pick-plate
bucket).
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Any

from bimanual.logging_utils import get_logger
from bimanual.sim import robot_spec as rs
from bimanual.sim.env import BimanualTableEnv
from bimanual.sim.oracle_predicates import in_region
from bimanual.sim.scene_builder import REGIONS

log = get_logger(__name__)


def _frame_to_batch_image(frame: np.ndarray, device: str) -> torch.Tensor:
    """uint8 (H, W, 3) -> float32 (1, 3, H, W) in [0, 1], matching
    LeRobotDataset's own image convention."""
    t = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
    return t.unsqueeze(0).to(device)


def load_act_bundle(checkpoint: str | Path, device: str = "mps") -> tuple[Any, Any, Any]:
    """Load ACT and the exact normalization pipelines saved with it."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    checkpoint = Path(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    overrides = {"device_processor": {"device": device}}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides=overrides,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides=overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def _bimanual_state_vector(env: BimanualTableEnv) -> np.ndarray:
    """Must exactly match experts/generate.py's _bimanual_vector layout:
    left 5 joints + gripper, then right 5 joints + gripper."""
    proprio = env.proprio()
    out = []
    for prefix in ("left", "right"):
        out.extend(proprio[f"{prefix}_joint_pos"][:5])
        out.append(proprio[f"{prefix}_gripper_opening"])
    return np.asarray(out, dtype=np.float32)


def _action_to_ctrl(env: BimanualTableEnv, action: np.ndarray) -> np.ndarray:
    """12-D ACT action (same layout as _bimanual_state_vector) -> full
    model.nu actuator ctrl vector, drawer left at 0."""
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    ctrl = np.zeros(env.model.nu)
    idx = 0
    for prefix in ("left", "right"):
        for suffix in rs.JOINT_SUFFIXES:
            ctrl[names.index(rs.joint_name(prefix, suffix))] = action[idx]
            idx += 1
        ctrl[names.index(rs.gripper_joint_name(prefix))] = action[idx]
        idx += 1
    return ctrl


def run_closed_loop_episode(
    env: BimanualTableEnv,
    policy,
    obj: str = "plate",
    region: str = "plate_region",
    max_steps: int = 300,
    device: str = "mps",
    preprocessor=None,
    postprocessor=None,
) -> bool:
    """Runs `policy` closed-loop (re-querying every control tick, no
    action-chunk caching) until `obj` reads as in `region` via the
    oracle, or max_steps is hit. Returns whether it succeeded. Oracle
    success-checking here is an EVALUATION oracle (allowed per A2/A3 --
    only the policy's OWN inputs are proprioception plus images, checked
    below by construction, not the success signal used to stop it)."""
    policy.reset()
    for step in range(max_steps):
        frames = env.render()
        batch = {
            "observation.images.global": _frame_to_batch_image(frames["global"], device),
            "observation.images.left_wrist": _frame_to_batch_image(frames["left_wrist_cam"], device),
            "observation.images.right_wrist": _frame_to_batch_image(frames["right_wrist_cam"], device),
            "observation.state": torch.from_numpy(_bimanual_state_vector(env)).unsqueeze(0).to(device),
        }
        if preprocessor is not None:
            # Saved LeRobot processors include batching/device placement.
            raw = {key: value.squeeze(0).cpu() for key, value in batch.items()}
            batch = preprocessor(raw)
        with torch.no_grad():
            action = policy.select_action(batch)
        if postprocessor is not None:
            action = postprocessor(action)
        action_np = action.squeeze(0).cpu().numpy()

        ctrl = _action_to_ctrl(env, action_np)
        env.step(ctrl)

        if in_region(env.oracle_state(), obj, region):
            log.info("closed-loop success at step %d", step)
            return True

    log.info("closed-loop episode timed out after %d steps without success", max_steps)
    return False
