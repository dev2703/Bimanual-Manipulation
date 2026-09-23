"""Proprioceptive inter-arm features for cooperative ALOHA policies.

This is an opt-in extension of the 14-D ACT state contract. Existing ACT
checkpoints must keep their original input shape; new datasets/checkpoints
must declare the 37-D state explicitly.
"""

from __future__ import annotations

import mujoco
import numpy as np

from bimanual.policy.types import EmbodimentSpec
from bimanual.sim.aloha_env import ALOHA_BIMANUAL, AlohaPhysicalEnv

RELATIVE_NAMES = tuple(f"right_in_left.{axis}" for axis in "xyz") + tuple(
    f"right_in_left.rotation_{column}.{axis}" for column in (0, 1) for axis in "xyz"
)
TWIST_NAMES = tuple(
    f"{arm}.ee_{kind}.{axis}"
    for arm in ("left", "right") for kind in ("linear_velocity", "angular_velocity") for axis in "xyz"
)
GRIPPER_NAMES = ("left.gripper_opening", "right.gripper_opening")
INTER_ARM_NAMES = RELATIVE_NAMES + TWIST_NAMES + GRIPPER_NAMES
COOPERATIVE_STATE_NAMES = ALOHA_BIMANUAL.state_names + INTER_ARM_NAMES
ALOHA_COOPERATIVE = EmbodimentSpec(
    name="aloha2_menagerie_cooperative",
    model_revision=ALOHA_BIMANUAL.model_revision,
    state_names=COOPERATIVE_STATE_NAMES,
    action_names=ALOHA_BIMANUAL.action_names,
    camera_names=ALOHA_BIMANUAL.camera_names,
    control_hz=ALOHA_BIMANUAL.control_hz,
    policy_hz=ALOHA_BIMANUAL.policy_hz,
    gripper_semantics=ALOHA_BIMANUAL.gripper_semantics,
)


def inter_arm_context(env: AlohaPhysicalEnv) -> np.ndarray:
    """Relative right gripper pose and both EE twists from robot state only.

    The relative pose is expressed in the left gripper frame. MuJoCo spatial
    velocities are returned in world coordinates, angular before linear.
    Nothing here reads an object pose, goal site, contact, or oracle state.
    """
    model, data = env.model, env.data
    left_id = model.site("left/gripper").id
    right_id = model.site("right/gripper").id
    left_rotation = data.site_xmat[left_id].reshape(3, 3)
    right_rotation = data.site_xmat[right_id].reshape(3, 3)
    relative_position = left_rotation.T @ (data.site_xpos[right_id] - data.site_xpos[left_id])
    relative_rotation = left_rotation.T @ right_rotation
    features: list[np.ndarray] = [relative_position, relative_rotation[:, :2].T.reshape(-1)]
    for site_id in (left_id, right_id):
        twist = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, twist, 0)
        features.extend((twist[3:], twist[:3]))
    state = env.state_vector()
    features.append(state[[6, 13]])
    result = np.concatenate(features).astype(np.float32)
    if result.shape != (len(INTER_ARM_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("invalid inter-arm proprioception")
    return result


def cooperative_state(env: AlohaPhysicalEnv) -> np.ndarray:
    """37-D opt-in state; paired dataset and checkpoint schemas must match."""
    state = np.concatenate((env.state_vector(), inter_arm_context(env)))
    ALOHA_COOPERATIVE.validate(state)
    return state
