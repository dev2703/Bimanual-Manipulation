"""Exogenous object movement for contact-task recovery benchmarks.

This is an environment disturbance between policy actions, never an expert
grasp shortcut or an inference input. The resulting image must be observed
before the expert or policy chooses its next action.
"""

from __future__ import annotations

import mujoco
import numpy as np

from bimanual.sim.aloha_env import AlohaTableSettingEnv


def move_ungrasped_object(env: AlohaTableSettingEnv, name: str, delta_xy: np.ndarray) -> None:
    delta = np.asarray(delta_xy, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > 0.06:
        raise ValueError("disturbance must be a finite XY displacement of at most 6 cm")
    joint = env.model.joint(f"{name}_free")
    qpos = int(joint.qposadr[0])
    dof = int(joint.dofadr[0])
    env.data.qpos[qpos:qpos + 2] += delta
    env.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def close_drawer_exogenously(env: AlohaTableSettingEnv) -> None:
    """Training/evaluation disturbance after the expert has released the handle."""
    joint = env.model.joint("drawer_slide")
    env.data.qpos[joint.qposadr[0]] = 0.0
    env.data.qvel[joint.dofadr[0]] = 0.0
    mujoco.mj_forward(env.model, env.data)
