"""Physical ALOHA trajectory execution with synchronized optional recording."""

from __future__ import annotations

import numpy as np

from bimanual.control.aloha_ik import AlohaIK, top_down_quaternion
from bimanual.control.trajectories import min_jerk_trajectory
from bimanual.sim.aloha_env import AlohaPhysicalEnv

OPEN = 0.037
CLOSED = 0.002


def _record_due(record: list[dict] | None, sim_time: float) -> bool:
    return record is not None and (not record or sim_time + 1e-9 >= record[-1]["timestamp"] + 0.1)


def step_recorded(
    env: AlohaPhysicalEnv,
    action: np.ndarray,
    arm: str,
    phase: str,
    record: list[dict] | None,
) -> None:
    if _record_due(record, float(env.data.time)):
        record.append(
            {
                "timestamp": float(env.data.time),
                "observation": env.observation(),
                "frames": env.render(),
                "oracle": env.oracle_state(),
                "action": action.astype(np.float32, copy=True),
                "arm": arm,
                "phase": phase,
            }
        )
    env.step(action)


def move_arm(
    env: AlohaPhysicalEnv,
    ik: AlohaIK,
    target: np.ndarray,
    gripper: float,
    steps: int,
    arm: str,
    phase: str,
    record: list[dict] | None,
    quaternion: np.ndarray | None = None,
) -> None:
    orientation = top_down_quaternion() if quaternion is None else np.asarray(quaternion, dtype=np.float64)
    result = ik.solve(env.data.qpos, {arm: (target, orientation)})
    if not result.converged[arm]:
        raise RuntimeError(
            f"ALOHA IK failed: position={result.position_error[arm]:.4f}, "
            f"orientation={result.orientation_error[arm]:.4f}"
        )
    start = 0 if arm == "left" else 7
    trajectory = min_jerk_trajectory(env.data.ctrl[start : start + 6], result.joint_targets[arm], steps)
    for joints in trajectory:
        action = env.data.ctrl.copy()
        action[start : start + 6] = joints
        action[start + 6] = gripper
        step_recorded(env, action, arm, phase, record)
