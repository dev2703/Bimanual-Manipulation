"""Differential IK for the Menagerie ALOHA 2 physical baseline."""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


@dataclass
class AlohaIKResult:
    joint_targets: dict[str, np.ndarray]
    position_error: dict[str, float]
    orientation_error: dict[str, float]
    converged: dict[str, bool]
    iterations: int


class AlohaIK:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.configuration = mink.Configuration(model)
        self.tasks = {
            arm: mink.FrameTask(
                frame_name=f"{arm}/gripper",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.2,
                lm_damping=1e-2,
            )
            for arm in ("left", "right")
        }
        self.posture = mink.PostureTask(model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(model)]

    def solve(
        self,
        qpos: np.ndarray,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        max_iters: int = 120,
        dt: float = 0.02,
        position_tolerance: float = 3e-3,
        orientation_tolerance: float = 0.12,
    ) -> AlohaIKResult:
        self.configuration.update(qpos.copy())
        self.posture.set_target(qpos)
        active = [self.posture]
        for arm, (position, quaternion) in targets.items():
            target = mink.SE3.from_rotation_and_translation(
                mink.SO3(np.asarray(quaternion, dtype=np.float64)),
                np.asarray(position, dtype=np.float64),
            )
            self.tasks[arm].set_target(target)
            active.append(self.tasks[arm])

        errors = {}
        iteration = 0
        for iteration in range(1, max_iters + 1):
            velocity = mink.solve_ik(self.configuration, active, dt, "daqp", limits=self.limits)
            self.configuration.integrate_inplace(velocity, dt)
            errors = {arm: self.tasks[arm].compute_error(self.configuration) for arm in targets}
            if all(
                np.linalg.norm(error[:3]) < position_tolerance
                and np.linalg.norm(error[3:]) < orientation_tolerance
                for error in errors.values()
            ):
                break

        position_error = {arm: float(np.linalg.norm(error[:3])) for arm, error in errors.items()}
        orientation_error = {arm: float(np.linalg.norm(error[3:])) for arm, error in errors.items()}
        joint_targets = {}
        for arm in targets:
            joint_targets[arm] = np.asarray(
                [self.configuration.q[self.model.joint(f"{arm}/{joint}").qposadr[0]] for joint in ARM_JOINTS]
            )
        converged = {
            arm: position_error[arm] < position_tolerance and orientation_error[arm] < orientation_tolerance
            for arm in targets
        }
        return AlohaIKResult(
            joint_targets=joint_targets,
            position_error=position_error,
            orientation_error=orientation_error,
            converged=converged,
            iterations=iteration,
        )


def top_down_quaternion() -> np.ndarray:
    """Gripper approach axis points down while finger motion stays lateral."""
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    return quaternion

