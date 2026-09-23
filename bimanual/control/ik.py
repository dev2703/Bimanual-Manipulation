"""Differential IK per arm, built on mink.

Decision A1 (docs/decisions.md): the task space is position-dominant with
a low-weight orientation cost, not a full 6-DOF pose target -- the
vendored SO-101 has 5 joints per arm and cannot generically track an
arbitrary orientation. Orientation is a soft task here rather than a hard
constraint, so the solver prioritizes reaching the target position and
only weakly pulls the gripper toward the requested approach direction.

This module solves both arms in ONE mink.Configuration built from the
live BimanualTableEnv model, using a separate FrameTask per arm. Object
freejoints, the drawer, and (when solving for one arm) the other arm's
joints have no task referencing them, so they stay effectively fixed
during the solve (mink's damping term pulls unconstrained velocities
toward zero) -- there's no need for two separate single-arm sub-models.

IK is solved in mink's own scratch MjData (owned by Configuration), never
against the env's live `data` -- callers read off the resulting joint
targets and hand them to env.step() themselves, keeping this module a
pure planner with no side effects on the simulated environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from bimanual.control.collision import build_arm_collision_avoidance_limits
from bimanual.sim import robot_spec as rs

POSITION_COST = 1.0
ORIENTATION_COST = 0.01  # soft: see decision A1 above.
# 0.05 caused a local plateau ~13mm short of some targets during Phase 2
# testing (a QP local-minimum artifact, not slow convergence -- more
# iterations didn't help). 0.01 keeps ~0.5mm of extra convergence margin
# under the 5mm gate while still weakly biasing orientation.
LM_DAMPING = 1e-2
POSTURE_COST = 1e-2
SOLVER = "daqp"
DT = 0.02
MAX_ITERS = 80
POS_TOL = 5e-3  # meters, matches the Phase 2 gate in docs/phases.md
ORI_TOL = 0.35  # rad, loose since orientation is a soft task


@dataclass
class IKResult:
    joint_targets: dict[str, np.ndarray]  # prefix -> 5-vector, order = rs.JOINT_SUFFIXES
    pos_error: dict[str, float]  # prefix -> meters
    ori_error: dict[str, float]  # prefix -> radians
    converged: dict[str, bool]
    iterations: int


class BimanualIK:
    """Owns one mink.Configuration + one FrameTask per arm for the whole
    bimanual scene."""

    def __init__(self, model: mujoco.MjModel, avoid_collisions: bool = True) -> None:
        self.model = model
        self.configuration = mink.Configuration(model)
        self.tasks: dict[str, mink.FrameTask] = {
            prefix: mink.FrameTask(
                frame_name=rs.ee_site_name(prefix),
                frame_type="site",
                position_cost=POSITION_COST,
                orientation_cost=ORIENTATION_COST,
                lm_damping=LM_DAMPING,
            )
            for prefix in ("left", "right")
        }
        # Posture regularization: with only 2 spare DOF once position (3)
        # is constrained, the solver's nullspace resolution otherwise
        # picks an arbitrary elbow/forearm configuration each solve --
        # empirically this let the forearm swing through a tabletop
        # object even while the tracked end-effector site moved along a
        # provably clear straight line (Phase 3 finding, docs/decisions.md).
        # A low-weight PostureTask biases redundant DOF toward staying
        # near the arm's current configuration instead.
        self.posture_task = mink.PostureTask(model, cost=POSTURE_COST)
        self.limits: list = [mink.ConfigurationLimit(model)]
        if avoid_collisions:
            # Steers the IK velocity solve away from arm-arm/arm-table
            # contact before it happens, on top of the hard joint limits
            # above. Phase 2 gate: 100 random dual-arm reach seeds, zero
            # arm-arm contacts (tests/test_ik.py).
            self.limits.extend(build_arm_collision_avoidance_limits(model))

    def solve(
        self,
        qpos: np.ndarray,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        max_iters: int = MAX_ITERS,
        dt: float = DT,
        pos_tol: float = POS_TOL,
        ori_tol: float = ORI_TOL,
        extra_limits: list | None = None,
    ) -> IKResult:
        """targets: {prefix: (world_pos[3], world_quat_wxyz[4])}. Only the
        arms present in `targets` are actively driven; the rest of the
        configuration (other arm, objects, drawer) stays fixed."""
        self.configuration.update(qpos.copy())

        self.posture_task.set_target(qpos)
        active_tasks = [self.posture_task]
        for prefix, (pos, quat) in targets.items():
            se3 = mink.SE3.from_rotation_and_translation(
                mink.SO3(np.asarray(quat, dtype=np.float64)), np.asarray(pos, dtype=np.float64)
            )
            self.tasks[prefix].set_target(se3)
            active_tasks.append(self.tasks[prefix])

        limits = self.limits + list(extra_limits or [])
        n_iters = 0
        for n_iters in range(1, max_iters + 1):
            vel = mink.solve_ik(
                self.configuration, active_tasks, dt, SOLVER, limits=limits
            )
            self.configuration.integrate_inplace(vel, dt)
            errs = {
                prefix: self.tasks[prefix].compute_error(self.configuration)
                for prefix in targets
            }
            # Position-primary convergence (decision A1): with 5 joints,
            # a full 3-DOF position + 3-DOF orientation match is generally
            # over-constrained, so orientation is tracked as a soft task
            # and reported for diagnostics but does not gate convergence.
            if all(np.linalg.norm(e[:3]) < pos_tol for e in errs.values()):
                break

        pos_error = {p: float(np.linalg.norm(e[:3])) for p, e in errs.items()}
        ori_error = {p: float(np.linalg.norm(e[3:])) for p, e in errs.items()}
        converged = {p: pos_error[p] < pos_tol for p in targets}

        joint_targets = {}
        for prefix in targets:
            joint_targets[prefix] = np.array(
                [self._get_qpos(name) for name in rs.arm_joint_names(prefix)]
            )

        return IKResult(
            joint_targets=joint_targets,
            pos_error=pos_error,
            ori_error=ori_error,
            converged=converged,
            iterations=n_iters,
        )

    def _get_qpos(self, joint_name: str) -> float:
        adr = self.model.joint(joint_name).qposadr[0]
        return float(self.configuration.q[adr])
