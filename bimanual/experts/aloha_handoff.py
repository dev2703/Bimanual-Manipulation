"""Contact-only two-arm transfer of a graspable dinner-service baton."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaPhysicalEnv

GRASP_OFFSET = 0.052
# A closer-to-center receiver grasp was tried to reduce the post-release
# cantilever moment (see RECEIVER_SETTLE below), but it made the receiver's
# target sit too close to the carrier's own gripper and collapsed contact
# rate further (measured: ~30% -> ~5% of seeds getting any contact). Reverted
# to the symmetric far-end offset; the cantilever-slip failure mode after
# release is still open and needs dedicated tuning, not another guess.


def receiver_grasp_from_object(env: AlohaPhysicalEnv) -> np.ndarray:
    """Expert-only target: right end of the moving baton in world space."""
    body = env.model.body("task_block").id
    rotation = env.data.xmat[body].reshape(3, 3)
    return env.data.xpos[body].copy() + rotation[:, 0] * GRASP_OFFSET + [0, 0, 0.015]


@dataclass(frozen=True)
class AlohaHandoffResult:
    success: bool
    initial_height: float
    peak_height: float
    after_release_height: float
    final_height: float
    carrier_opening: float
    receiver_opening: float
    record: list[dict]
    receiver_contact_steps: int = 0


def handoff_succeeded(
    initial_height: float, peak_height: float, after_release_height: float,
    final_height: float, carrier_opening: float, receiver_opening: float,
    receiver_contact_steps: int,
) -> bool:
    """Require receiver contact, carrier release, and retained support."""
    return bool(peak_height > initial_height + 0.07
                and receiver_contact_steps >= 20
                and after_release_height > initial_height + 0.06
                and final_height > initial_height + 0.06
                and carrier_opening > 0.03 and receiver_opening < 0.02)


def _track_to_object(
    env: AlohaPhysicalEnv, ik: AlohaIK, offset: list[float], total_steps: int,
    arm: str, phase: str, record: list[dict] | None, segment_steps: int = 16,
    gripper: float = OPEN,
) -> None:
    remaining = total_steps
    while remaining > 0:
        step = min(segment_steps, remaining)
        target = receiver_grasp_from_object(env) + np.asarray(offset)
        try:
            move_arm(env, ik, target, gripper, step, arm, phase, record)
        except RuntimeError:
            # The swinging object can momentarily put the re-localized target
            # just outside this segment's reachable envelope; hold position
            # and re-localize again next segment rather than aborting the
            # whole handoff over one transient IK miss.
            pass
        remaining -= step


def run_baton_handoff(env: AlohaPhysicalEnv, record_frames: bool = False) -> AlohaHandoffResult:
    """Left carries, right pinches, left unloads, right retains and retreats."""
    ik = AlohaIK(env.model)
    record: list[dict] | None = [] if record_frames else None
    initial = env.oracle_state()["task_block_pos"].copy()
    left_grasp = initial + [-GRASP_OFFSET, 0, 0.038]
    move_arm(env, ik, left_grasp + [0, 0, 0.16], OPEN, 50, "left", "CARRIER_APPROACH", record)
    move_arm(env, ik, left_grasp, OPEN, 50, "left", "CARRIER_PRE_GRASP", record)
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[6] = CLOSED
        step_recorded(env, action, "left", "CARRIER_GRASP", record)
    left_site = env.data.site_xpos[env.model.site("left/gripper").id].copy()
    move_arm(env, ik, left_site + [0, 0, 0.10], CLOSED, 100, "left", "CARRIER_LIFT", record)
    peak = float(env.oracle_state()["task_block_pos"][2])
    # A single-point grasp leaves the baton swinging after the lift; let it
    # damp out before the receiver starts tracking it, or the receiver spends
    # its whole approach chasing a moving target and never settles on it.
    for _ in range(60):
        step_recorded(env, env.data.ctrl.copy(), "left", "CARRIER_SETTLE", record)

    # Establish a stationary rendezvous in the shared workspace. Asking the
    # receiver to chase the freely swinging baton produced intermittent
    # glancing contacts; the carrier can instead move the measured receiver
    # grasp point to one fixed pose and damp it before the second arm arrives.
    rendezvous = np.array([0.04, 0.02, 0.16])
    receiver_point = receiver_grasp_from_object(env)
    left_site = env.data.site_xpos[env.model.site("left/gripper").id].copy()
    move_arm(
        env, ik, left_site + rendezvous - receiver_point,
        CLOSED, 140, "left", "CARRIER_PRESENT", record,
    )
    for _ in range(80):
        step_recorded(env, env.data.ctrl.copy(), "left", "PRESENT_SETTLE", record)

    _track_to_object(env, ik, [0, 0, 0.07], 80, "right", "RECEIVER_APPROACH", record)
    _track_to_object(env, ik, [0, 0, 0], 100, "right", "RECEIVER_PRE_GRASP", record)
    # The carrier closes on a table-supported, motionless object before ever
    # lifting; the receiver instead closes on an already-swinging, mid-air
    # object. Hold the tracked position open a moment longer so the arm's
    # own residual approach velocity settles before the fingers close,
    # rather than sealing a glancing grip mid-motion.
    for _ in range(20):
        step_recorded(env, env.data.ctrl.copy(), "right", "RECEIVER_DWELL", record)
    # Keep following the moving grasp point while the fingers close. Holding
    # the last open-loop joint target here let the suspended baton swing out
    # of the receiver before closure completed.
    _track_to_object(
        env, ik, [0, 0, 0], 100, "right", "RECEIVER_CLOSE_TRACK",
        record, segment_steps=8, gripper=CLOSED,
    )
    receiver_contacts = 0
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[13] = CLOSED
        step_recorded(env, action, "right", "DUAL_HOLD", record)
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            names = (env.model.geom(contact.geom1).name, env.model.geom(contact.geom2).name)
            if "handoff_baton" in names and any(name.startswith("right/") for name in names):
                receiver_contacts += 1
                break
    for _ in range(40):
        action = env.data.ctrl.copy()
        action[6] = OPEN
        step_recorded(env, action, "left", "CARRIER_RELEASE", record)
    # The object becomes a single-point cantilever the instant the carrier
    # releases; let the receiver's pinch re-equilibrate under that new load
    # before any further motion disturbs it.
    for _ in range(30):
        action = env.data.ctrl.copy()
        action[13] = CLOSED
        step_recorded(env, action, "right", "RECEIVER_SETTLE", record)
    after_release = float(env.oracle_state()["task_block_pos"][2])
    left_site = env.data.site_xpos[env.model.site("left/gripper").id].copy()
    move_arm(env, ik, left_site + [-0.08, 0, 0.08], OPEN, 80, "left", "CARRIER_RETREAT", record)
    right_site = env.data.site_xpos[env.model.site("right/gripper").id].copy()
    move_arm(env, ik, right_site + [0.06, 0, 0.06], CLOSED, 80, "right", "RECEIVER_RETREAT", record)
    for _ in range(30):
        step_recorded(env, env.data.ctrl.copy(), "right", "RECEIVER_HOLD", record)
    final = float(env.oracle_state()["task_block_pos"][2])
    state = env.state_vector()
    success = handoff_succeeded(float(initial[2]), peak, after_release, final,
                                float(state[6]), float(state[13]), receiver_contacts)
    return AlohaHandoffResult(success, float(initial[2]), peak, after_release,
                              final, float(state[6]), float(state[13]), record or [], receiver_contacts)
