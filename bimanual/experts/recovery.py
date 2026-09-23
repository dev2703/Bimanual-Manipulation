"""Perturbation injection for recovery-trajectory generation (plan.md
section 17 bucket D / decision A6).

Minimal version given Phase 3 time budget: perturb an object's position
mid-episode (simulating "something moved the object after the plan was
made") and let the SAME scripted skill re-observe and retry from the new
position, rather than blindly replaying the original plan. This is the
simplest form of the observe -> replan loop the full executor (Phase 5)
will generalize.
"""

from __future__ import annotations

import numpy as np

from bimanual.control.ik import BimanualIK
from bimanual.experts.executor import execute
from bimanual.experts.skills import pick, place, place_target
from bimanual.logging_utils import get_logger
from bimanual.sim.env import BimanualTableEnv
from bimanual.sim.oracle_predicates import in_region
from bimanual.sim.scene_builder import REGIONS

log = get_logger(__name__)


def perturb_object_position(env: BimanualTableEnv, obj_name: str, rng: np.random.Generator, max_shift: float = 0.06) -> None:
    """Directly displaces an object's freejoint position by a random
    horizontal shift -- the "someone moved it" event. Zeroes velocity so
    it doesn't look like the object was thrown."""
    assert env.model is not None and env.data is not None
    adr = env._obj_qpos_adr(obj_name)
    shift = rng.uniform(-max_shift, max_shift, size=2)
    env.data.qpos[adr] += shift[0]
    env.data.qpos[adr + 1] += shift[1]
    joint_id = env.model.body(obj_name).jntadr[0]
    dof_start = env.model.jnt_dofadr[joint_id]
    env.data.qvel[dof_start : dof_start + 6] = 0.0
    import mujoco

    mujoco.mj_forward(env.model, env.data)


def pick_with_recovery(
    ik: BimanualIK,
    env: BimanualTableEnv,
    prefix: str,
    obj_name: str,
    region: str,
    rng: np.random.Generator,
    perturb_after: str = "APPROACH",
) -> tuple[bool, bool]:
    """Runs pick() normally, but after the phase named perturb_after,
    injects a perturbation and RE-PLANS the rest of pick() from the
    object's NEW oracle position rather than continuing the stale plan.
    Returns (recovered, placed_ok)."""
    obj_pos = env.oracle_state()[f"{obj_name}_pos"].copy()
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
    segs = pick(ik, env.data.qpos, prefix, ee_pos, obj_pos, obj_name)

    ran_any = False
    for seg in segs:
        execute(env, prefix, [seg], obj_name=obj_name)
        ran_any = True
        if seg.phase == perturb_after:
            pre_perturb_pos = env.oracle_state()[f"{obj_name}_pos"].copy()
            perturb_object_position(env, obj_name, rng)
            # replan the REMAINDER from the object's new position
            new_obj_pos = env.oracle_state()[f"{obj_name}_pos"].copy()
            log.info(
                "perturbed %s after phase %s: %s -> %s, replanning",
                obj_name, perturb_after, pre_perturb_pos.round(3), new_obj_pos.round(3),
            )
            new_ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
            remaining = pick(ik, env.data.qpos, prefix, new_ee_pos, new_obj_pos, obj_name)
            execute(env, prefix, remaining, obj_name=obj_name)
            break

    rx, ry, _ = REGIONS[region]
    target = place_target(obj_name, (rx, ry))
    ee2 = env.proprio()[f"{prefix}_ee_pos"].copy()
    place_segs = place(ik, env.data.qpos, prefix, ee2, target, obj_name)
    execute(env, prefix, place_segs, obj_name=obj_name)
    for _ in range(10):
        env.step(env.data.ctrl)

    placed_ok = in_region(env.oracle_state(), obj_name, region)
    return True, placed_ok
