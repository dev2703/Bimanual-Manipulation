"""Tests for the scripted-expert layer (bimanual/experts/*), previously
uncovered: skills.py (pick/place/open_drawer), executor.py, recovery.py,
table_setting.py.

These need a live BimanualTableEnv + BimanualIK (unlike
tests/test_control_and_predicates.py's pure-logic tests), so they're
slower and fewer in number -- focused on behavioral contracts and known
edge cases rather than success-rate measurement (that's
scripts/expert_success.py's job, run manually per docs/decisions.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from bimanual.control.ik import BimanualIK
from bimanual.experts.executor import SETTLE_MAX_TICKS, SETTLE_PHASES, execute, execute_open_drawer
from bimanual.experts.recovery import perturb_object_position, pick_with_recovery
from bimanual.experts.skills import open_drawer, pick, place
from bimanual.experts.table_setting import TASK_GRAPH, assign_arm, run_full_episode
from bimanual.sim.env import BimanualTableEnv, ResetOptions
from bimanual.sim.oracle_predicates import in_region
from bimanual.sim.randomization import sample_scene_config
from bimanual.sim import robot_spec as rs
from bimanual.sim.scene_builder import DRAWER_OPEN_DIST, REGIONS


@pytest.fixture()
def env_and_ik():
    env = BimanualTableEnv()
    ik = BimanualIK(env.model)
    return env, ik


# --------------------------------------------------------------------------
# assign_arm
# --------------------------------------------------------------------------


def test_assign_arm_sign_boundary():
    assert assign_arm(-0.01) == "left"
    assert assign_arm(0.0) == "right"  # x==0 is not < 0, so it goes right
    assert assign_arm(0.01) == "right"


# --------------------------------------------------------------------------
# pick() / place() produce well-formed, physically sane trajectories
# --------------------------------------------------------------------------


def test_pick_returns_expected_phase_sequence(env_and_ik):
    env, ik = env_and_ik
    obj_pos = env.oracle_state()["mug_pos"].copy()
    prefix = assign_arm(obj_pos[0])
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()

    segments = pick(ik, env.data.qpos, prefix, ee_pos, obj_pos, "mug")
    phases = [s.phase for s in segments]
    assert phases == ["RISE", "TRANSPORT", "APPROACH", "GRASP", "LIFT"]
    for seg in segments:
        assert seg.trajectory.n_steps > 0
        assert np.all(np.isfinite(seg.trajectory.joint_targets))
        assert np.all(np.isfinite(seg.trajectory.gripper_targets))


def test_place_returns_expected_phase_sequence(env_and_ik):
    env, ik = env_and_ik
    obj_pos = env.oracle_state()["plate_pos"].copy()
    prefix = assign_arm(obj_pos[0])
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
    target = np.array([REGIONS["plate_region"][0], REGIONS["plate_region"][1], 0.40])

    segments = place(ik, env.data.qpos, prefix, ee_pos, target, "plate")
    phases = [s.phase for s in segments]
    assert phases == ["RISE", "TRANSPORT", "PLACE", "RELEASE", "RETRACT"]


def test_open_drawer_returns_expected_phase_sequence(env_and_ik):
    env, ik = env_and_ik
    prefix = "left"
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
    handle_pos = env.data.geom_xpos[env.model.geom("drawer_handle").id].copy()

    segments = open_drawer(ik, env.data.qpos, prefix, ee_pos, DRAWER_OPEN_DIST, handle_pos)
    phases = [s.phase for s in segments]
    assert phases == ["RISE", "TRANSPORT", "APPROACH", "GRASP", "PULL", "RELEASE", "RETRACT"]


# --------------------------------------------------------------------------
# executor.execute / execute_open_drawer against a live env
# --------------------------------------------------------------------------


def test_execute_grasp_and_release_attach_and_detach_object(env_and_ik):
    env, ik = env_and_ik
    obj = "plate"
    obj_pos = env.oracle_state()[f"{obj}_pos"].copy()
    prefix = assign_arm(obj_pos[0])
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()

    pick_segs = pick(ik, env.data.qpos, prefix, ee_pos, obj_pos, obj)
    assert env._held[prefix] is None
    execute(env, prefix, pick_segs, obj_name=obj)
    # pick() ends holding the object (GRASP ran, no RELEASE yet).
    assert env._held[prefix] is not None
    assert env._held[prefix][0] == obj

    ee2 = env.proprio()[f"{prefix}_ee_pos"].copy()
    target = np.array([REGIONS["plate_region"][0], REGIONS["plate_region"][1], 0.40])
    place_segs = place(ik, env.data.qpos, prefix, ee2, target, obj)
    execute(env, prefix, place_segs, obj_name=obj)
    assert env._held[prefix] is None


def test_execute_without_obj_name_still_auto_grasps_via_env_step(env_and_ik):
    # execute()'s own explicit grasp() call is skipped when obj_name is
    # None, but BimanualTableEnv.step()'s auto-grasp (proximity + closed
    # gripper, see env.py) is independent of executor.py entirely -- it
    # fires regardless of whether the caller passed obj_name, which is
    # exactly the point: a learned policy driving the arm through
    # env.step() directly (no obj_name concept at all) must still be
    # able to pick things up.
    env, ik = env_and_ik
    obj_pos = env.oracle_state()["mug_pos"].copy()
    prefix = assign_arm(obj_pos[0])
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
    segs = pick(ik, env.data.qpos, prefix, ee_pos, obj_pos, "mug")

    execute(env, prefix, segs, obj_name=None)
    assert env._held[prefix] is not None
    assert env._held[prefix][0] == "mug"


def test_execute_record_hook_captures_one_entry_per_tick(env_and_ik):
    env, ik = env_and_ik
    obj_pos = env.oracle_state()["plate_pos"].copy()
    prefix = assign_arm(obj_pos[0])
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
    segs = pick(ik, env.data.qpos, prefix, ee_pos, obj_pos, "plate")

    record: list = []
    execute(env, prefix, segs, obj_name="plate", record=record)

    # Control executes at 30 Hz while synchronized policy observations are
    # captured at 10 Hz, including any adaptive settling interval.
    planned_ticks = sum(s.trajectory.n_steps for s in segs)
    n_settle_phases = sum(1 for s in segs if s.phase in SETTLE_PHASES)
    assert planned_ticks // 3 <= len(record) <= (planned_ticks + n_settle_phases * SETTLE_MAX_TICKS) // 3 + 1
    for entry in record:
        assert set(entry) == {"phase", "arm", "proprio", "oracle_state", "action", "frames", "timestamp"}
        assert entry["arm"] == prefix
        assert entry["action"].shape == (rs.N_BIMANUAL_ACTIONS,)
        assert set(entry["frames"]) == {"global", "left_wrist_cam", "right_wrist_cam"}
    timestamps = np.array([entry["timestamp"] for entry in record])
    np.testing.assert_allclose(np.diff(timestamps), 0.1, atol=1e-8)


def test_execute_open_drawer_advances_drawer_actuator(env_and_ik):
    env, ik = env_and_ik
    prefix = "left"
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
    handle_pos = env.data.geom_xpos[env.model.geom("drawer_handle").id].copy()
    segs = open_drawer(ik, env.data.qpos, prefix, ee_pos, DRAWER_OPEN_DIST, handle_pos)

    execute_open_drawer(env, prefix, segs, DRAWER_OPEN_DIST)
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    drawer_ctrl = env.data.ctrl[names.index("drawer_act")]
    # PULL phase should have ramped the drawer actuator toward fully open
    # (negative direction, see scene_builder.py) by the end of execution.
    assert drawer_ctrl == pytest.approx(-DRAWER_OPEN_DIST, abs=1e-6)


# --------------------------------------------------------------------------
# recovery.py
# --------------------------------------------------------------------------


def test_perturb_object_position_shifts_xy_within_bound_and_zeros_velocity(env_and_ik):
    env, ik = env_and_ik
    rng = np.random.default_rng(0)
    before = env.oracle_state()["mug_pos"].copy()

    perturb_object_position(env, "mug", rng, max_shift=0.06)

    after = env.oracle_state()["mug_pos"].copy()
    xy_shift = np.hypot(after[0] - before[0], after[1] - before[1])
    assert 0.0 <= xy_shift <= 0.06 * np.sqrt(2) + 1e-9

    joint_id = env.model.body("mug").jntadr[0]
    dof_start = env.model.jnt_dofadr[joint_id]
    assert np.all(env.data.qvel[dof_start : dof_start + 6] == 0.0)


def test_perturb_object_position_does_not_move_z(env_and_ik):
    env, ik = env_and_ik
    rng = np.random.default_rng(1)
    before_z = env.oracle_state()["mug_pos"][2]
    perturb_object_position(env, "mug", rng)
    after_z = env.oracle_state()["mug_pos"][2]
    # Only x, y are perturbed; z (height) is untouched by design (the
    # object should still be resting on the table, not teleported into
    # the air or through it).
    assert after_z == pytest.approx(before_z, abs=1e-6)


def test_pick_with_recovery_runs_and_reports_recovered_true(env_and_ik):
    env, ik = env_and_ik
    rng = np.random.default_rng(0)
    obj_pos = env.oracle_state()["plate_pos"].copy()
    prefix = assign_arm(obj_pos[0])

    recovered, placed_ok = pick_with_recovery(
        ik, env, prefix, "plate", "plate_region", rng, perturb_after="APPROACH",
    )
    assert recovered is True
    assert isinstance(placed_ok, (bool, np.bool_))


# --------------------------------------------------------------------------
# table_setting.run_full_episode
# --------------------------------------------------------------------------


def test_task_graph_covers_the_manipulated_objects():
    # fork/spoon were descoped out of the manipulation graph (A5,
    # docs/decisions.md) and are pre-placed on the table instead.
    objects = {obj for obj, _region, _needs_drawer in TASK_GRAPH}
    assert objects == {"mug", "bottle", "plate"}


def test_run_full_episode_returns_one_result_per_task_graph_entry_plus_drawer():
    cfg = sample_scene_config("L1", seed=0)
    env = BimanualTableEnv(ResetOptions(scene_cfg=cfg))
    ik = BimanualIK(env.model)

    results = run_full_episode(env, ik)
    steps = [r.step for r in results]
    assert steps[0] == "open_drawer"
    assert set(steps[1:]) == {f"pick_place_{obj}" for obj, _r, _n in TASK_GRAPH}


def test_run_full_episode_still_attempts_every_object_when_the_drawer_fails(env_and_ik, monkeypatch):
    """Cutlery is pre-placed now (A5 descope), so no pick-place goal
    depends on the drawer any more -- a failed drawer must no longer
    block anything downstream, but it must still be reported."""
    env, ik = env_and_ik

    import bimanual.experts.table_setting as table_setting_mod

    monkeypatch.setattr(table_setting_mod, "drawer_open", lambda *_a, **_kw: False)
    results = run_full_episode(env, ik)

    by_step = {r.step: r for r in results}
    assert by_step["open_drawer"].success is False
    for obj, _region, _needs in TASK_GRAPH:
        assert f"pick_place_{obj}" in by_step
        assert "reason" not in by_step[f"pick_place_{obj}"].detail


def test_run_full_episode_records_error_detail_on_planning_exception(env_and_ik, monkeypatch):
    env, ik = env_and_ik

    import bimanual.experts.table_setting as table_setting_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic IK failure")

    monkeypatch.setattr(table_setting_mod, "pick", _boom)
    results = run_full_episode(env, ik)

    by_step = {r.step: r for r in results}
    assert by_step["pick_place_mug"].success is False
    assert "synthetic IK failure" in by_step["pick_place_mug"].detail["error"]


def test_run_full_episode_deterministic_given_same_seed():
    def _run(seed: int):
        cfg = sample_scene_config("L1", seed=seed)
        env = BimanualTableEnv(ResetOptions(scene_cfg=cfg))
        ik = BimanualIK(env.model)
        return run_full_episode(env, ik)

    results_a = _run(3)
    results_b = _run(3)
    assert [(r.step, bool(r.success)) for r in results_a] == [(r.step, bool(r.success)) for r in results_b]
