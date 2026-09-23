"""Tests for bimanual/policy/act_runner.py's pure conversion helpers
(state vector layout, action-to-ctrl mapping) -- no trained checkpoint
needed. The closed-loop rollout itself (run_closed_loop_episode) needs a
real ACTPolicy and is exercised manually via evaluation/eval_act.py, not
here.
"""

from __future__ import annotations

import numpy as np

from bimanual.policy.act_runner import _action_to_ctrl, _bimanual_state_vector
from bimanual.sim import robot_spec as rs
from bimanual.sim.env import BimanualTableEnv


def test_bimanual_state_vector_matches_generate_py_layout():
    from bimanual.experts.generate import _bimanual_vector

    env = BimanualTableEnv()
    proprio = env.proprio()
    arm_qpos = {
        p: np.concatenate([proprio[f"{p}_joint_pos"], [proprio[f"{p}_gripper_opening"]]])
        for p in ("left", "right")
    }
    expected = _bimanual_vector(env, arm_qpos)
    actual = _bimanual_state_vector(env)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_bimanual_state_vector_shape():
    env = BimanualTableEnv()
    vec = _bimanual_state_vector(env)
    assert vec.shape == (rs.N_BIMANUAL_ACTIONS,)


def test_action_to_ctrl_round_trips_through_correct_actuator_indices():
    env = BimanualTableEnv()
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    action = np.arange(rs.N_BIMANUAL_ACTIONS, dtype=np.float64)

    ctrl = _action_to_ctrl(env, action)
    assert ctrl.shape == (env.model.nu,)

    idx = 0
    for prefix in ("left", "right"):
        for suffix in rs.JOINT_SUFFIXES:
            actuator_idx = names.index(rs.joint_name(prefix, suffix))
            assert ctrl[actuator_idx] == action[idx]
            idx += 1
        actuator_idx = names.index(rs.gripper_joint_name(prefix))
        assert ctrl[actuator_idx] == action[idx]
        idx += 1


def test_action_to_ctrl_leaves_drawer_actuator_at_zero():
    env = BimanualTableEnv()
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    action = np.ones(rs.N_BIMANUAL_ACTIONS, dtype=np.float64) * 5.0
    ctrl = _action_to_ctrl(env, action)
    if "drawer_act" in names:
        assert ctrl[names.index("drawer_act")] == 0.0


def test_recorded_action_is_the_command_not_the_measured_state():
    """Regression guard for the bug that made ACT unlearnable: the
    dataset writer recorded `action = observation.state`, i.e. the
    identity function. It trains to a beautiful loss (L1 0.065) and then
    commands "stay exactly where you are" at inference -- measured 0/20
    closed-loop. The commanded target and the measured state MUST differ,
    because the position-controlled arm lags its command.
    """
    from bimanual.control.ik import BimanualIK
    from bimanual.experts.generate import _bimanual_vector, record_pick_place_episode
    from bimanual.sim.env import ResetOptions
    from bimanual.sim.randomization import sample_scene_config

    cfg = sample_scene_config("L1", seed=0)
    env = BimanualTableEnv(ResetOptions(scene_cfg=cfg))
    ik = BimanualIK(env.model)
    record = record_pick_place_episode(env, ik, "right", "plate", "plate_region")
    assert record

    diffs = []
    for tick in record:
        p = tick["proprio"]
        state = _bimanual_vector(
            env,
            {k: np.concatenate([p[f"{k}_joint_pos"], [p[f"{k}_gripper_opening"]]]) for k in ("left", "right")},
        )
        diffs.append(float(np.abs(tick["action"] - state).max()))

    diffs = np.array(diffs)
    assert diffs.mean() > 1e-3, "action is (near-)identical to state -- the identity-function bug is back"
    assert np.mean(diffs > 0.01) > 0.2, "action barely differs from state across the episode"
