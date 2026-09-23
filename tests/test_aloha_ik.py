from __future__ import annotations

import numpy as np

from bimanual.control.aloha_ik import AlohaIK, top_down_quaternion
from bimanual.sim.aloha_env import AlohaPhysicalEnv


def test_aloha_ik_reaches_above_block():
    env = AlohaPhysicalEnv()
    ik = AlohaIK(env.model)
    block = env.oracle_state()["task_block_pos"]
    target = block + np.array([-0.01, 0.0, 0.12])
    result = ik.solve(env.data.qpos, {"left": (target, top_down_quaternion())})
    assert result.converged["left"], (result.position_error, result.orientation_error)
    assert result.joint_targets["left"].shape == (6,)


def test_top_down_orientation_has_downward_approach_axis():
    import mujoco

    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, top_down_quaternion())
    matrix = matrix.reshape(3, 3)
    np.testing.assert_allclose(matrix[:, 0], [0, 0, -1], atol=1e-7)
