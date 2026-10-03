import mujoco
import numpy as np

from bimanual.evaluation.aloha_predicates import handoff_succeeded
from bimanual.experts.aloha_handoff import grip_quaternion


def test_handoff_gate_requires_receiver_support_after_carrier_release():
    assert handoff_succeeded(0.02, 0.13, 0.10, 0.11, 0.037, 0.008, 30)
    assert not handoff_succeeded(0.02, 0.13, 0.02, 0.02, 0.037, 0.008, 30)
    assert not handoff_succeeded(0.02, 0.13, 0.10, 0.11, 0.037, 0.008, 0)


def _axes(arm: str) -> np.ndarray:
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, grip_quaternion(arm))
    return rotation.reshape(3, 3)


def test_gripper_housings_point_outboard_while_both_pinch_across_the_baton():
    left, right = _axes("left"), _axes("right")
    np.testing.assert_allclose(left[:, 0], [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(right[:, 0], [0, 0, -1], atol=1e-9)
    assert abs(left[1, 1]) == 1.0 and abs(right[1, 1]) == 1.0
    assert left[0, 2] < 0 < right[0, 2]
