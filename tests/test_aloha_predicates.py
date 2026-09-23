import numpy as np

from bimanual.evaluation.aloha_predicates import mug_placed


def test_mug_success_requires_carry_and_released_upright_placement():
    initial = np.array([0.30, 0.12, 0.03])
    final = np.array([0.30, -0.08, 0.029])
    target = np.array([0.30, -0.08, 0.009])
    evidence = dict(
        peak_height=0.16, carried_to_target=True,
        upright_cosine=1.0, gripper_opening=0.037,
    )
    assert mug_placed(initial, final, target, **evidence)
    assert not mug_placed(initial, final, target, **{**evidence, "carried_to_target": False})
    assert not mug_placed(initial, final, target, **{**evidence, "upright_cosine": 0.0})
    assert not mug_placed(initial, final, target, **{**evidence, "gripper_opening": 0.01})
    assert not mug_placed(initial, initial, target, **evidence)
