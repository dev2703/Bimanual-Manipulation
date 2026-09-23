from __future__ import annotations

from bimanual.experts.aloha_block import run_block_grasp_lift
from bimanual.sim.aloha_env import AlohaPhysicalEnv


def test_aloha_block_expert_lifts_and_retains_nominal_block_without_attachment():
    env = AlohaPhysicalEnv()
    result = run_block_grasp_lift(env)
    assert result.success, result
    assert not hasattr(env, "grasp")
