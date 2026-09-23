import numpy as np
import torch

from bimanual.policy.aloha_act_runner import CONTROL_STEPS_PER_ACTION, run_aloha_act_episode


class _Policy:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def select_action(self, batch):
        self.calls += 1
        assert batch["observation.state"].shape == (1, 14)
        return torch.zeros(1, 14)


class _Env:
    def __init__(self):
        self.steps = 0

    def render(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return {"overhead_cam": frame, "wrist_cam_left": frame, "wrist_cam_right": frame}

    def state_vector(self):
        return np.zeros(14, dtype=np.float32)

    def step(self, action):
        self.steps += 1

    def oracle_state(self):
        height = 0.02 if self.steps < 6 else 0.11
        return {"task_block_pos": np.array([0.0, 0.0, height])}


def test_aloha_runner_holds_each_policy_action_for_three_control_steps():
    env, policy = _Env(), _Policy()
    callbacks = []
    result = run_aloha_act_episode(
        env, policy, device="cpu", retain_steps=2,
        after_control_step=lambda: callbacks.append(env.steps),
    )
    assert CONTROL_STEPS_PER_ACTION == 3
    assert result.success
    assert result.policy_steps == 3
    assert result.control_steps == env.steps == 9
    assert policy.calls == 3
    assert callbacks == list(range(1, 10))
