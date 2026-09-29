from pathlib import Path

import mujoco
import numpy as np

from bimanual.sim.aloha_env import AlohaTableSettingEnv
from scripts.aloha_bead_gate import BEADS, _place_beads, bead_transfer


SCENE = Path(__file__).parents[1] / "assets/robots/aloha/task_pour_beads.xml"


def test_beads_settle_on_table_and_collide_with_each_other():
    env = AlohaTableSettingEnv(SCENE)
    try:
        for i, name in enumerate(BEADS):
            joint = env.model.body(name).jntadr[0]
            address = env.model.jnt_qposadr[joint]
            env.data.qpos[address:address + 7] = (0, -.15, .05 + .007*i, 1, 0, 0, 0)
        mujoco.mj_forward(env.model, env.data)
        for _ in range(90):
            env.step(env.data.ctrl.copy())
        positions = np.array([env.data.xpos[env.model.body(n).id].copy() for n in BEADS])
        assert np.all(positions[:, 2] > 0)
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        assert np.min(distances[np.triu_indices(6, 1)]) > .005
    finally:
        env.close()


def test_containment_rejects_beads_below_mug():
    env = AlohaTableSettingEnv(SCENE)
    try:
        _place_beads(env)
        mug = env.oracle_state()["mug_pos"]
        joint = env.model.body(BEADS[0]).jntadr[0]
        address = env.model.jnt_qposadr[joint]
        env.data.qpos[address:address + 3] = mug + [0, 0, -.1]
        mujoco.mj_forward(env.model, env.data)
        assert bead_transfer(env)["in_mug"] == 0
        env.data.qpos[address:address + 3] = mug
        mujoco.mj_forward(env.model, env.data)
        assert bead_transfer(env)["in_mug"] == 1
    finally:
        env.close()
