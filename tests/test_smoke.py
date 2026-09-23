import mujoco
import numpy as np

from scripts.smoke_render import SMOKE_XML


def test_model_loads_and_steps():
    model = mujoco.MjModel.from_xml_string(SMOKE_XML)
    data = mujoco.MjData(model)
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_same_seed_is_deterministic():
    model = mujoco.MjModel.from_xml_string(SMOKE_XML)

    def rollout():
        data = mujoco.MjData(model)
        for _ in range(200):
            mujoco.mj_step(model, data)
        return data.qpos.copy()

    qpos_a = rollout()
    qpos_b = rollout()
    assert np.array_equal(qpos_a, qpos_b)


def test_offscreen_render_shapes():
    model = mujoco.MjModel.from_xml_string(SMOKE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=256, width=256)
    try:
        for cam in ("global", "left_wrist", "right_wrist"):
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            assert frame.shape == (256, 256, 3)
    finally:
        renderer.close()
