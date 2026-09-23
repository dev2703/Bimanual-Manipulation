"""Phase 1 gate tests (docs/phases.md):

- same seed -> bit-identical qpos after 200 steps
- every object/region region reachable by at least one arm (reachability
  script's own numeric check is the source of truth; here we just assert
  it reports PASS so a regression fails CI)
- drawer opens fully under direct actuation without interpenetration
- all three cameras render at 256x256 without error
"""

from __future__ import annotations

import numpy as np
import pytest

from bimanual.sim.env import BimanualTableEnv
from bimanual.sim.scene_builder import DRAWER_OPEN_DIST


def test_same_seed_is_deterministic():
    env_a = BimanualTableEnv()
    env_a.reset(seed=42)
    env_b = BimanualTableEnv()
    env_b.reset(seed=42)

    ctrl = np.zeros(env_a.model.nu)
    for _ in range(200):
        env_a.step(ctrl)
        env_b.step(ctrl)

    assert np.array_equal(env_a.data.qpos, env_b.data.qpos)


def test_reachability_study_passes():
    from scripts.reachability_study import check_targets_reachable, sample_reachable_points
    from bimanual.sim.scene_builder import CABINET_POS, REGIONS, TABLE_HEIGHT

    rng = np.random.default_rng(0)
    env = BimanualTableEnv()
    points_by_arm = {
        prefix: sample_reachable_points(env, prefix, 1500, rng) for prefix in ("left", "right")
    }
    targets = {name: (x, y, TABLE_HEIGHT + 0.03) for name, (x, y, _r) in REGIONS.items()}
    targets["drawer_handle"] = (CABINET_POS[0], CABINET_POS[1] - 0.245, CABINET_POS[2] + 0.10)

    report = check_targets_reachable(points_by_arm, targets)
    unreachable = {name: info for name, info in report.items() if not info["reachable"]}
    assert not unreachable, f"unreachable targets: {unreachable}"


def test_drawer_opens_fully_without_interpenetration():
    import mujoco

    env = BimanualTableEnv()
    m, d = env.model, env.data
    ctrl = np.zeros(m.nu)
    names = [m.actuator(i).name for i in range(m.nu)]
    ctrl[names.index("drawer_act")] = -DRAWER_OPEN_DIST

    for _ in range(400):
        env.step(ctrl)

    opening = env.oracle_state()["drawer_opening"]
    assert opening == pytest.approx(-DRAWER_OPEN_DIST, abs=0.01)
    assert np.all(np.isfinite(d.qpos))

    mujoco.mj_forward(m, d)
    deep_penetrations = [c.dist for c in d.contact[: d.ncon] if c.dist < -0.01]
    assert not deep_penetrations, f"deep interpenetration after opening drawer: {deep_penetrations}"


def test_home_pose_has_no_interpenetration():
    import mujoco

    env = BimanualTableEnv()
    mujoco.mj_forward(env.model, env.data)
    deep_penetrations = [
        (
            env.model.body(env.model.geom_bodyid[c.geom1]).name,
            env.model.body(env.model.geom_bodyid[c.geom2]).name,
            round(float(c.dist), 4),
        )
        for c in env.data.contact[: env.data.ncon]
        if c.dist < -0.01
    ]
    assert not deep_penetrations, f"deep interpenetration at home pose: {deep_penetrations}"


def test_cameras_render_at_256():
    env = BimanualTableEnv()
    frames = env.render(width=256, height=256)
    assert set(frames) == {"global", "left_wrist_cam", "right_wrist_cam"}
    for cam, frame in frames.items():
        assert frame.shape == (256, 256, 3), cam
        assert frame.max() > 0, f"{cam} rendered a blank frame"
