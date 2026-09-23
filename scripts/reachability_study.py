"""Phase 1 reachability study.

Samples joint configurations for one arm, records the end-effector site's
world position via forward kinematics, and checks two things:

1. Every object region and the drawer handle lies within at least one
   arm's sampled reachable point cloud (within a tolerance), so the scene
   layout in scene_builder.py is actually usable.
2. Reports a home-pose candidate (already wired into env.py's HOME_POSE)
   that clears the table by a healthy margin.

This is a coarse, sampling-based check -- not the IK-based reachability
test that lands in Phase 2 (tests/test_ik.py). Its job is to catch a badly
laid-out scene now, before any control code is built on top of it.
"""

from __future__ import annotations

import numpy as np

from bimanual.sim import robot_spec as rs
from bimanual.sim.env import BimanualTableEnv
from bimanual.sim.scene_builder import CABINET_POS, REGIONS, TABLE_HEIGHT

N_SAMPLES = 4000
TOLERANCE = 0.05  # meters


def sample_reachable_points(env: BimanualTableEnv, prefix: str, n: int, rng: np.random.Generator) -> np.ndarray:
    assert env.model is not None and env.data is not None
    points = np.zeros((n, 3))
    joint_names = rs.arm_joint_names(prefix)
    limits = [rs.JOINT_LIMITS[s] for s in rs.JOINT_SUFFIXES]
    site_id = env.model.site(rs.ee_site_name(prefix)).id
    for i in range(n):
        for name, lim in zip(joint_names, limits):
            angle = rng.uniform(lim.lo, lim.hi)
            env._set_joint_qpos(name, angle)
        import mujoco

        mujoco.mj_forward(env.model, env.data)
        points[i] = env.data.site_xpos[site_id]
    return points


def check_targets_reachable(points_by_arm: dict[str, np.ndarray], targets: dict[str, tuple[float, float, float]]) -> dict[str, dict]:
    report = {}
    for name, (tx, ty, tz) in targets.items():
        best = None
        for arm, pts in points_by_arm.items():
            dist = np.min(np.linalg.norm(pts - np.array([tx, ty, tz]), axis=1))
            if best is None or dist < best[1]:
                best = (arm, dist)
        report[name] = {"nearest_arm": best[0], "min_dist": float(best[1]), "reachable": best[1] <= TOLERANCE}
    return report


def main() -> None:
    rng = np.random.default_rng(0)
    env = BimanualTableEnv()

    points_by_arm = {}
    for prefix in ("left", "right"):
        points_by_arm[prefix] = sample_reachable_points(env, prefix, N_SAMPLES, rng)
        pc = points_by_arm[prefix]
        print(
            f"{prefix} arm reachable point cloud: "
            f"x[{pc[:,0].min():.3f},{pc[:,0].max():.3f}] "
            f"y[{pc[:,1].min():.3f},{pc[:,1].max():.3f}] "
            f"z[{pc[:,2].min():.3f},{pc[:,2].max():.3f}]"
        )

    targets = {name: (x, y, TABLE_HEIGHT + 0.03) for name, (x, y, _r) in REGIONS.items()}
    targets["drawer_handle"] = (CABINET_POS[0], CABINET_POS[1] - 0.245, CABINET_POS[2] + 0.15 - 0.05)

    report = check_targets_reachable(points_by_arm, targets)
    print("\nTarget reachability (tolerance {:.0f} mm):".format(TOLERANCE * 1000))
    all_ok = True
    for name, info in report.items():
        status = "OK" if info["reachable"] else "FAIL"
        if not info["reachable"]:
            all_ok = False
        print(f"  {name:24s} nearest={info['nearest_arm']:5s} dist={info['min_dist']*1000:6.1f}mm  {status}")

    handoff_pts_left = points_by_arm["left"]
    handoff_pts_right = points_by_arm["right"]
    handoff_region = REGIONS["handoff_region"]
    hx, hy, _ = handoff_region
    d_left = np.min(np.linalg.norm(handoff_pts_left - np.array([hx, hy, TABLE_HEIGHT + 0.03]), axis=1))
    d_right = np.min(np.linalg.norm(handoff_pts_right - np.array([hx, hy, TABLE_HEIGHT + 0.03]), axis=1))
    both_reach_handoff = d_left <= TOLERANCE and d_right <= TOLERANCE
    print(f"\nhandoff_region reachable by BOTH arms: {both_reach_handoff} (left={d_left*1000:.1f}mm, right={d_right*1000:.1f}mm)")

    print("\nOverall:", "PASS" if all_ok and both_reach_handoff else "FAIL -- adjust scene_builder.py layout or robot_spec limits")


if __name__ == "__main__":
    main()
