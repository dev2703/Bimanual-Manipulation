"""Bead-transfer report for the pour scene.

Beads are free spheres. A bead counts as transferred only when it ends inside
the mug's horizontal radius and below the mug rim. This is not a claim that
liquid was poured.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.experts.aloha_motion import CLOSED
from bimanual.experts.aloha_pour import _tilted_quaternion, run_pour_pose
from bimanual.sim.aloha_env import AlohaTableSettingEnv

BEADS = tuple(f"bead_{index}" for index in range(6))


def _place_beads(env) -> None:
    bottle = env.oracle_state()["bottle_pos"]
    for index, name in enumerate(BEADS):
        joint = int(env.model.body(name).jntadr[0])
        address = int(env.model.jnt_qposadr[joint])
        # Start inside the hollow bottle, in two separated layers. The old
        # scene placed free beads above a solid cylinder, so they could only
        # fall onto the table.
        offset = np.array([(-0.006, 0.0, 0.006)[index % 3],
                           (-0.004, 0.004)[index % 2],
                           0.036 + 0.008 * (index // 3)])
        env.data.qpos[address:address + 7] = (* (bottle + offset), 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)


def bead_transfer(env) -> dict:
    mug = env.oracle_state()["mug_pos"]
    in_mug = 0
    spilled = 0
    for name in BEADS:
        pos = env.data.xpos[env.model.body(name).id]
        inside = float(np.linalg.norm(pos[:2] - mug[:2])) < 0.04 and float(pos[2]) < float(mug[2]) + 0.04
        if inside:
            in_mug += 1
        elif float(pos[2]) < 0.05:
            spilled += 1
    return {
        "in_mug": in_mug,
        "spilled": spilled,
        "beads": len(BEADS),
        "fraction_in_mug": in_mug / len(BEADS),
        "final_positions": {
            name: env.data.xpos[env.model.body(name).id].round(6).tolist()
            for name in BEADS
        },
    }


def _empty_bottle(env) -> None:
    """Continue the physical tilt after the geometric pour-pose dwell."""
    ik = AlohaIK(env.model)
    for angle in (1.10, 1.45, 1.80):
        site = env.data.site_xpos[env.model.site("left/gripper").id].copy()
        bottle = env.oracle_state()["bottle_pos"]
        mug = env.oracle_state()["mug_pos"]
        desired_bottle = mug + np.array([-0.06, 0.0, 0.08])
        target = site + desired_bottle - bottle
        result = ik.solve(
            env.data.qpos, {"left": (target, _tilted_quaternion(angle))},
            orientation_tolerance=0.85,
        )
        start = env.data.ctrl.copy()
        for alpha in np.linspace(0.0, 1.0, 60):
            action = start.copy()
            action[:6] = (1.0 - alpha) * start[:6] + alpha * result.joint_targets["left"]
            action[6] = CLOSED
            action[13] = CLOSED
            env.step(action)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--bottle-grasp-height", type=float, default=0.035)
    args = parser.parse_args()
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_pour_beads.xml"
    rows = []
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(scene)
        try:
            env.reset(seed=seed, randomize_objects=True, position_jitter=0.015)
            _place_beads(env)
            pose = run_pour_pose(env, bottle_grasp_height=args.bottle_grasp_height)
            # Give the granular proxy time to leave the neck and settle. Keep
            # applying the final actuator command; no object poses are frozen.
            for _ in range(120):
                env.step(env.data.ctrl.copy())
            beads = bead_transfer(env)
        finally:
            env.close()
        row = {
            "seed": seed,
            "pour_pose": {k: v for k, v in asdict(pose).items() if k != "record"},
            "mug_position": env.oracle_state()["mug_pos"].round(6).tolist(),
            "bottle_position": env.oracle_state()["bottle_pos"].round(6).tolist(),
            "final_bottle_upright_cosine": env.object_upright_cosine("bottle"),
            **beads,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    fraction = float(np.mean([row["fraction_in_mug"] for row in rows]))
    report = {
        "skill": "bead_transfer",
        "episodes": args.episodes,
        "mean_fraction_in_mug": fraction,
        "mean_spilled": float(np.mean([row["spilled"] for row in rows])),
        "note": "bead transfer only; not a liquid pour",
        "episodes_detail": rows,
    }
    out = Path("outputs/gates")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bead_transfer.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("skill", "episodes", "mean_fraction_in_mug", "mean_spilled")}))


if __name__ == "__main__":
    main()
