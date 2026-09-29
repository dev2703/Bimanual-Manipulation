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
from bimanual.experts.aloha_pour import _mouth, run_pour_pose
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
                           0.032])
        env.data.qpos[address:address + 7] = (* (bottle + offset), 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)


def bead_transfer(env) -> dict:
    mug_body = int(env.model.body("mug").id)
    mug = env.data.xpos[mug_body].copy()
    world_to_mug = env.data.xmat[mug_body].reshape(3, 3).T
    in_mug = 0
    spilled = 0
    for name in BEADS:
        pos = env.data.xpos[env.model.body(name).id]
        local = world_to_mug @ (pos - mug)
        # The collision cage has a 28 mm wall radius, a bottom at -30 mm,
        # and a rim at +30 mm once the capsule radii are included.
        inside = float(np.linalg.norm(local[:2])) < 0.021 and -0.021 < float(local[2]) < 0.021
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
    """Shake the tipped container with wrist commands so beads clear the rim."""
    action = env.data.ctrl.copy()
    center = float(action[4])
    for offset in (-0.12, 0.03, -0.12, 0.03, -0.12):
        start = float(action[4])
        goal = float(np.clip(center + offset, -1.6, 2.0))
        for value in np.linspace(start, goal, 20):
            action[4] = value
            action[6] = CLOSED
            action[13] = CLOSED
            env.step(action)


def _carry_bottle_relaxed(env, ik: AlohaIK, desired: np.ndarray) -> None:
    """Translate the tipped bottle while preserving its achieved orientation."""
    bottle = env.oracle_state()["bottle_pos"]
    site_id = env.model.site("left/gripper").id
    site = env.data.site_xpos[site_id].copy()
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(
        quaternion, env.data.site_xmat[site_id].reshape(3, 3).ravel().copy()
    )
    result = ik.solve(
        env.data.qpos,
        {"left": (site + np.asarray(desired) - bottle, quaternion)},
        orientation_tolerance=0.35,
    )
    start = env.data.ctrl.copy()
    for alpha in np.linspace(0.0, 1.0, 50):
        action = start.copy()
        action[:6] = ((1.0 - alpha) * start[:6]
                      + alpha * result.joint_targets["left"])
        action[6] = CLOSED
        action[13] = CLOSED
        env.step(action)


def _align_bottle_over_mug(env) -> None:
    """Put the tilted bottle's physical outlet above the held mug opening."""
    ik = AlohaIK(env.model)
    for _ in range(3):
        _unused, bottle_up = _mouth(env)
        mug_body = int(env.model.body("mug").id)
        mug = env.data.xpos[mug_body].copy()
        mug_up = env.data.xmat[mug_body].reshape(3, 3)[:, 2]
        desired_outlet = mug + mug_up * 0.040
        desired_bottle = desired_outlet - bottle_up * 0.030
        _carry_bottle_relaxed(env, ik, desired_bottle)


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
            pose = run_pour_pose(
                env,
                bottle_grasp_height=args.bottle_grasp_height,
                mug_anchor=np.array([-0.08, 0.12, 0.16]),
                tilt_delta=-1.05,
                bottle_anchor=np.array([0.02, 0.08, 0.18]),
            )
            pre_align = {
                "mug": env.oracle_state()["mug_pos"].round(6).tolist(),
                "bottle": env.oracle_state()["bottle_pos"].round(6).tolist(),
            }
            # Keep the stable two-arm pose rather than crossing either arm
            # through the other gripper after the pour pose is reached.
            post_align = {
                "mug": env.oracle_state()["mug_pos"].round(6).tolist(),
                "bottle": env.oracle_state()["bottle_pos"].round(6).tolist(),
            }
            _empty_bottle(env)
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
            "pre_align": pre_align,
            "post_align": post_align,
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
