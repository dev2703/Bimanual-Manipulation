"""Bead-transfer report for the pour scene.

Beads are free spheres. A bead counts as transferred only when it ends inside
the mug's horizontal radius and below the mug rim. This is not a claim that
liquid was poured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from bimanual.experts.aloha_pour import run_pour_pose
from bimanual.sim.aloha_env import AlohaTableSettingEnv

BEADS = tuple(f"bead_{index}" for index in range(6))


def _place_beads(env) -> None:
    bottle = env.oracle_state()["bottle_pos"]
    for index, name in enumerate(BEADS):
        joint = int(env.model.body(name).jntadr[0])
        address = int(env.model.jnt_qposadr[joint])
        offset = np.array([(index - 2.5) * 0.012, 0.0, 0.08 + (index % 2) * 0.012])
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
    return {"in_mug": in_mug, "spilled": spilled, "beads": len(BEADS), "fraction_in_mug": in_mug / len(BEADS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    args = parser.parse_args()
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_pour_beads.xml"
    rows = []
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(scene)
        try:
            env.reset(seed=seed, randomize_objects=True, position_jitter=0.015)
            _place_beads(env)
            pose = run_pour_pose(env)
            beads = bead_transfer(env)
        finally:
            env.close()
        row = {"seed": seed, "pour_pose": bool(pose.success), **beads}
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
