"""Full scripted dinner sequence: drawer, plate, cutlery, mug, handoff, pour."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from bimanual.experts.aloha_cutlery import run_fork_place, run_spoon_place
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_handoff import run_baton_handoff
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.experts.aloha_pour import run_pour_pose
from bimanual.sim.aloha_env import AlohaTableSettingEnv
from bimanual.skills.registry import handoff_carrier
from bimanual.data.scene_artifacts import scene_artifact_hashes
from bimanual.evaluation.skill_gate import _git_commit
from bimanual.policy.types import file_sha256

def _place_baton(env) -> None:
    # The handoff expert is written from the neutral arm pose. Later dinner
    # stages leave the wrists elsewhere, which makes the carrier miss.
    neutral = env.model.key("neutral_pose")
    env.data.qpos[:16] = neutral.qpos[:16]
    env.data.ctrl[:] = neutral.ctrl
    joint = env.model.joint("task_block_free")
    address = int(joint.qposadr[0])
    # Stage the baton on the open drawer floor. The table area under the open
    # drawer is not reachable from above.
    env.data.qpos[address:address + 7] = (0.0, 0.14, 0.085, 1.0, 0.0, 0.0, 0.0)
    env.data.qvel[int(env.model.jnt_dofadr[joint.id]):int(env.model.jnt_dofadr[joint.id]) + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(40):
        env.step(env.data.ctrl.copy())


def _handoff(env):
    _place_baton(env)
    return run_baton_handoff(env, carrier=handoff_carrier(int(getattr(env, "gate_seed", 0))))


# The pour scene's glass spot lies inside the open drawer here, and the left arm
# cannot reach the glass's place site. Pour in front of the plate instead.
DINNER_GLASS_TARGET = np.array([0.0, -0.23, 0.033])
DINNER_MUG_ANCHOR = DINNER_GLASS_TARGET + [0.0, 0.0, 0.21]
DINNER_RIGHT_CLEAR = np.array([0.25, -0.15, 0.20])


def _pour(env):
    # Staging, not a learned skill: set the handed-off baton on the drawer
    # floor and return both arms to neutral so the pour does not start from
    # the receiver's raised pose, which sweeps the drawer shut.
    neutral = env.model.key("neutral_pose")
    env.data.qpos[:16] = neutral.qpos[:16]
    env.data.ctrl[:] = neutral.ctrl
    joint = env.model.joint("task_block_free")
    address = int(joint.qposadr[0])
    env.data.qpos[address:address + 7] = (0.0, 0.20, 0.085, 1.0, 0.0, 0.0, 0.0)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(30):
        env.step(env.data.ctrl.copy())
    return run_pour_pose(env, glass_target=DINNER_GLASS_TARGET,
                         mug_anchor=DINNER_MUG_ANCHOR, right_clear=DINNER_RIGHT_CLEAR)


SEQUENCE = (
    ("drawer", run_drawer_open),
    # The spoon's placed bowl overlaps the plate's randomized starting area.
    # Move the plate first so lifting it cannot scoop up the placed spoon.
    ("plate", run_plate_pick_place),
    ("fork", run_fork_place),
    ("spoon", run_spoon_place),
    ("mug", run_mug_pick_place),
    ("handoff", _handoff),
    ("pour", _pour),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/gates/dinner_sequence.json"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_combined_v2.xml"
    success_count = 0
    stage_counts = {name: 0 for name, _ in SEQUENCE}
    rows = []
    for index in range(args.episodes):
        seed = args.seed_offset + index
        env = AlohaTableSettingEnv(scene)
        stages = {}
        diagnostics = {}
        try:
            env.reset(seed=seed, randomize_objects=True)
            env.gate_seed = seed
            for name, expert in SEQUENCE:
                try:
                    result = expert(env)
                    stages[name] = bool(result.success)
                    diagnostics[name] = {"drawer_opening": float(env.oracle_state()["drawer_opening"][0])}
                    state = env.oracle_state()
                    diagnostics[name]["placement_xy_errors"] = {
                        obj: float(np.linalg.norm(state[f"{obj}_pos"][:2]
                                   - env.data.site_xpos[env.model.site(f"{obj}_region").id][:2]))
                        for obj in ("plate", "fork", "spoon")
                    }
                    if name == "pour":
                        diagnostics[name].update({k: v for k, v in asdict(result).items() if k != "record"})
                except RuntimeError as exc:
                    stages[name] = False
                    diagnostics[name] = {"error": str(exc)}
                    break
            state = env.oracle_state()
            plate_site = env.data.site_xpos[env.model.site("plate_region").id]
            fork_site = env.data.site_xpos[env.model.site("fork_region").id]
            spoon_site = env.data.site_xpos[env.model.site("spoon_region").id]
            # Pour moves the glass to the pour spot, so it is not required to stay on its place site.
            retention = {
                "drawer": bool(float(state["drawer_opening"][0]) > 0.11),
                "plate": bool(np.linalg.norm(state["plate_pos"][:2] - plate_site[:2]) < 0.04),
                "fork": bool(np.linalg.norm(state["fork_pos"][:2] - fork_site[:2]) < 0.05),
                "spoon": bool(np.linalg.norm(state["spoon_pos"][:2] - spoon_site[:2]) < 0.05),
            }
            retained = all(retention.values())
        finally:
            env.close()
        for name in stages:
            stage_counts[name] += stages[name]
        success = len(stages) == len(SEQUENCE) and all(stages.values()) and retained
        success_count += success
        row = {"seed": seed, "stages": stages, "retained": retained,
               "retention": retention, "diagnostics": diagnostics, "success": success}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    summary = {
        "skill": "dinner_sequence",
        "episodes": args.episodes,
        "successes": success_count,
        "success_rate": success_count / args.episodes,
        "threshold": 0.70,
        "passed": success_count / args.episodes >= 0.70,
        "seed_offset": args.seed_offset,
        "stage_successes": stage_counts,
        "sequence": [name for name, _ in SEQUENCE],
        "cumulative_stage_successes": {
            name: sum(all(row["stages"].get(previous, False)
                          for previous, _ in SEQUENCE[:index + 1]) for row in rows)
            for index, (name, _) in enumerate(SEQUENCE)
        },
        "retention_successes": {name: sum(row["retention"][name] for row in rows)
                                for name in ("drawer", "plate", "fork", "spoon")},
        "episodes_detail": rows,
        "git_commit": _git_commit(),
        "scene_hashes": scene_artifact_hashes(scene),
        "source_hashes": {str(path.relative_to(scene.parents[3])): file_sha256(path)
                          for path in [Path(__file__).resolve(),
                                       scene.parents[3] / "bimanual/evaluation/pour_geometry.py",
                                       *sorted((scene.parents[3] / "bimanual/experts").glob("aloha_*.py"))]},
        "measurement": {"pour": "ballistic_stream_pose_proxy; no simulated liquid"},
        "scripted_interventions": ["neutral arm resets before handoff and pour",
                                   "baton repositioning before handoff and pour"],
        "standalone_pour_report": "outputs/gates/pour_stream_pose_50.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {"episodes_detail", "scene_hashes", "source_hashes"}}, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("dinner sequence gate is below 70%")


if __name__ == "__main__":
    main()
