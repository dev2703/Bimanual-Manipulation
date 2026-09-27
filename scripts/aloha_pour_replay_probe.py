"""Replay exact 30 Hz expert controls without privileged pose-freezing hooks.

This is a prerequisite diagnostic, not the recorded 10 Hz dataset replay audit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bimanual.data.scene_artifacts import scene_artifact_hashes
from bimanual.experts.aloha_pour import _mouth, pour_pose_reached, run_pour_pose
from bimanual.skills.registry import get_skill


def probe(seed: int) -> dict:
    skill = get_skill("pour_pose")
    env = skill.make_env()
    actions = []
    try:
        env.reset(seed=seed, randomize_objects=True)
        step = env.step

        def capture(action):
            actions.append(action.copy())
            return step(action)

        env.step = capture
        expert = run_pour_pose(env)
    finally:
        env.close()

    env = skill.make_env()
    try:
        env.reset(seed=seed, randomize_objects=True)
        dwell = 0
        best_tilt = best_error = 1.0
        for index, action in enumerate(actions):
            env.step(action)
            # Match the expert's final 25-control-step dwell window and metric.
            if index >= len(actions) - 25:
                mouth, up = _mouth(env)
                mug = env.oracle_state()["mug_pos"]
                error = float(np.linalg.norm(mouth[:2] - mug[:2]))
                best_tilt = min(best_tilt, float(up[2]))
                best_error = min(best_error, error)
                if up[2] < 0.88 and error < 0.10 and mug[2] > 0.08:
                    dwell += 1
        state = env.oracle_state()
        joints = env.state_vector()
        metrics = {
            "mug_height": float(state["mug_pos"][2]),
            "bottle_height": float(state["bottle_pos"][2]),
            "tilt_cosine": best_tilt,
            "mouth_xy_error": best_error,
            "dwell_steps": dwell,
            "mug_opening": float(joints[13]),
            "bottle_opening": float(joints[6]),
        }
        return {
            "seed": seed,
            "expert": {k: v for k, v in asdict(expert).items() if k != "record"},
            "action_only_success": pour_pose_reached(**metrics),
            "control_steps": len(actions),
            "replay": metrics,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/gates/pour_action_replay_probe.json"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    rows = []
    for seed in range(args.seed_offset, args.seed_offset + args.episodes):
        row = probe(seed)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "note": "Exact 30 Hz action replay without pose freezes; not a 10 Hz dataset audit.",
        "episodes": args.episodes,
        "seed_offset": args.seed_offset,
        "successes": sum(row["action_only_success"] for row in rows),
        "passed": all(row["expert"]["success"] and row["action_only_success"] for row in rows),
        "scene_hashes": scene_artifact_hashes(get_skill("pour_pose").scene_path()),
        "episodes_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
