"""Replay exact 30 Hz expert controls without privileged pose-freezing hooks.

This is a prerequisite diagnostic, not the recorded 10 Hz dataset replay audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bimanual.data.scene_artifacts import scene_artifact_hashes
from bimanual.experts.aloha_pour import run_pour_pose
from bimanual.evaluation.pour_geometry import pour_alignment, pour_sample_valid, pour_succeeded
from bimanual.skills.registry import get_skill


def probe(seed: int, bottle_grasp_height: float = .075) -> dict:
    skill = get_skill("pour_pose")
    env = skill.make_env()
    actions = []
    interference = {}
    try:
        env.reset(seed=seed, randomize_objects=True)
        step = env.step

        def capture(action):
            actions.append(action.copy())
            observation = step(action)
            pairs_this_step = set()
            for contact in env.data.contact:
                bodies = [env.model.body(env.model.geom_bodyid[g]).name for g in contact.geom]
                if not any(body.startswith("left/") for body in bodies):
                    continue
                if not any(body in {"mug", "world"} for body in bodies):
                    continue
                names = [env.model.geom(g).name or body for g, body in zip(contact.geom, bodies)]
                pair = " / ".join(sorted(names))
                item = interference.setdefault(pair, {"control_steps": 0, "max_penetration_m": 0.})
                item["max_penetration_m"] = max(item["max_penetration_m"], max(0., -float(contact.dist)))
                pairs_this_step.add(pair)
            for pair in pairs_this_step:
                interference[pair]["control_steps"] += 1
            return observation

        env.step = capture
        expert = run_pour_pose(env, bottle_grasp_height=bottle_grasp_height)
    finally:
        env.close()

    env = skill.make_env()
    try:
        env.reset(seed=seed, randomize_objects=True)
        dwell = 0
        for index, action in enumerate(actions):
            env.step(action)
            # Match the expert's final 25-control-step dwell window and metric.
            if index >= len(actions) - 25:
                if pour_sample_valid(env):
                    dwell += 1
                else:
                    dwell = 0
        state = env.oracle_state()
        joints = env.state_vector()
        metrics = {
            "mug_height": float(state["mug_pos"][2]),
            "bottle_height": float(state["bottle_pos"][2]),
            **pour_alignment(env),
            "dwell_steps": dwell,
            "mug_opening": float(joints[13]),
            "bottle_opening": float(joints[6]),
            "table_supported": True,
        }
        return {
            "seed": seed,
            "expert": {k: v for k, v in asdict(expert).items() if k != "record"},
            "action_only_success": pour_succeeded(env, dwell),
            "control_steps": len(actions),
            "replay": metrics,
            "alignment": pour_alignment(env),
            "left_arm_interference": interference,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--bottle-grasp-height", type=float, default=.075)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/gates/pour_action_replay_probe.json"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    rows = []
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        "bimanual/experts/aloha_pour.py", "bimanual/control/aloha_ik.py",
        "bimanual/experts/aloha_motion.py", "scripts/aloha_pour_replay_probe.py",
        "bimanual/evaluation/pour_geometry.py",
    )
    source_hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                     for path in source_paths}
    for seed in range(args.seed_offset, args.seed_offset + args.episodes):
        try:
            row = probe(seed, args.bottle_grasp_height)
        except RuntimeError as exc:
            row = {"seed": seed, "expert": {"success": False},
                   "action_only_success": False, "error": str(exc)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    successes = sum(row["expert"]["success"] and row["action_only_success"] for row in rows)
    report = {
        "note": "Exact 30 Hz action replay without pose freezes; not a 10 Hz dataset audit.",
        "episodes": args.episodes,
        "seed_offset": args.seed_offset,
        "bottle_grasp_height": args.bottle_grasp_height,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "threshold": args.threshold,
        "passed": successes / args.episodes >= args.threshold,
        "scene_hashes": scene_artifact_hashes(get_skill("pour_pose").scene_path()),
        "source_expert": get_skill("pour_pose").source_expert,
        "measurement": "ballistic_stream_pose_proxy; no simulated liquid",
        "source_hashes": source_hashes,
        "episodes_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
