"""Held-out scripted-expert gate driven by the skill registry."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from bimanual.data.scene_artifacts import scene_artifact_hashes
from bimanual.skills.registry import Skill


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def reset_for_skill(skill: Skill, env, seed: int, jitter: float, visuals: bool = False) -> None:
    """Reset to the seeded start of `skill`, including any declared precondition.

    `visuals` adds Level 2 colour and lighting randomization for the seed.
    """
    if skill.randomize == "block":
        env.reset(seed=seed, randomize_block=True)
    else:
        env.reset(seed=seed, randomize_objects=True, position_jitter=jitter)
    env.set_visuals(seed if visuals else None)
    env.gate_seed = seed
    if skill.setup is not None:
        skill.setup(env)


def run_gate(skill: Skill, episodes: int, seed_offset: int, jitter: float, threshold: float) -> dict:
    rows = []
    for index in range(episodes):
        seed = seed_offset + index
        env = skill.make_env()
        try:
            reset_for_skill(skill, env, seed, jitter)
            result = skill.run(env)
            success = bool(result.success)
        except Exception as exc:
            success = False
            rows.append({"seed": seed, "success": False, "error": str(exc)})
            print(json.dumps(rows[-1]), flush=True)
            continue
        finally:
            env.close()
        row = {"seed": seed, "success": success}
        carrier = getattr(result, "carrier", None)
        if carrier is not None:
            row["carrier"] = carrier
        rows.append(row)
        print(json.dumps(row), flush=True)
    successes = sum(row["success"] for row in rows)
    report = {
        "skill": skill.name,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "threshold": threshold,
        "passed": successes / episodes >= threshold,
        "seed_offset": seed_offset,
        "jitter": jitter,
        "git_commit": _git_commit(),
        "scene_hashes": scene_artifact_hashes(skill.scene_path()),
        "episodes_detail": rows,
    }
    out = Path("outputs/gates")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{skill.name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("skill", "episodes", "successes", "success_rate", "passed")}))
    if not report["passed"]:
        raise SystemExit(1)
    return report
