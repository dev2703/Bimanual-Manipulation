"""Run one scripted-expert gate and write outputs/gates/<skill>.json.

    python scripts/gate.py --list
    python scripts/gate.py --skill mug_pick_place --episodes 50
"""

from __future__ import annotations

import argparse

from bimanual.evaluation.skill_gate import run_gate
from bimanual.skills.registry import SKILLS, get_skill


def _print_skills() -> None:
    for name, skill in sorted(SKILLS.items()):
        status = "passed" if skill.gate_passed else "open"
        print(f"{name:20s} gate={status:6s} arms={','.join(skill.arms):10s} {skill.instruction}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skill", choices=sorted(SKILLS))
    parser.add_argument("--list", action="store_true", help="list registered skills and exit")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--jitter", type=float, default=0.015)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    if args.list:
        _print_skills()
        return
    if args.skill is None:
        parser.error("--skill is required (see --list)")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    run_gate(get_skill(args.skill), args.episodes, args.seed_offset, args.jitter, args.threshold)


if __name__ == "__main__":
    main()
