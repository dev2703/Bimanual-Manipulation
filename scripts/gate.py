"""Run one scripted-expert gate: python scripts/gate.py --skill mug_pick_place."""

from __future__ import annotations

import argparse

from bimanual.evaluation.skill_gate import run_gate
from bimanual.skills.registry import get_skill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--jitter", type=float, default=0.015)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    run_gate(get_skill(args.skill), args.episodes, args.seed_offset, args.jitter, args.threshold)


if __name__ == "__main__":
    main()
