"""Phase 3 gate measurement (docs/phases.md): per-skill and composed
success over N seeds at randomization level L1.

Run with: uv run python scripts/expert_success.py
"""

from __future__ import annotations

from collections import defaultdict

from bimanual.control.ik import BimanualIK
from bimanual.experts.table_setting import run_full_episode
from bimanual.sim.env import BimanualTableEnv, ResetOptions
from bimanual.sim.randomization import sample_scene_config

N_SEEDS = 50


def main() -> None:
    per_step_successes: dict[str, int] = defaultdict(int)
    per_step_total: dict[str, int] = defaultdict(int)
    full_success = 0

    for seed in range(N_SEEDS):
        cfg = sample_scene_config("L1", seed=seed)
        env = BimanualTableEnv(ResetOptions(scene_cfg=cfg))
        ik = BimanualIK(env.model)

        results = run_full_episode(env, ik)
        all_ok = True
        for r in results:
            per_step_total[r.step] += 1
            if r.success:
                per_step_successes[r.step] += 1
            else:
                all_ok = False
        if all_ok:
            full_success += 1

    print(f"Over {N_SEEDS} seeds at L1:\n")
    for step in sorted(per_step_total):
        n = per_step_total[step]
        k = per_step_successes[step]
        print(f"  {step:24s} {k:3d}/{n:3d}  ({k/n:.1%})")
    print(f"\n  {'FULL EPISODE':24s} {full_success:3d}/{N_SEEDS:3d}  ({full_success/N_SEEDS:.1%})")


if __name__ == "__main__":
    main()
