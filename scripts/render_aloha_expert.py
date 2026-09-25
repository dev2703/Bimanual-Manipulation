"""Render inspectable three-camera videos of the scripted ALOHA experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from bimanual.experts.aloha_bottle import run_bottle_grasp_lift
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_handoff import run_baton_handoff
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.sim.aloha_env import AlohaPhysicalEnv, AlohaTableSettingEnv


def _montage(frames: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [frames["overhead_cam"], frames["wrist_cam_left"], frames["wrist_cam_right"]], axis=1,
    )


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=10, codec="libx264", quality=8) as writer:
        for tick in records:
            writer.append_data(_montage(tick["frames"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("combined", "handoff"), default="combined")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"outputs/visualizations/aloha_{args.task}_{args.seed}.mp4")

    records: list[dict] = []
    stages = {}
    if args.task == "combined":
        scene = Path("assets/robots/aloha/task_table_setting_combined_v2.xml")
        env = AlohaTableSettingEnv(scene)
        experts = (
            ("drawer", run_drawer_open),
            ("plate", run_plate_pick_place),
            ("mug", run_mug_pick_place),
            ("bottle_lift", run_bottle_grasp_lift),
        )
        try:
            env.reset(seed=args.seed, randomize_objects=True)
            for name, expert in experts:
                result = expert(env, record_frames=True)
                records.extend(result.record)
                stages[name] = bool(result.success)
        finally:
            env.close()
    else:
        scene = Path("assets/robots/aloha/task_handoff_baton.xml")
        env = AlohaPhysicalEnv(scene)
        try:
            env.reset(seed=args.seed, randomize_block=True)
            result = run_baton_handoff(env, record_frames=True)
            records.extend(result.record)
            stages["handoff"] = bool(result.success)
        finally:
            env.close()

    _write(output, records)
    print(json.dumps({"output": str(output), "frames": len(records), "stages": stages}, sort_keys=True))


if __name__ == "__main__":
    main()
