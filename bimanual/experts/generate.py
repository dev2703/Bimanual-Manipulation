"""Phase 4 dataset writer: runs a scripted skill and writes a
LeRobotDataset at 10fps with the schema decision A2/A7 call for.

Given the Phase 3 measurement (docs/decisions.md), only the plate skill
clears a usable success rate (84% at L1); mug/bottle/cutlery need more
IK/grasp-offset tuning than this session's remaining budget allows. This
generator defaults to the pick_place_plate bucket so the Phase 4 pipeline
(schema, contract test, ACT baseline) can be built and tested end to end
on a real, working skill, matching training/configs/act_pick_plate.yaml's
name. Other skills can be passed via --skill once their success rate
improves.

Feature schema:
  observation.images.{global,left_wrist,right_wrist}: (256,256,3) uint8
  observation.state: 12-D proprioception (5 joints + gripper, both arms).
    NOT privileged (decision A2, tier "proprioception").
  action: 12-D joint targets (both arms), matching observation.state's
    layout, decision A1's joint-space baseline.
  task: serialized instruction string (decision A7).
  privileged.object_poses: (5, 3) object xyz -- oracle-only.
  privileged.phase: string phase label (Long-VLA-style, plan.md).
  privileged.progress: float in [0, 1], fraction through the episode.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bimanual.control.ik import BimanualIK
from bimanual.experts.executor import execute
from bimanual.experts.skills import pick, place, place_target
from bimanual.logging_utils import get_logger
from bimanual.sim import robot_spec as rs
from bimanual.sim.env import BimanualTableEnv, ResetOptions
from bimanual.sim.oracle_predicates import in_region
from bimanual.sim.randomization import LEVELS, sample_scene_config
from bimanual.sim.scene_builder import REGIONS
from bimanual.policy.types import EpisodeManifest, SO101_BIMANUAL

log = get_logger(__name__)

OBJECT_ORDER = ("plate", "mug", "bottle", "fork", "spoon")
IMAGE_KEYS = {"global": "observation.images.global", "left_wrist_cam": "observation.images.left_wrist", "right_wrist_cam": "observation.images.right_wrist"}


def make_features() -> dict:
    return {
        "observation.images.global": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.images.left_wrist": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.images.right_wrist": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (rs.N_BIMANUAL_ACTIONS,), "names": None},
        "action": {"dtype": "float32", "shape": (rs.N_BIMANUAL_ACTIONS,), "names": None},
        "privileged.object_poses": {"dtype": "float32", "shape": (len(OBJECT_ORDER), 3), "names": None},
        "privileged.progress": {"dtype": "float32", "shape": (1,), "names": None},
        "privileged.phase": {"dtype": "string", "shape": (1,), "names": None},
        "privileged.drawer_opening": {"dtype": "float32", "shape": (1,), "names": None},
        "privileged.drawer_handle_y": {"dtype": "float32", "shape": (1,), "names": None},
        "privileged.sim_time": {"dtype": "float32", "shape": (1,), "names": None},
        "privileged.scene_seed": {"dtype": "int64", "shape": (1,), "names": None},
        "privileged.randomization_level": {"dtype": "string", "shape": (1,), "names": None},
    }


def _bimanual_vector(env: BimanualTableEnv, arm_qpos: dict[str, np.ndarray]) -> np.ndarray:
    """12-D: left 5 joints + gripper, then right 5 joints + gripper."""
    out = []
    for prefix in ("left", "right"):
        out.extend(arm_qpos[prefix][:5])
        out.append(arm_qpos[prefix][5])
    return np.asarray(out, dtype=np.float32)


def record_pick_place_episode(env: BimanualTableEnv, ik: BimanualIK, prefix: str, obj: str, region: str) -> list[dict]:
    """Runs pick()+place() for one object and returns a list of raw
    per-tick records (env.proprio()/oracle_state()/phase/frames), which
    write_episode() below converts into the LeRobotDataset schema."""
    obj_pos = env.oracle_state()[f"{obj}_pos"].copy()
    ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
    pick_segs = pick(ik, env.data.qpos, prefix, ee_pos, obj_pos, obj)

    rx, ry, _ = REGIONS[region]
    target = place_target(obj, (rx, ry))

    record: list[dict] = []
    execute(env, prefix, pick_segs, obj_name=obj, record=record)
    ee2 = env.proprio()[f"{prefix}_ee_pos"].copy()
    place_segs = place(ik, env.data.qpos, prefix, ee2, target, obj)
    execute(env, prefix, place_segs, obj_name=obj, record=record)
    return record


def write_episode(
    dataset,
    env: BimanualTableEnv,
    record: list[dict],
    task_str: str,
    scene_seed: int = 0,
    randomization_level: str = "L0",
) -> None:
    n = len(record)
    for i, tick in enumerate(record):
        frames = tick["frames"]
        proprio = tick["proprio"]
        oracle = tick["oracle_state"]
        arm_qpos = {p: np.concatenate([proprio[f"{p}_joint_pos"], [proprio[f"{p}_gripper_opening"]]]) for p in ("left", "right")}
        state = _bimanual_vector(env, arm_qpos)
        object_poses = np.stack([oracle[f"{o}_pos"] for o in OBJECT_ORDER]).astype(np.float32)

        frame = {
            "observation.images.global": frames["global"],
            "observation.images.left_wrist": frames["left_wrist_cam"],
            "observation.images.right_wrist": frames["right_wrist_cam"],
            "observation.state": state,
            # The COMMANDED target for this tick, not the achieved state --
            # see executor.commanded_action_vector(). These differ because
            # the position-controlled arm lags its command, and training on
            # action==state teaches the identity function (fits beautifully,
            # then refuses to move at inference).
            "action": tick["action"],
            "privileged.object_poses": object_poses,
            "privileged.progress": np.array([i / max(n - 1, 1)], dtype=np.float32),
            "privileged.phase": tick["phase"],
            "privileged.drawer_opening": np.array([oracle["drawer_opening"]], dtype=np.float32),
            "privileged.drawer_handle_y": np.array([oracle["drawer_handle_pos"][1]], dtype=np.float32),
            "privileged.sim_time": np.array([tick["timestamp"]], dtype=np.float32),
            "privileged.scene_seed": np.array([scene_seed], dtype=np.int64),
            "privileged.randomization_level": randomization_level,
            "task": task_str,
        }
        dataset.add_frame(frame)
    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", default="plate", choices=["plate", "mug", "bottle"])
    parser.add_argument("--episodes", type=int, default=20, help="number of SUCCESSFUL episodes to collect")
    parser.add_argument(
        "--max-attempts", type=int, default=0,
        help="cap on total (success+failure) attempts; 0 = 15x --episodes, sized for the current ~18%% plate success rate",
    )
    parser.add_argument(
        "--only-successes", action=argparse.BooleanOptionalAction, default=True,
        help="only write episodes that actually reach the target region (default: on -- a scripted skill with "
             "~18%% raw success rate would otherwise produce a dataset that's ~80%% failed demonstrations)",
    )
    parser.add_argument("--root", default="outputs/dataset_pick_place")
    parser.add_argument("--repo-id", default="local/bimanual-pick-place")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset root")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"dataset root already exists: {root}; pass --overwrite to replace it")
        import shutil
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=10,
        features=make_features(),
        root=str(root),
        robot_type="bimanual_so101",
        use_videos=True,
    )

    region_by_obj = {"plate": "plate_region", "mug": "mug_region", "bottle": "bottle_region"}
    # "plate": "right" matches experts/table_setting.py's FIXED_ARM_OVERRIDE
    # -- plate_region was grid-searched for the right arm specifically
    # (see scene_builder.py's REGIONS comment); using "left" here would
    # silently try to reach it from the wrong side.
    arm_by_obj = {"plate": "right", "mug": "right", "bottle": "left"}
    obj = args.skill
    region = region_by_obj[obj]
    prefix = arm_by_obj[obj]

    max_attempts = args.max_attempts or args.episodes * 15

    t0 = time.time()
    n_frames = 0
    n_success = 0
    n_attempts = 0
    split_seed_offset = {"train": 0, "val": 100_000, "test": 200_000}[args.split]
    manifests: list[dict] = []
    while n_success < args.episodes and n_attempts < max_attempts:
        # Cycle L0/L1/L2 for scenario diversity (nominal, geometric
        # jitter, geometric+visual jitter) rather than a single fixed
        # randomization level -- each successful episode's seed and
        # level are logged so a specific scenario can be replayed later.
        level = LEVELS[n_attempts % len(LEVELS)]
        scene_seed = split_seed_offset + n_attempts
        cfg = sample_scene_config(level, seed=scene_seed)
        env = BimanualTableEnv(ResetOptions(scene_cfg=cfg))
        ik = BimanualIK(env.model)
        record = record_pick_place_episode(env, ik, prefix, obj, region)
        n_attempts += 1

        ok = in_region(env.oracle_state(), obj, region)
        if args.only_successes and not ok:
            log.info("attempt %d (level=%s seed=%d): FAILED, discarding", n_attempts, level, scene_seed)
            continue

        task_str = f"Task: set the table. Now: place {obj} in its region with {prefix} arm."
        write_episode(dataset, env, record, task_str, scene_seed, level)
        manifests.append(
            EpisodeManifest(
                environment="bimanual_table_legacy",
                embodiment=SO101_BIMANUAL.name,
                model_revision=SO101_BIMANUAL.model_revision,
                seed=scene_seed,
                split=args.split,
                source_expert="mink_waypoint_v1",
            ).to_dict()
        )
        n_frames += len(record)
        n_success += 1
        log.info(
            "episode %d/%d (level=%s seed=%d): %d frames, %s",
            n_success, args.episodes, level, scene_seed, len(record), "OK" if ok else "recorded anyway",
        )

    dt = time.time() - t0
    log.info(
        "wrote %d/%d requested episodes from %d attempts (%.1f%% yield), %d frames in %.1fs (%.1f frames/s)",
        n_success, args.episodes, n_attempts, 100 * n_success / max(n_attempts, 1), n_frames, dt, n_frames / dt,
    )
    if n_success < args.episodes:
        log.warning(
            "hit max_attempts (%d) before collecting %d successes -- only got %d. "
            "Raise --max-attempts or fix the underlying skill success rate.",
            max_attempts, args.episodes, n_success,
        )
    manifest_path = root / "episode_manifests.json"
    manifest_path.write_text(json.dumps(manifests, indent=2) + "\n")


if __name__ == "__main__":
    main()
