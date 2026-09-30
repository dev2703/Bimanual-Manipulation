"""Static empty-glass pose preview. This is not a physical expert rollout.

Run with --render to write an image; without it, validate geometry only.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from bimanual.control.aloha_ik import AlohaIK, ARM_JOINTS, top_down_quaternion
from bimanual.experts.aloha_pour import (
    MOUTH_OFFSET, MUG_RIM_OFFSET, POUR_GAP, outlet_aligned, pour_alignment,
)
from bimanual.sim.aloha_env import AlohaTableSettingEnv

ASSETS = Path(__file__).resolve().parents[1] / "assets/robots/aloha"


def preview_env():
    root = ET.parse(ASSETS / "task_table_setting.xml").getroot()
    mug = root.find(".//body[@name='mug']")
    solid = mug.find("geom[@name='mug_body']")
    mug.remove(solid)
    # An open vessel for the preview; the expert's contact scene is unchanged.
    ET.SubElement(mug, "geom", name="mug_body", type="cylinder",
                  pos="0 0 -.028", size=".032 .002", mass=".024",
                  rgba=".65 .85 .95 1")
    for i in range(48):
        angle = i * 2 * np.pi / 48
        x, y = .030 * np.cos(angle), .030 * np.sin(angle)
        ET.SubElement(mug, "geom", type="capsule", size=".002",
                      fromto=f"{x} {y} -.028 {x} {y} .028",
                      mass=".0005", rgba=".65 .85 .95 1")
    with tempfile.NamedTemporaryFile(suffix=".xml", dir=ASSETS) as scene:
        scene.write(ET.tostring(root))
        scene.flush()
        return AlohaTableSettingEnv(scene.name)


def _set_object(env, name, position, quaternion):
    joint = env.model.body(name).jntadr[0]
    address = env.model.jnt_qposadr[joint]
    env.data.qpos[address:address + 7] = (*position, *quaternion)


def set_preview_pose(env):
    """Set a static target pose; no physics steps or training records are made."""
    angle = np.deg2rad(65)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    mug = np.array([.06, .10, .18])
    mouth = mug + [0, 0, MUG_RIM_OFFSET + POUR_GAP]
    bottle = mouth - rotation[:, 2] * MOUTH_OFFSET
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    _set_object(env, "mug", mug, [1, 0, 0, 0])
    _set_object(env, "bottle", bottle, quaternion)
    mujoco.mj_forward(env.model, env.data)
    top = np.empty(9)
    mujoco.mju_quat2Mat(top, top_down_quaternion())
    left_quat = np.empty(4)
    mujoco.mju_mat2Quat(left_quat, (rotation @ top.reshape(3, 3).T).ravel())
    ik = AlohaIK(env.model)
    result = ik.solve(env.data.qpos, {
        "left": (bottle + rotation[:, 2] * .025, left_quat),
        "right": (mug + [.045, 0, .025], top_down_quaternion()),
    })
    for arm in ("left", "right"):
        for joint, value in zip(ARM_JOINTS, result.joint_targets[arm]):
            env.data.qpos[env.model.joint(f"{arm}/{joint}").qposadr[0]] = value
        for finger in ("left_finger", "right_finger"):
            env.data.qpos[env.model.joint(f"{arm}/{finger}").qposadr[0]] = .024 if arm == "left" else .008
    mujoco.mj_forward(env.model, env.data)
    metrics = pour_alignment(env)
    return {
        "mode": "static_pose_preview",
        "physical_rollout": False,
        "bead_transfer": "deferred",
        "target_gap_m": POUR_GAP,
        "alignment_passed": outlet_aligned(metrics),
        **metrics,
        "arm_ik_converged": result.converged,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--view", action="store_true",
                        help="Open the static MuJoCo viewer (use mjpython on macOS).")
    parser.add_argument("--output", type=Path, default=Path("outputs/pour_pose_preview"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = preview_env()
    try:
        report = set_preview_pose(env)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.025, .10, .19]
        camera.distance = .65
        camera.azimuth = 270
        camera.elevation = -30
        if args.render:
            from PIL import Image
            env.model.vis.global_.offwidth = 960
            env.model.vis.global_.offheight = 720
            with mujoco.Renderer(env.model, height=720, width=960) as renderer:
                renderer.update_scene(env.data, camera=camera)
                Image.fromarray(renderer.render()).save(args.output / "pose.png")
        (args.output / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        if args.view:
            import mujoco.viewer as viewer
            with viewer.launch_passive(env.model, env.data) as window:
                window.cam.lookat[:] = camera.lookat
                window.cam.distance = camera.distance
                window.cam.azimuth = camera.azimuth
                window.cam.elevation = camera.elevation
                while window.is_running():
                    window.sync()
                    time.sleep(1 / 30)
        return 0 if report["alignment_passed"] and all(report["arm_ik_converged"].values()) else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
