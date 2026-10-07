"""Render the dinner scene with its authored robot meshes.

Examples:
    python scripts/preview_dinner_scene.py
    python scripts/preview_dinner_scene.py --robot so101
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("aloha", "so101"), default="aloha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"outputs/scene_preview/{args.robot}.png")
    if args.robot == "aloha":
        from bimanual.sim.aloha_env import AlohaTableSettingEnv

        env = AlohaTableSettingEnv("assets/robots/aloha/task_table_setting_combined_v2.xml")
        camera = "audience_cam"
    else:
        from bimanual.legacy_so101.sim.env import BimanualTableEnv
        from bimanual.legacy_so101.sim.scene_builder import TABLE_HEIGHT

        env = BimanualTableEnv()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0, .05, TABLE_HEIGHT + .12)
        camera.distance = 1.35
        camera.azimuth = 90
        camera.elevation = -28
    try:
        option = mujoco.MjvOption()
        option.geomgroup[1] = 0  # capture rig
        option.geomgroup[5] = 1  # presentation details
        option.sitegroup[:] = 0
        env.model.vis.global_.offwidth = 960
        env.model.vis.global_.offheight = 720
        env.model.vis.quality.offsamples = 8
        with mujoco.Renderer(env.model, height=720, width=960) as renderer:
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            output.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(output, renderer.render())
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
