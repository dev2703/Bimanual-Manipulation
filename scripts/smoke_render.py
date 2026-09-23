"""Phase 0 smoke test.

Builds a trivial MuJoCo scene with three cameras (standing in for the
eventual global/left-wrist/right-wrist rig), renders each offscreen on
macOS, and writes a short MP4. This only proves the render path works
end-to-end on this machine before any robot/scene assets exist.

Run with: make demo
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

SMOKE_XML = """
<mujoco model="smoke_test">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="ball" pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="0.9 0.2 0.2 1"/>
    </body>
    <camera name="global" pos="0 -1.5 1.2" mode="targetbody" target="ball"/>
    <camera name="left_wrist" pos="-0.6 -0.6 0.8" mode="targetbody" target="ball"/>
    <camera name="right_wrist" pos="0.6 -0.6 0.8" mode="targetbody" target="ball"/>
  </worldbody>
</mujoco>
"""

CAMERAS = ("global", "left_wrist", "right_wrist")
N_STEPS = 90
WIDTH, HEIGHT = 256, 256


def main() -> None:
    model = mujoco.MjModel.from_xml_string(SMOKE_XML)
    data = mujoco.MjData(model)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    writers = {
        cam: imageio.get_writer(out_dir / f"smoke_{cam}.mp4", fps=30)
        for cam in CAMERAS
    }

    try:
        for _ in range(N_STEPS):
            mujoco.mj_step(model, data)
            for cam in CAMERAS:
                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                writers[cam].append_data(np.asarray(frame))
    finally:
        for w in writers.values():
            w.close()
        renderer.close()

    for cam in CAMERAS:
        print(f"wrote outputs/smoke_{cam}.mp4")


if __name__ == "__main__":
    main()
