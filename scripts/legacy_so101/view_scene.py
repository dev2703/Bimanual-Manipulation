"""Interactive real-time viewer for the bimanual scene.

macOS needs Cocoa on the main thread, so this must be run with the
bundled `mjpython` binary, not plain `python`:

    .venv/bin/mjpython scripts/legacy_so101/view_scene.py

Drag joints with ctrl+right-click, orbit with left-drag, zoom with
scroll. Close the window to exit.
"""

from __future__ import annotations

import time

import mujoco.viewer

from bimanual.legacy_so101.sim.env import BimanualTableEnv, CONTROL_HZ
from bimanual.legacy_so101.sim.scene_builder import TABLE_HEIGHT


def main() -> None:
    env = BimanualTableEnv()
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = (0, .05, TABLE_HEIGHT + .12)
                viewer.cam.distance = 1.35
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -28
                viewer.opt.sitegroup[:] = 0
            while viewer.is_running():
                started = time.monotonic()
                with viewer.lock():
                    env.step(env.data.ctrl.copy())
                viewer.sync()
                time.sleep(max(0, 1 / CONTROL_HZ - (time.monotonic() - started)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
