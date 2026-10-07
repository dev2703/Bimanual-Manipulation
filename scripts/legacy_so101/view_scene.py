"""Interactive real-time viewer for the bimanual scene.

macOS needs Cocoa on the main thread, so this must be run with the
bundled `mjpython` binary, not plain `python`:

    .venv/bin/mjpython scripts/legacy_so101/view_scene.py

Drag joints with ctrl+right-click, orbit with left-drag, zoom with
scroll. Close the window to exit.
"""

from __future__ import annotations

import mujoco.viewer

from bimanual.legacy_so101.sim.env import BimanualTableEnv
from bimanual.legacy_so101.sim.scene_builder import TABLE_HEIGHT


def main() -> None:
    env = BimanualTableEnv()
    env.model.stat.center[:] = (0, .05, TABLE_HEIGHT + .12)
    env.model.stat.extent = .9
    env.model.vis.global_.azimuth = 90
    env.model.vis.global_.elevation = -28
    try:
        mujoco.viewer.launch(env.model, env.data)
    finally:
        env.close()


if __name__ == "__main__":
    main()
