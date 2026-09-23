"""Interactive real-time viewer for the bimanual scene.

macOS needs Cocoa on the main thread, so this must be run with the
bundled `mjpython` binary, not plain `python`:

    .venv/bin/mjpython scripts/view_scene.py

Drag joints with ctrl+right-click, orbit with left-drag, zoom with
scroll. Close the window to exit.
"""

from __future__ import annotations

import mujoco.viewer

from bimanual.sim.env import BimanualTableEnv


def main() -> None:
    env = BimanualTableEnv()
    mujoco.viewer.launch(env.model, env.data)


if __name__ == "__main__":
    main()
