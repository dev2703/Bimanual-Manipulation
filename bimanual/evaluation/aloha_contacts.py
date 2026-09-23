"""Contact predicates shared by physical experts and held-action replay."""

from __future__ import annotations

import mujoco


def right_gripper_touches_drawer_handle(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        names = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
        if "drawer_handle" in names and any(name.startswith("right/") for name in names):
            return True
    return False
