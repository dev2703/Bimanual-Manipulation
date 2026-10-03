"""Contact predicates shared by physical experts, registry checks and replay."""

from __future__ import annotations

import mujoco


def body_touches(model: mujoco.MjModel, data: mujoco.MjData, body: str, other_prefix: str) -> bool:
    """True if `body` is in contact with any body whose name starts with `other_prefix`."""
    for index in range(data.ncon):
        contact = data.contact[index]
        names = (model.body(int(model.geom_bodyid[contact.geom1])).name,
                 model.body(int(model.geom_bodyid[contact.geom2])).name)
        if body in names and any(name.startswith(other_prefix) for name in names):
            return True
    return False


def right_gripper_touches_drawer_handle(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        names = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
        if "drawer_handle" in names and any(name.startswith("right/") for name in names):
            return True
    return False
