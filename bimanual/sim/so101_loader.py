"""Loads the vendored SO-101 MJCF (assets/robots/so101/so101.xml, see
NOTICE.md there) and instantiates two renamed, repositioned copies for the
bimanual scene.

Why this exists: MJCF has no native way to include the same body/joint/site
tree twice with different names, so a single-arm MJCF can't just be
<include>d twice. This module parses the vendored file once with
ElementTree, and for each arm:
  - deep-copies the kinematic body tree rooted at "base",
  - prefixes every body/joint/site name with "{prefix}_" so left and right
    don't collide,
  - repositions the copied "base" body at the arm's mount pose,
  - adds a wrist camera to the gripper body, aimed at the moving jaw via
    MuJoCo's `mode="targetbody"` (robust -- avoids hand-computing camera
    xyaxes, which is exactly what went wrong in the original hand-built
    proxy model this replaces),
  - deep-copies and prefixes the <actuator> entries (name + joint refs).

Mesh and material definitions are NOT prefixed and are emitted once,
shared by both arm instances, since they just reference the same STL
geometry data.

The joint names this project's kinematic proxy previously invented
(waist/shoulder/elbow/wrist_pitch/wrist_roll) are replaced by the real
vendor names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper. robot_spec.py's JOINT_SUFFIXES / GRIPPER_SUFFIX are
updated to match -- see that module.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

SO101_XML_PATH = Path(__file__).resolve().parents[2] / "assets" / "robots" / "so101" / "so101.xml"

_RENAME_TAGS = ("body", "joint", "site")


def _load_source_tree() -> ET.ElementTree:
    return ET.parse(SO101_XML_PATH)


def _prefix_names(elem: ET.Element, prefix: str) -> None:
    """Recursively prefixes name= on body/joint/site elements, and joint=
    refs on actuator elements, in place."""
    if elem.tag in _RENAME_TAGS and "name" in elem.attrib:
        elem.attrib["name"] = f"{prefix}_{elem.attrib['name']}"
    if elem.tag == "actuator" or elem.tag == "position":
        if "joint" in elem.attrib:
            elem.attrib["joint"] = f"{prefix}_{elem.attrib['joint']}"
        if "name" in elem.attrib:
            elem.attrib["name"] = f"{prefix}_{elem.attrib['name']}"
    for child in elem:
        _prefix_names(child, prefix)


def shared_asset_and_default_xml() -> tuple[str, str]:
    """Mesh/material <asset> and the so101_new_calib/sts3215/backlash
    <default> block, shared once by both arm instances (see module
    docstring: these are not per-instance)."""
    tree = _load_source_tree()
    root = tree.getroot()
    asset_xml = "\n".join(ET.tostring(e, encoding="unicode") for e in root.findall("asset"))
    default_xml = "\n".join(ET.tostring(e, encoding="unicode") for e in root.findall("default"))
    return asset_xml, default_xml


def build_arm_fragment(
    prefix: str,
    base_pos: tuple[float, float, float],
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> tuple[str, str]:
    """Returns (body_xml, actuator_xml) for one arm instance, with all
    body/joint/site names prefixed and the base body repositioned to
    base_pos/base_quat in the bimanual scene's world frame."""
    tree = _load_source_tree()
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None
    base_body = worldbody.find("body")
    assert base_body is not None and base_body.attrib.get("name") == "base"

    arm = ET.fromstring(ET.tostring(base_body, encoding="unicode"))
    _prefix_names(arm, prefix)
    arm.attrib["pos"] = f"{base_pos[0]} {base_pos[1]} {base_pos[2]}"
    arm.attrib["quat"] = f"{base_quat[0]} {base_quat[1]} {base_quat[2]} {base_quat[3]}"

    # Wrist camera: attach to the gripper body (closest link to the
    # end-effector) and let MuJoCo's targetbody mode aim it at the moving
    # jaw, rather than hand-deriving xyaxes.
    gripper_body_name = f"{prefix}_gripper"
    jaw_body_name = f"{prefix}_moving_jaw_so101_v1"
    gripper_body = arm.find(f".//body[@name='{gripper_body_name}']")
    assert gripper_body is not None, "vendored so101.xml body names changed upstream"
    ET.SubElement(
        gripper_body,
        "camera",
        {
            "name": f"{prefix}_wrist_cam",
            "pos": "-0.02 -0.09 -0.10",
            "mode": "targetbody",
            "target": jaw_body_name,
            "fovy": "70",
        },
    )

    actuator_root = root.find("actuator")
    assert actuator_root is not None
    actuators = ET.fromstring(ET.tostring(actuator_root, encoding="unicode"))
    _prefix_names(actuators, prefix)
    actuator_xml = "\n".join(ET.tostring(e, encoding="unicode") for e in actuators)

    body_xml = ET.tostring(arm, encoding="unicode")
    return body_xml, actuator_xml


def ee_site_name(prefix: str) -> str:
    """The vendored model's own end-effector reference frame, renamed."""
    return f"{prefix}_gripperframe"


def jaw_body_name(prefix: str) -> str:
    return f"{prefix}_moving_jaw_so101_v1"
