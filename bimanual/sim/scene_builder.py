"""Builds the bimanual table-setting MJCF scene as a string.

Layout (world frame, meters):
  table top:            z = TABLE_HEIGHT, spanning x in [-0.35, 0.35], y in [-0.30, 0.40]
  left arm base:         (-0.28, -0.25, TABLE_HEIGHT), facing +y
  right arm base:        ( 0.28, -0.25, TABLE_HEIGHT), facing +y
  cabinet/drawer:        back edge, centered at (0, 0.42, TABLE_HEIGHT), opens toward -y
  object regions:        sites on the table between the arms and the cabinet

The scene is generated in Python (rather than hand-duplicated XML) so the
two arms are guaranteed identical apart from name prefix and base pose. This
keeps assets/scenes/bimanual_table.xml reproducible from code instead of
being hand-maintained twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bimanual.sim import so101_loader

TABLE_HEIGHT = 0.40
TABLE_HALF_X = 0.35
TABLE_Y_MIN = -0.30
TABLE_Y_MAX = 0.40

SO101_MESHDIR = str(Path(__file__).resolve().parents[2] / "assets" / "robots" / "so101" / "meshes")

LEFT_BASE = (-0.28, -0.25, TABLE_HEIGHT)
RIGHT_BASE = (0.28, -0.25, TABLE_HEIGHT)
# The vendored SO-101's zero pose reaches out along its own local +x axis
# (measured, see docs/decisions.md Phase 1 notes). Rotating each base 90
# degrees about z (quat for a +90 deg z rotation) makes zero pose reach
# toward +y, i.e. forward into the table, matching this scene's layout.
ARM_BASE_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)

CABINET_POS = (0.0, 0.39, TABLE_HEIGHT)
DRAWER_OPEN_DIST = 0.12  # slides toward -y (toward the arms) when opened

# The cabinet body sits on top of the table (bottom flush with the
# tabletop), so its origin is one half-height (0.15) above TABLE_HEIGHT.
# The cutlery tray inside the drawer is at local z=-0.075 with half-height
# 0.01; this constant is its top surface in world z, used to spawn the
# fork/spoon resting on the tray rather than floating or clipping through it.
DRAWER_TRAY_TOP_Z = TABLE_HEIGHT + 0.15 - 0.075 + 0.01

# Object placement regions: (name, xy, radius) on the table surface.
#
# plate_region/mug_region/bottle_region were originally identical to
# those objects' own SceneConfig spawn xy (docs/decisions.md Phase 4
# notes): since L1's XY_JITTER (0.03m) never exceeds a region's radius
# (0.05-0.07m), an object that is NEVER TOUCHED still reads as
# in_region()==True 100% of the time -- confirmed empirically (zero
# action for 300 steps still "succeeds"). This made every plate/mug/
# bottle pick-place success measurement since Phase 1 nearly vacuous.
# Regions are now placed away from spawn (fork/spoon already had this
# property, since they spawn inside the drawer) so success actually
# requires displacing the object.
#
# The first fix attempt (plate_region at x=0.0) passed
# tests/test_scene.py::test_reachability_study_passes but still failed
# almost every real episode -- that test only checks the coarse
# full-joint-range reachable point cloud, not whether
# control/bimanual.py's actual straight-line plan_arm_motion() converges
# along the specific path a skill really takes. x=0.0 sits equidistant
# from both arm bases and is a documented dead zone for this vendored
# arm's kinematics: a real place() run to (0, 0.15) landed a MEASURED
# 243mm from target even though the coarse reachability check passed it.
#
# The SECOND fix attempt then put mug_region/bottle_region at y=+0.15 and
# left cutlery at y=+0.05, all of which are geometrically UNPLACEABLE:
# cabinet_bottom spans x[-0.22,0.22] y[0.11,0.43] at table height, and a
# fully OPEN drawer sweeps x[-0.19,0.19] y[0.01,0.29]. Objects "placed"
# there land on top of the cabinet/drawer, not the table (measured: a mug
# settling at z=0.515 instead of 0.44). That alone explains mug/bottle
# sitting at 0%.
#
# Placeable table area is therefore: y < 0.01 for |x| < 0.22, or |x| >
# 0.22 at any y. Intersected with each arm's IK-validated workspace
# (|x| in [0.08,0.28], see tests/test_ik.py) this leaves the front strip
# and the outer side strips, which is what the layout below uses.
# Verified by tests/test_physics_interactions.py (regions are on open
# table, clear of the cabinet and of the open drawer's swept volume).
REGIONS = {
    "plate_region": (0.11, -0.20, 0.06),
    "mug_region": (0.24, -0.04, 0.05),
    "bottle_region": (-0.24, -0.04, 0.05),
    "left_cutlery_region": (-0.12, -0.20, 0.05),
    "right_cutlery_region": (0.26, -0.20, 0.05),
    "handoff_region": (0.0, -0.05, 0.06),
}


@dataclass(frozen=True)
class SceneConfig:
    """Nominal (L0) object placements; randomization.py perturbs these."""

    # Spawns must also clear the cabinet/open-drawer footprint documented
    # on REGIONS above, and sit far enough from their own target region
    # that success requires real displacement. mug/bottle spawn on the
    # outer side strips (|x| > 0.22, clear of the cabinet at any y); the
    # plate spawns in the front strip (y < 0.01) so an opening drawer
    # can't sweep into it.
    plate_xy: tuple[float, float] = (0.15, -0.06)
    mug_xy: tuple[float, float] = (0.27, 0.12)
    bottle_xy: tuple[float, float] = (-0.27, 0.12)
    # Pre-placed at their own cutlery regions (A5 descope): they are no
    # longer part of TASK_GRAPH, so this is scene dressing + collision
    # realism, not a task whose success could be claimed vacuously.
    fork_xy: tuple[float, float] = (-0.12, -0.20)
    spoon_xy: tuple[float, float] = (0.26, -0.20)
    drawer_opening: float = 0.0  # 0 = closed, DRAWER_OPEN_DIST = fully open
    table_rgba: tuple[float, float, float, float] = (0.55, 0.4, 0.28, 1.0)
    object_rgba_overrides: dict[str, tuple[float, float, float, float]] | None = None


def _arm_fragment(prefix: str, base_xyz: tuple[float, float, float]) -> tuple[str, str]:
    """Delegates to the vendored SO-101 model (so101_loader.py) instead of
    a hand-built kinematic chain. Returns (body_xml, actuator_xml)."""
    return so101_loader.build_arm_fragment(prefix, base_xyz, ARM_BASE_QUAT)


_OBJECT_MATERIAL = {
    "plate": "ceramic_mat",
    "mug": "ceramic_mat",
    "bottle": "glass_mat",
    "fork": "steel_mat",
    "spoon": "steel_mat",
}

# Single source of truth for manipulable-object geometry/physics.
#
# Sizes are scaled to THIS robot, not to a human kitchen. The SO-101 is a
# small 5-DOF arm with a ~3-4cm jaw on a 70x70cm table; the original
# objects were human-scale (an 18cm-diameter plate, a 6cm-diameter mug)
# and none of them were physically graspable by this gripper -- they
# "worked" only because grasping is a kinematic attach. Worse, an 18cm
# disc pinned to the gripper collides with the arm's own forearm and with
# other objects during transit, deflecting the position-controlled arm off
# its planned path (measured: place() transit ending 12cm off target).
# Every object is now at most ~4cm across at its grasp point, matching the
# convention used by sim manipulation benchmarks (robosuite/ALOHA size
# props to the gripper rather than to reality).
#
# Contact parameters: MuJoCo's defaults (friction 1/0.005/0.0001, condim 3,
# no explicit mass) let released objects skid and roll for tens of cm.
# condim=4 adds torsional friction, the raised sliding/torsional/rolling
# coefficients stop the skid, and explicit low masses keep contact forces
# small enough that the light arm isn't knocked off course.
# Aspect ratio matters as much as absolute size: the first pass at these
# sizes made the bottle 14cm tall and 3.6cm wide (ratio ~4) and it tipped
# over on essentially every placement. Heights are kept under ~2x the
# diameter so a placed object actually stays upright.
OBJECT_SPEC: dict[str, dict] = {
    "plate": {"type": "cylinder", "radius": 0.05, "half_height": 0.006, "mass": 0.05, "on_tray": False},
    "mug": {"type": "cylinder", "radius": 0.025, "half_height": 0.030, "mass": 0.03, "on_tray": False},
    "bottle": {"type": "cylinder", "radius": 0.025, "half_height": 0.045, "mass": 0.04, "on_tray": False},
    # on_tray=False: cutlery is now PRE-PLACED on the table (decision
    # A5 descope, invoked -- see docs/decisions.md). Extracting a flat
    # capsule from inside the drawer measured 0-33%, and the table has
    # no room for a 5th spawn+region pair inside arm reach anyway.
    "fork": {"type": "capsule", "radius": 0.010, "half_height": 0.010, "mass": 0.02, "on_tray": False},
    "spoon": {"type": "capsule", "radius": 0.010, "half_height": 0.010, "mass": 0.02, "on_tray": False},
}

OBJECT_CONTACT_ATTRS = (
    'condim="4" friction="1.0 0.05 0.002" solref="0.01 1" solimp="0.9 0.95 0.001"'
)


def resting_center_z(kind: str) -> float:
    """World z of an object's BODY CENTER where it SPAWNS -- on the table,
    or on the drawer's cutlery tray for fork/spoon.

    Use table_center_z() for placement targets: cutlery spawns in the
    drawer but is always placed onto the TABLE, and conflating the two
    released forks/spoons 8.5cm above the table so they bounced away
    (cutlery measured 0% until this was split out).
    """
    spec = OBJECT_SPEC[kind]
    base = DRAWER_TRAY_TOP_Z if spec["on_tray"] else TABLE_HEIGHT
    return base + spec["half_height"]


def table_center_z(kind: str) -> float:
    """World z of an object's BODY CENTER when resting on the TABLE.

    place() targets must be expressed against this, not against
    TABLE_HEIGHT alone: the body frame sits half_height above the
    surface, so commanding the bare table height drives the object a
    half-height INTO the table -- the descend then stalls fighting the
    contact and the object is released ~15cm up and rolls away.
    """
    return TABLE_HEIGHT + OBJECT_SPEC[kind]["half_height"]


def _object_body(name: str, xy: tuple[float, float], kind: str, rgba: str) -> str:
    x, y = xy
    spec = OBJECT_SPEC[kind]
    mat = _OBJECT_MATERIAL[kind]
    z = resting_center_z(kind)
    common = f'material="{mat}" rgba="{rgba}" mass="{spec["mass"]}" {OBJECT_CONTACT_ATTRS}'

    if spec["type"] == "cylinder":
        geom = f'<geom type="cylinder" size="{spec["radius"]} {spec["half_height"]}" {common}/>'
    elif spec["type"] == "capsule":
        # Thickened handle so a parallel-jaw gripper can close around it
        # (decision A5) even though the head geometry is a stand-in.
        r = spec["radius"]
        geom = f'<geom type="capsule" fromto="0 -0.055 0 0 0.055 0" size="{r}" {common}/>'
    else:
        raise ValueError(spec["type"])

    return f"""
    <body name="{name}" pos="{x} {y} {z}">
      <freejoint/>
      {geom}
    </body>"""


def _drawer_xml(opening: float) -> str:
    cx, cy, cz = CABINET_POS
    # cz is the table height; the cabinet's bottom must sit AT or ABOVE the
    # tabletop, not inside the table's solid box geom. Cabinet half-height
    # is 0.15, so its body origin sits 0.15 above the tabletop.
    cabinet_body_z = cz + 0.15
    return f"""
    <body name="cabinet" pos="{cx} {cy} {cabinet_body_z:.4f}">
      <geom name="cabinet_back" type="box" size="0.22 0.02 0.15" pos="0 0.04 0" material="wood_mat"/>
      <geom name="cabinet_wall_r" type="box" size="0.02 0.16 0.15" pos="0.20 -0.12 0" material="wood_mat"/>
      <geom name="cabinet_wall_l" type="box" size="0.02 0.16 0.15" pos="-0.20 -0.12 0" material="wood_mat"/>
      <geom name="cabinet_bottom" type="box" size="0.22 0.16 0.02" pos="0 -0.12 -0.13" material="wood_mat"/>
      <body name="drawer" pos="0 {-opening:.4f} 0">
        <joint name="drawer_slide" type="slide" axis="0 1 0"
               range="{-DRAWER_OPEN_DIST} 0" damping="2.0"/>
        <geom name="drawer_floor" type="box" size="0.19 0.14 0.015" pos="0 -0.12 -0.10" material="wood_mat" rgba="0.95 0.9 0.85 1"/>
        <geom name="drawer_front" type="box" size="0.19 0.015 0.05" pos="0 -0.24 -0.05" material="wood_mat" rgba="0.95 0.9 0.85 1"/>
        <geom name="drawer_wall_r" type="box" size="0.015 0.14 0.05" pos="0.175 -0.12 -0.05" material="wood_mat" rgba="0.95 0.9 0.85 1"/>
        <geom name="drawer_wall_l" type="box" size="0.015 0.14 0.05" pos="-0.175 -0.12 -0.05" material="wood_mat" rgba="0.95 0.9 0.85 1"/>
        <!-- raised cutlery tray (decision A5): lets a parallel-jaw gripper
             close around the fork/spoon handle instead of scraping the floor -->
        <geom name="drawer_tray" type="box" size="0.12 0.06 0.01" pos="0 -0.12 -0.075" material="steel_mat" rgba="0.75 0.75 0.78 1"/>
        <geom name="drawer_handle" type="cylinder" size="0.008 0.05"
              pos="0 -0.245 -0.05" euler="90 0 0" material="steel_mat" rgba="0.15 0.15 0.15 1"/>
      </body>
    </body>"""


def _region_sites(show: bool) -> str:
    if not show:
        return ""
    lines = []
    for name, (x, y, r) in REGIONS.items():
        # Faint and thin -- these are placement-tolerance markers for
        # oracle_predicates.py, not meant to visually dominate the scene.
        # Pass show_regions=False to build_bimanual_table_mjcf() to drop
        # them entirely for a clean demo render.
        lines.append(
            f'    <site name="{name}" pos="{x} {y} {TABLE_HEIGHT + 0.0005:.4f}" '
            f'size="{r} {r} 0.0005" type="cylinder" rgba="0.2 0.9 0.3 0.10"/>'
        )
    return "\n".join(lines)


def _visual_asset_xml(table_rgba: tuple[float, float, float, float]) -> str:
    """Skybox, wood-grain textures (derived from table_rgba so L2 color
    randomization still works), and glossy/metal/ceramic materials for the
    tableware. Purely cosmetic -- no physics/collision effect."""

    def _scale(rgba: tuple[float, float, float, float], factor: float) -> tuple[float, float, float]:
        return tuple(min(1.0, max(0.0, c * factor)) for c in rgba[:3])

    # Subtle contrast + a long/thin texrepeat aspect ratio reads as wood
    # planks with grain fleck, rather than a checkerboard floor tile.
    wood_light = _scale(table_rgba, 1.10)
    wood_dark = _scale(table_rgba, 0.88)

    def rgb(t: tuple[float, float, float]) -> str:
        return f"{t[0]:.3f} {t[1]:.3f} {t[2]:.3f}"

    return f"""
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.75 0.82 0.9" rgb2="0.95 0.95 0.96"
             width="512" height="512"/>
    <texture name="wood_tex" type="2d" builtin="checker" mark="random" markrgb="{rgb(_scale(table_rgba, 1.02))}"
             rgb1="{rgb(wood_light)}" rgb2="{rgb(wood_dark)}" width="300" height="300"/>
    <material name="wood_mat" texture="wood_tex" texuniform="true" texrepeat="2 40"
              specular="0.35" shininess="0.45" reflectance="0.05"/>
    <texture name="floor_tex" type="2d" builtin="checker" mark="edge" markrgb="0.85 0.85 0.85"
             rgb1="0.70 0.66 0.58" rgb2="0.62 0.58 0.51" width="300" height="300"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="10 10"
              specular="0.2" shininess="0.2" reflectance="0.02"/>
    <material name="ceramic_mat" specular="0.55" shininess="0.55" reflectance="0.08"/>
    <material name="glass_mat" specular="0.9" shininess="0.85" reflectance="0.25"/>
    <material name="steel_mat" specular="0.85" shininess="0.75" reflectance="0.35"/>
  </asset>
"""


def build_bimanual_table_mjcf(cfg: SceneConfig | None = None, show_regions: bool = True) -> str:
    """show_regions controls whether the (physics-inert) placement-tolerance
    markers from REGIONS are emitted. Leave True for training/eval so
    oracle_predicates.py's targets exist as sites; pass False for a clean
    demo/presentation render (e.g. scripts/view_scene.py)."""
    cfg = cfg or SceneConfig()
    overrides = cfg.object_rgba_overrides or {}
    plate_rgba = " ".join(str(v) for v in overrides.get("plate", (0.9, 0.9, 0.9, 1.0)))
    mug_rgba = " ".join(str(v) for v in overrides.get("mug", (0.8, 0.2, 0.2, 1.0)))
    bottle_rgba = " ".join(str(v) for v in overrides.get("bottle", (0.1, 0.4, 0.7, 0.6)))
    fork_rgba = " ".join(str(v) for v in overrides.get("fork", (0.75, 0.75, 0.8, 1.0)))
    spoon_rgba = " ".join(str(v) for v in overrides.get("spoon", (0.75, 0.75, 0.8, 1.0)))

    left_body, left_act = _arm_fragment("left", LEFT_BASE)
    right_body, right_act = _arm_fragment("right", RIGHT_BASE)
    so101_asset_xml, so101_default_xml = so101_loader.shared_asset_and_default_xml()
    drawer = _drawer_xml(cfg.drawer_opening)
    regions = _region_sites(show_regions)
    visual_asset_xml = _visual_asset_xml(cfg.table_rgba)

    plate = _object_body("plate", cfg.plate_xy, "plate", plate_rgba)
    mug = _object_body("mug", cfg.mug_xy, "mug", mug_rgba)
    bottle = _object_body("bottle", cfg.bottle_xy, "bottle", bottle_rgba)
    fork = _object_body("fork", cfg.fork_xy, "fork", fork_rgba)
    spoon = _object_body("spoon", cfg.spoon_xy, "spoon", spoon_rgba)

    return f"""
<mujoco model="bimanual_table">
  <compiler angle="radian" meshdir="{SO101_MESHDIR}" autolimits="true"/>
  <option timestep="0.0016666666666666668" integrator="implicitfast"/>

  <contact>
    <!-- The drawer is modeled nested snugly inside the cabinet shell, so
         their geoms overlap by construction. The slide joint's range
         already constrains the drawer's travel, so the shell does not
         need to physically collide with it; without this exclusion the
         two interpenetrate and the resulting contact force pins the
         drawer regardless of the actuator command. -->
    <exclude body1="cabinet" body2="drawer"/>
  </contact>

  {so101_default_xml}
  {so101_asset_xml}
  {visual_asset_xml}

  <default>
    <geom contype="1" conaffinity="1" friction="1.0 0.01 0.001" solimp="0.9 0.95 0.001"/>
    <joint limited="true"/>
  </default>

  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.85 0.88 0.92 1"/>
  </visual>

  <worldbody>
    <light pos="0.3 -0.3 1.8" dir="-0.2 0.3 -1" diffuse="0.85 0.82 0.75" specular="0.3 0.3 0.3"/>
    <light pos="-0.4 0.2 1.6" dir="0.2 -0.1 -1" diffuse="0.35 0.35 0.4"/>

    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0" material="floor_mat"/>

    <body name="table" pos="0 0.05 {TABLE_HEIGHT / 2:.4f}">
      <geom type="box" size="{TABLE_HALF_X} 0.35 {TABLE_HEIGHT / 2:.4f}" material="wood_mat"/>
    </body>

    {regions}

    {drawer}

    {plate}
    {mug}
    {bottle}
    {fork}
    {spoon}

    {left_body}
    {right_body}

    <camera name="global" pos="0 -0.75 0.95" xyaxes="1 0 0 0 0.75 0.66"/>
  </worldbody>

  <actuator>
{left_act}
{right_act}
    <position name="drawer_act" joint="drawer_slide" kp="30"
              ctrlrange="{-DRAWER_OPEN_DIST} 0"/>
  </actuator>
</mujoco>
"""
