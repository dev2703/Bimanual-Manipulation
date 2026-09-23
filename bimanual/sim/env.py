"""BimanualTableEnv: the Phase 1 environment API.

Wraps the generated MJCF scene (scene_builder.py) with reset/step/render
and a strict split between:

  proprio()      -- allowed to a learned policy at inference (decision A2,
                     tier "proprioception"): joint pos/vel, gripper opening,
                     forward-kinematics EE pose, relative left-right EE
                     transform.
  oracle_state()  -- forbidden to a learned policy at inference. Simulator
                     object poses, drawer joint value, contacts, and
                     anything else that requires privileged simulator
                     access. Used only for training labels, the RGB
                     verifier's supervision, and evaluation oracles
                     (bimanual/sim/oracle_predicates.py).

tests/test_observation_contract.py (Phase 4) will assert that no key
returned by proprio() overlaps with oracle_state()'s privileged namespace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from bimanual.sim import robot_spec as rs
from bimanual.sim.scene_builder import SceneConfig, build_bimanual_table_mjcf

ARM_PREFIXES = ("left", "right")

# Radians, in the vendored SO-101's own joint frame. Chosen numerically by
# scripts/reachability_study.py to bend the arm forward and down over the
# table from the (rotated, see scene_builder.ARM_BASE_QUAT) zero pose.
HOME_POSE = {
    # Idle pose: end-effector nearly directly above the arm's own base
    # (small XY offset), z around 0.7, at least 0.18m clear of every
    # nominal object position, AND self-collision-free (Phase 3 finding,
    # docs/decisions.md).
    #
    # Three attempts, in order: (1) hovering just above the table put the
    # end-effector centimeters from wherever an object happened to sit,
    # so a later redundant-DOF elbow reconfiguration could brush it; (2)
    # maximizing object clearance alone found a pose 40cm past the table
    # edge, clear of objects but so far from the workspace that any
    # transit's IK solve was itself prone to a local-minimum plateau; (3)
    # a closer, more folded pose fixed both of those but self-collided
    # (shoulder against lower_arm/wrist -- a non-adjacent body pair
    # MuJoCo doesn't auto-exclude). This pose passes all three checks,
    # found by constrained random search over 30k samples, not derived
    # analytically -- revisit if the scene layout changes.
    "shoulder_pan": -0.06,
    "shoulder_lift": -1.65,
    "elbow_flex": -0.43,
    "wrist_flex": 1.62,
    "wrist_roll": 0.0,
}

CAMERA_NAMES = ("global", "left_wrist_cam", "right_wrist_cam")
CONTROL_HZ = 30
PHYSICS_HZ = 600
SUBSTEPS_PER_CONTROL_STEP = PHYSICS_HZ // CONTROL_HZ

# Auto-grasp/release (Phase 4 finding, docs/decisions.md): contact
# friction alone provides ~0 holding force with this vendored gripper
# (verified by replaying a successful expert pick() trajectory without
# ever calling grasp() -- the object's height didn't change at all). The
# scripted experts always called grasp()/release() explicitly by phase
# label, which works but means ANY policy without access to those phase
# labels (e.g. a learned ACT/SmolVLA policy whose action space is just
# joint targets) has no way to ever hold an object. Engaging the same
# kinematic pin automatically -- whenever a gripper is sufficiently
# closed AND near a graspable object -- makes holding a property of
# (gripper state, proximity) instead of a scripted-only callback, so a
# learned policy can trigger it just by closing the gripper at the right
# place. Thresholds match oracle_predicates.holding()'s tolerances.
GRASPABLE_OBJECTS = ("plate", "mug", "bottle", "fork", "spoon")
AUTO_GRASP_XY_TOL = 0.04
AUTO_GRASP_Z_TOL = 0.06
AUTO_GRASP_CLOSE_FRACTION = 0.5  # 0 = fully open, 1 = fully closed


@dataclass
class ResetOptions:
    scene_cfg: SceneConfig = field(default_factory=SceneConfig)
    home_pose: dict[str, float] = field(default_factory=lambda: dict(HOME_POSE))
    interaction_mode: Literal["legacy_kinematic", "physical_contact"] = "legacy_kinematic"


class BimanualTableEnv:
    """Deterministic MuJoCo env for the bimanual table-setting task.

    Not gym-compatible on purpose (yet) -- Phase 1 only needs a minimal,
    explicit API. A gym.Env wrapper can be added later without changing
    this class.
    """

    def __init__(self, options: ResetOptions | None = None) -> None:
        self._base_options = options or ResetOptions()
        self.interaction_mode = self._base_options.interaction_mode
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._rng: np.random.Generator = np.random.default_rng(0)
        # Scripted/kinematic grasp (decision, docs/decisions.md Phase 3
        # notes): the vendored gripper's real pinch geometry proved
        # unreliable to calibrate precisely within budget, so a held
        # object is kinematically re-pinned to the gripper's EE site
        # every step instead of relying on contact friction alone. This
        # is a documented simplification, not silent -- see grasp()/
        # release() below. {prefix: (obj_name, rel_pos, rel_quat) | None}
        self._held: dict[str, tuple[str, np.ndarray, np.ndarray] | None] = {
            "left": None, "right": None,
        }
        self.reset(seed=0)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int, options: ResetOptions | None = None) -> dict:
        opts = options or self._base_options
        self.interaction_mode = opts.interaction_mode
        self._rng = np.random.default_rng(seed)

        xml = build_bimanual_table_mjcf(opts.scene_cfg)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        for prefix in ARM_PREFIXES:
            for suffix, angle in opts.home_pose.items():
                self._set_joint_qpos(rs.joint_name(prefix, suffix), angle)

        self._held = {"left": None, "right": None}
        self._apply_contact_groups()
        mujoco.mj_forward(self.model, self.data)
        return self.proprio()

    def _apply_contact_groups(self) -> None:
        """Disables ARM<->OBJECT contacts via contype/conaffinity bitmasks.

        Grasping here is a kinematic attach (see _held), not friction, so
        physical arm-object contact serves no functional purpose -- but it
        actively corrupts the scripted motions: on RETRACT the still-
        closing jaws drag/knock the object they just released (measured:
        mugs and bottles ending up tipped on their side, and objects
        skidding 10-25cm past their target). Objects still collide with
        the table, cabinet, drawer and each other, so placement settling
        and drawer-pushes-cutlery both stay physical.

        Bits: 1 = static world/furniture, 2 = arms, 4 = objects.
        """
        assert self.model is not None
        object_body_ids = {self.model.body(o).id for o in GRASPABLE_OBJECTS}
        arm_root_ids = {self.model.body(f"{p}_base").id for p in ARM_PREFIXES}

        def is_descendant(body_id: int, roots: set[int]) -> bool:
            while body_id > 0:
                if body_id in roots:
                    return True
                body_id = self.model.body_parentid[body_id]
            return False

        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id in object_body_ids:
                contype, conaffinity = (4, 4 | 1 | 2) if self.is_physical else (4, 4 | 1)
            elif is_descendant(body_id, arm_root_ids):
                contype, conaffinity = (2, 2 | 1 | 4) if self.is_physical else (2, 2 | 1)
            else:
                contype, conaffinity = 1, 1 | 2 | 4
            self.model.geom_contype[geom_id] = contype
            self.model.geom_conaffinity[geom_id] = conaffinity

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------
    def step(self, ctrl: np.ndarray) -> dict:
        """ctrl is the 13-D actuator vector: left (5 joints + gripper),
        right (5 joints + gripper), drawer. Order matches model.actuator
        names, i.e. rs.N_BIMANUAL_ACTIONS + 1.
        """
        assert self.model is not None and self.data is not None
        ctrl = np.asarray(ctrl, dtype=np.float64)
        assert ctrl.shape == (self.model.nu,), (ctrl.shape, self.model.nu)
        self.data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS_PER_CONTROL_STEP):
            mujoco.mj_step(self.model, self.data)
        if not self.is_physical:
            self._auto_grasp_release()
            self._apply_held_objects()
        return self.proprio()

    @property
    def is_physical(self) -> bool:
        return self.interaction_mode == "physical_contact"

    def _gripper_closed_fraction(self, prefix: str) -> float:
        """0.0 = fully open, 1.0 = fully closed, correcting for the
        vendored joint's inverted numeric range (rs.GRIPPER_OPEN is the
        LOW qpos value, rs.GRIPPER_CLOSED is the HIGH one)."""
        qpos = self._get_joint_qpos(rs.gripper_joint_name(prefix))
        span = rs.GRIPPER_CLOSED - rs.GRIPPER_OPEN
        return float((qpos - rs.GRIPPER_OPEN) / span)

    def _auto_grasp_release(self) -> None:
        """Called every step() after physics, before re-pinning held
        objects (see GRASPABLE_OBJECTS/AUTO_GRASP_* docstring note
        above). Idempotent alongside the explicit grasp()/release() the
        scripted experts already call -- both paths converge on the
        same self._held state."""
        assert self.model is not None and self.data is not None
        held_objects = {h[0] for h in self._held.values() if h is not None}
        for prefix in ARM_PREFIXES:
            closed_frac = self._gripper_closed_fraction(prefix)
            if self._held[prefix] is not None:
                if closed_frac < AUTO_GRASP_CLOSE_FRACTION:
                    self.release(prefix)
                continue
            if closed_frac < AUTO_GRASP_CLOSE_FRACTION:
                continue

            site_id = self.model.site(rs.ee_site_name(prefix)).id
            ee_pos = self.data.site_xpos[site_id]
            for obj in GRASPABLE_OBJECTS:
                if obj in held_objects:
                    continue  # already held by the other arm
                obj_pos = self.data.xpos[self.model.body(obj).id]
                xy_close = float(np.hypot(ee_pos[0] - obj_pos[0], ee_pos[1] - obj_pos[1])) < AUTO_GRASP_XY_TOL
                z_close = abs(ee_pos[2] - obj_pos[2]) < AUTO_GRASP_Z_TOL
                if xy_close and z_close:
                    self.grasp(prefix, obj)
                    held_objects.add(obj)
                    break

    # ------------------------------------------------------------------
    # scripted/kinematic grasp (see __init__ docstring note on _held)
    # ------------------------------------------------------------------
    def grasp(self, prefix: str, obj_name: str) -> None:
        """Pins obj_name to arm prefix's EE site at its CURRENT relative
        pose (not a fixed/idealized one), so nothing snaps. Held objects
        are re-pinned every step() until release()."""
        if self.is_physical:
            raise RuntimeError("grasp() is disabled in physical_contact mode; close the gripper through env.step()")
        assert self.model is not None and self.data is not None
        site_id = self.model.site(rs.ee_site_name(prefix)).id
        site_pos = self.data.site_xpos[site_id].copy()
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, self.data.site_xmat[site_id])

        adr = self._obj_qpos_adr(obj_name)
        obj_pos = self.data.qpos[adr : adr + 3].copy()
        obj_quat = self.data.qpos[adr + 3 : adr + 7].copy()

        site_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(site_quat_inv, site_quat)
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, obj_pos - site_pos, site_quat_inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, site_quat_inv, obj_quat)

        self._held[prefix] = (obj_name, rel_pos, rel_quat)
        self._set_object_collidable(obj_name, False)

    def release(self, prefix: str) -> None:
        held = self._held[prefix]
        self._held[prefix] = None
        if held is not None:
            self._set_object_collidable(held[0], True)

    def _set_object_collidable(self, obj_name: str, collidable: bool) -> None:
        """Enables/disables ALL contacts for an object's geoms.

        A kinematically pinned object has its qpos overwritten every step,
        so it never responds to contact -- but other bodies still react to
        IT. While carried, that means the held object ploughs through the
        arm carrying it and through anything else on the table, and the
        resulting contact forces deflect the position-controlled arm off
        its planned path (measured: a held plate pushing a transit 12cm
        off target, and objects getting swept across the table). Making a
        held object non-colliding for the duration of the carry, and
        physical again the instant it is released, keeps transport
        deterministic while still giving real contact physics for the
        parts that matter (settling onto the table after release).
        """
        assert self.model is not None
        body_id = self.model.body(obj_name).id
        # Restores the object bit-group set by _apply_contact_groups().
        contype, conaffinity = (4, 4 | 1) if collidable else (0, 0)
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] != body_id:
                continue
            self.model.geom_contype[geom_id] = contype
            self.model.geom_conaffinity[geom_id] = conaffinity

    def _apply_held_objects(self) -> None:
        assert self.model is not None and self.data is not None
        for prefix, held in self._held.items():
            if held is None:
                continue
            obj_name, rel_pos, rel_quat = held
            site_id = self.model.site(rs.ee_site_name(prefix)).id
            site_pos = self.data.site_xpos[site_id].copy()
            site_quat = np.zeros(4)
            mujoco.mju_mat2Quat(site_quat, self.data.site_xmat[site_id])

            new_pos = np.zeros(3)
            mujoco.mju_rotVecQuat(new_pos, rel_pos, site_quat)
            new_pos += site_pos
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, site_quat, rel_quat)

            adr = self._obj_qpos_adr(obj_name)
            self.data.qpos[adr : adr + 3] = new_pos
            self.data.qpos[adr + 3 : adr + 7] = new_quat
            dof_adr = self.model.body(obj_name).jntadr[0]
            dof_start = self.model.jnt_dofadr[dof_adr]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        if any(v is not None for v in self._held.values()):
            mujoco.mj_forward(self.model, self.data)

    def _obj_qpos_adr(self, obj_name: str) -> int:
        assert self.model is not None
        joint_id = self.model.body(obj_name).jntadr[0]
        return int(self.model.jnt_qposadr[joint_id])

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def proprio(self) -> dict:
        """Everything a real robot controller could know without vision:
        joint state, gripper opening, and FK-derived EE poses/transform.
        Safe to feed to a learned policy at inference (decision A2)."""
        assert self.model is not None and self.data is not None
        out: dict = {}
        ee_poses = {}
        for prefix in ARM_PREFIXES:
            qpos = np.array(
                [self._get_joint_qpos(rs.joint_name(prefix, s)) for s in rs.JOINT_SUFFIXES]
            )
            qvel = np.array(
                [self._get_joint_qvel(rs.joint_name(prefix, s)) for s in rs.JOINT_SUFFIXES]
            )
            gripper_opening = self._get_joint_qpos(rs.gripper_joint_name(prefix))
            site_id = self.model.site(rs.ee_site_name(prefix)).id
            ee_pos = self.data.site_xpos[site_id].copy()
            ee_mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
            ee_poses[prefix] = (ee_pos, ee_mat)
            out[f"{prefix}_joint_pos"] = qpos
            out[f"{prefix}_joint_vel"] = qvel
            out[f"{prefix}_gripper_opening"] = gripper_opening
            out[f"{prefix}_ee_pos"] = ee_pos
            out[f"{prefix}_ee_mat"] = ee_mat

        left_pos, left_mat = ee_poses["left"]
        right_pos, right_mat = ee_poses["right"]
        out["relative_ee_pos"] = right_pos - left_pos
        out["relative_ee_mat"] = left_mat.T @ right_mat
        return out

    def oracle_state(self) -> dict:
        """Privileged simulator state. Training labels and evaluation
        oracles ONLY -- never feed this to a learned policy at inference
        (decision A2/A3, enforced by tests/test_observation_contract.py
        in Phase 4)."""
        assert self.model is not None and self.data is not None
        out: dict = {"drawer_opening": self._get_joint_qpos("drawer_slide")}
        out["drawer_handle_pos"] = self.data.geom_xpos[self.model.geom("drawer_handle").id].copy()
        for obj in ("plate", "mug", "bottle", "fork", "spoon"):
            body_id = self.model.body(obj).id
            out[f"{obj}_pos"] = self.data.xpos[body_id].copy()
            out[f"{obj}_mat"] = self.data.xmat[body_id].reshape(3, 3).copy()
        # Most of the vendored SO-101 mesh geoms have no `name` attribute
        # (only their parent bodies are named -- and prefixed left_/right_
        # by so101_loader.py), so contacts are recorded by body name, not
        # geom name.
        contacts = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.body(self.model.geom_bodyid[c.geom1]).name
            b2 = self.model.body(self.model.geom_bodyid[c.geom2]).name
            contacts.append((b1, b2))
        out["contacts"] = contacts
        return out

    def render(self, cameras: tuple[str, ...] = CAMERA_NAMES, width: int = 256, height: int = 256) -> dict[str, np.ndarray]:
        assert self.model is not None and self.data is not None
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        frames = {}
        for cam in cameras:
            self._renderer.update_scene(self.data, camera=cam)
            frames[cam] = self._renderer.render().copy()
        return frames

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _set_joint_qpos(self, name: str, value: float) -> None:
        assert self.model is not None and self.data is not None
        adr = self.model.joint(name).qposadr[0]
        self.data.qpos[adr] = value

    def _get_joint_qpos(self, name: str) -> float:
        assert self.model is not None and self.data is not None
        adr = self.model.joint(name).qposadr[0]
        return float(self.data.qpos[adr])

    def _get_joint_qvel(self, name: str) -> float:
        assert self.model is not None and self.data is not None
        adr = self.model.joint(name).dofadr[0]
        return float(self.data.qvel[adr])
