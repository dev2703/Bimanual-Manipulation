"""Physical-contact ALOHA 2 environment based on MuJoCo Menagerie."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from bimanual.policy.types import EmbodimentSpec

JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")
ACTION_NAMES = tuple(f"{arm}.{joint}" for arm in ("left", "right") for joint in JOINTS)
ALOHA_BIMANUAL = EmbodimentSpec(
    name="aloha2_menagerie",
    model_revision="mujoco-menagerie@822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7",
    state_names=ACTION_NAMES,
    action_names=ACTION_NAMES,
    camera_names=("overhead_cam", "wrist_cam_left", "wrist_cam_right"),
)

PHYSICS_HZ = 600
CONTROL_HZ = 30
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ


class AlohaPhysicalEnv:
    """Minimal contact-only baseline; no grasp, pin, or ghost-object API."""

    def __init__(self, scene_path: str | Path | None = None) -> None:
        if scene_path is None:
            scene_path = Path(__file__).parents[2] / "assets" / "robots" / "aloha" / "task_block.xml"
        self.scene_path = Path(scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = 1.0 / PHYSICS_HZ
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self.after_step: Callable[[], None] | None = None
        self._policy_scene_option = mujoco.MjvOption()
        self._policy_scene_option.geomgroup[5] = 0
        self._policy_scene_option.sitegroup[5] = 0
        self._actuator_names = tuple(self.model.actuator(i).name for i in range(self.model.nu))
        if self._actuator_names != tuple(name.replace(".", "/") for name in ACTION_NAMES):
            raise ValueError(f"unexpected ALOHA actuator order: {self._actuator_names}")
        self.reset()

    def reset(self, seed: int = 0, randomize_block: bool = False) -> dict[str, np.ndarray]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("neutral_pose").id)
        block_joint = self.model.body("task_block").jntadr[0]
        block_qpos = self.model.jnt_qposadr[block_joint]
        if randomize_block:
            rng = np.random.default_rng(seed)
            block_x = float(rng.uniform(-0.05, 0.05))
            block_y = float(rng.uniform(0.14, 0.22))
        else:
            block_x, block_y = 0.0, 0.18
        self.data.qpos[block_qpos : block_qpos + 7] = (block_x, block_y, 0.02, 1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        ALOHA_BIMANUAL.validate(self.state_vector(), action)
        self.data.ctrl[:] = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        if self.after_step is not None:
            self.after_step()
        return self.observation()

    def state_vector(self) -> np.ndarray:
        values = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            values.append(self.data.qpos[qpos_address])
        return np.asarray(values, dtype=np.float32)

    def observation(self) -> dict[str, np.ndarray]:
        state = self.state_vector()
        velocity = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            velocity.append(self.data.qvel[int(self.model.jnt_dofadr[joint_id])])
        return {
            "observation.state": state,
            "observation.velocity": np.asarray(velocity, dtype=np.float32),
            "timestamp": np.asarray([self.data.time], dtype=np.float64),
        }

    def oracle_state(self) -> dict[str, np.ndarray]:
        body_id = self.model.body("task_block").id
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            contacts.append((self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name))
        return {"task_block_pos": self.data.xpos[body_id].copy(), "contacts": contacts}

    def render(self, width: int = 256, height: int = 256) -> dict[str, np.ndarray]:
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            self.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        frames = {}
        for camera in ALOHA_BIMANUAL.camera_names:
            self._renderer.update_scene(
                self.data, camera=camera, scene_option=self._policy_scene_option,
            )
            frames[camera] = self._renderer.render().copy()
        return frames

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


TABLE_SETTING_OBJECTS = {
    "plate": (0.15, -0.02, 0.007),
    "mug": (0.30, 0.12, 0.030),
    "bottle": (-0.30, 0.12, 0.045),
    "fork": (-0.10, 0.355, 0.077),
    "spoon": (0.10, 0.355, 0.077),
}


class AlohaTableSettingEnv(AlohaPhysicalEnv):
    """Primary dinner-table scene; skills are promoted here after block gates."""

    def __init__(self) -> None:
        scene = Path(__file__).parents[2] / "assets" / "robots" / "aloha" / "task_table_setting.xml"
        super().__init__(scene)

    def reset(
        self,
        seed: int = 0,
        randomize_objects: bool = False,
        position_jitter: float = 0.015,
    ) -> dict[str, np.ndarray]:
        if position_jitter < 0:
            raise ValueError("position_jitter must be non-negative")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("neutral_pose").id)
        rng = np.random.default_rng(seed)
        for name, nominal in TABLE_SETTING_OBJECTS.items():
            joint = self.model.joint(f"{name}_free")
            address = int(joint.qposadr[0])
            position = np.asarray(nominal, dtype=np.float64).copy()
            if randomize_objects:
                position[:2] += rng.uniform(-position_jitter, position_jitter, size=2)
            self.data.qpos[address : address + 7] = (*position, 1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.model.joint("drawer_slide").qposadr[0]] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def oracle_state(self) -> dict[str, np.ndarray]:
        state = {}
        for name in TABLE_SETTING_OBJECTS:
            state[f"{name}_pos"] = self.data.xpos[self.model.body(name).id].copy()
        state["drawer_opening"] = np.asarray(
            [self.data.qpos[self.model.joint("drawer_slide").qposadr[0]]], dtype=np.float64,
        )
        return state

    def mug_upright_cosine(self) -> float:
        address = self.model.joint("mug_free").qposadr[0]
        quaternion = self.data.qpos[address + 3 : address + 7]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        return float(rotation[8])
