"""Balance / reach / lift / carry environment for the Hulkbuster humanoid.

The robot is the generated MJCF at ``model/humanoid.xml``.  A
training scene is assembled around it with MjSpec: floor, a 0.30 m pedestal
holding a 0.5 m, 12 kg crate, a goal marker, and position actuators on all 29
hinges.  The grasp is a simplification: when both hand sites are within
``GRASP_RADIUS`` of the crate, a weld equality attaches the crate to the torso
at its current relative pose.  The policy therefore learns to approach and
lift, not to close fingers.

Stages (reward terms are added cumulatively):

0. stand / balance under random pushes
1. reach the crate with both hands
2. lift it (and hold it)
3. carry it to the goal marker

Run ``train_sac.py`` for stage 0, then warm-start the next stage.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Scene constants (metres, kilograms, seconds)
# ---------------------------------------------------------------------------
BOX_X = 0.78
BOX_Z0 = 0.55          # 0.5 m crate on the 0.30 m pedestal
BOX_HALF = 0.25
BOX_MASS = 12.0
GOAL_X = 1.50
GRASP_RADIUS = 0.45
NOMINAL_ROOT_Z = 1.2210

SIM_DT = 0.002
FRAME_SKIP = 10
CONTROL_DT = SIM_DT * FRAME_SKIP
MAX_EPISODE_STEPS = 400   # 8 s at 50 Hz

# per-joint action scale around the nominal pose (radians)
_ACTION_SCALE = {
    "shoulder_pitch": 0.80, "shoulder_roll": 0.70, "shoulder_yaw": 0.50,
    "elbow": 0.80, "wrist_yaw": 0.40, "wrist_pitch": 0.40, "wrist_roll": 0.40,
    "torso_roll": 0.45, "torso_pitch": 0.45, "torso_yaw": 0.45,
    "hip_pitch": 0.70, "hip_roll": 0.55, "hip_yaw": 0.45,
    "leg_knee": 0.80, "ankle_pitch": 0.55, "ankle_roll": 0.45,
}

_NOMINAL = {
    "hip_pitch": -0.20, "hip_roll": 0.0, "hip_yaw": 0.0,
    "leg_knee": 0.60, "ankle_pitch": -0.40, "ankle_roll": 0.0,
    "shoulder_pitch": -0.40, "shoulder_roll": 0.0, "shoulder_yaw": 0.0,
    "elbow": -0.50, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0,
    "torso_roll": 0.0, "torso_pitch": 0.0, "torso_yaw": 0.0,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _role(joint_name: str) -> str:
    for role in sorted(_ACTION_SCALE, key=len, reverse=True):
        if joint_name.endswith(role):
            return role
    return "hip_pitch"


def _nominal_for(joint_name: str) -> float:
    base = _NOMINAL[_role(joint_name)]
    if joint_name.startswith("right_") and _role(joint_name) == "shoulder_roll":
        return -base
    return base


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def build_model(robot_xml: str | Path, kp_scale: float = 5.0,
                kv_scale: float = 0.30) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.option.timestep = SIM_DT
    spec.option.tolerance = 1e-8
    spec.option.iterations = 20

    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20.0, 20.0, 0.1], pos=[0, 0, 0],
        rgba=[0.78, 0.78, 0.80, 1.0], friction=[1.0, 0.02, 0.01],
        contype=1, conaffinity=1,
    )

    spec.worldbody.add_geom(
        name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.25, 0.25, 0.15], pos=[BOX_X, 0, 0.15],
        rgba=[0.55, 0.55, 0.58, 1.0], friction=[0.9, 0.02, 0.01], condim=4,
    )

    crate = spec.worldbody.add_body(name="box", pos=[BOX_X, 0, BOX_Z0])
    crate.add_freejoint(name="box_freejoint")
    crate.add_geom(
        name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BOX_HALF] * 3, mass=BOX_MASS,
        rgba=[0.72, 0.42, 0.12, 1.0], friction=[0.9, 0.02, 0.01], condim=4,
    )

    spec.worldbody.add_geom(
        name="goal", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.30, 0.30, 0.004], pos=[GOAL_X, 0, 0.004],
        rgba=[0.05, 0.85, 0.15, 0.45], contype=0, conaffinity=0, group=1,
    )
    spec.worldbody.add_light(
        name="top_light", pos=[0, 0, 4], dir=[0, 0, -1],
        diffuse=[0.7, 0.7, 0.7],
    )

    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        effort = max(abs(float(joint.actfrcrange[0])), abs(float(joint.actfrcrange[1])))
        if effort <= 0.0:
            effort = 50.0
        kp, kv = kp_scale * effort, kv_scale * effort
        act = spec.add_actuator(name=f"act_{joint.name}", target=joint.name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=kp, kv=kv)
        act.ctrllimited = True
        act.ctrlrange = [float(joint.range[0]), float(joint.range[1])]
        act.forcelimited = True
        act.forcerange = [-effort, effort]

    spec.add_equality(
        name="grasp", type=mujoco.mjtEq.mjEQ_WELD,
        name1="torso_link", name2="box", active=0,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
    )
    return spec.compile()


class HulkbusterLiftEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": int(1 / CONTROL_DT)}

    def __init__(
        self,
        stage: int = 0,
        robot_xml: str | Path | None = None,
        randomize: bool = True,
        push_prob: float = 0.0015,
        render_mode: str | None = None,
        kp_scale: float = 5.0,
        kv_scale: float = 0.30,
        gravity_comp: bool = True,
        action_scale_mult: float = 1.0,
    ):
        super().__init__()
        assert stage in (0, 1, 2, 3)
        self.stage = stage
        self.randomize = randomize
        self.push_prob = push_prob
        self.render_mode = render_mode
        self.robot_xml = Path(robot_xml) if robot_xml else (
            _repo_root() / "model" / "humanoid.xml")
        if not self.robot_xml.exists():
            raise FileNotFoundError(f"{self.robot_xml} missing")

        self.model = build_model(self.robot_xml, kp_scale, kv_scale)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.renderer = None

        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.left_hand = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_sphere")
        self.right_hand = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_sphere")

        self.act_joint_names: list[str] = []
        self.qpos_adr: list[int] = []
        self.dof_adr: list[int] = []
        self.nominal = np.zeros(0, dtype=np.float64)
        self.action_scale = np.zeros(0, dtype=np.float64)
        self.joint_lo = np.zeros(0, dtype=np.float64)
        self.joint_hi = np.zeros(0, dtype=np.float64)
        for i in range(self.model.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            self.act_joint_names.append(name)
            self.qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            self.dof_adr.append(int(self.model.jnt_dofadr[jid]))
            self.nominal = np.append(self.nominal, _nominal_for(name))
            self.action_scale = np.append(self.action_scale, _ACTION_SCALE[_role(name)])
            lo, hi = self.model.jnt_range[jid]
            self.joint_lo = np.append(self.joint_lo, lo)
            self.joint_hi = np.append(self.joint_hi, hi)
        self.qpos_adr = np.asarray(self.qpos_adr)
        self.dof_adr = np.asarray(self.dof_adr)
        self.nominal = np.asarray(self.nominal)
        self.action_scale = np.asarray(self.action_scale)
        self.joint_lo = np.asarray(self.joint_lo)
        self.joint_hi = np.asarray(self.joint_hi)
        self.n_act = len(self.act_joint_names)
        self.action_scale = self.action_scale * action_scale_mult
        self.active = np.ones(self.n_act, dtype=np.float64)
        self.gravity_comp = gravity_comp
        self.act_effort = np.array([
            max(abs(float(self.model.actuator_forcerange[i, 0])),
                abs(float(self.model.actuator_forcerange[i, 1])))
            for i in range(self.model.nu)])

        bj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        self.box_qpos_adr = int(self.model.jnt_qposadr[bj])
        self.box_dof_adr = int(self.model.jnt_dofadr[bj])
        self.grasp_eq = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
        self.eq_data = self.model.eq_data  # writable view

        self.prev_action = np.zeros(self.n_act, dtype=np.float64)
        self.grasped = False
        self.push_steps = 0
        self.push_force = np.zeros(3)
        self.steps = 0

        # observation layout
        self.obs_dim = 1 + 4 + 3 + 3 + self.n_act * 3 + 6 + 1 + 3 + 1
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.n_act,), dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos
        q[:] = 0.0
        q[2] = NOMINAL_ROOT_Z
        q[3] = 1.0
        q[self.qpos_adr] = self.nominal
        if self.randomize:
            q[0] += rng.uniform(-0.03, 0.03)
            q[1] += rng.uniform(-0.03, 0.03)
            yaw = rng.uniform(-0.12, 0.12)
            q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            q[self.qpos_adr] += rng.uniform(-0.04, 0.04, size=self.n_act)
            q[self.qpos_adr] = np.clip(q[self.qpos_adr], self.joint_lo, self.joint_hi)

        b = self.box_qpos_adr
        q[b + 0] = BOX_X + (rng.uniform(-0.02, 0.02) if self.randomize else 0.0)
        q[b + 1] = rng.uniform(-0.03, 0.03) if self.randomize else 0.0
        q[b + 2] = BOX_Z0 + 0.003
        byaw = rng.uniform(-0.20, 0.20) if self.randomize else 0.0
        q[b + 3:b + 7] = [math.cos(byaw / 2), 0, 0, math.sin(byaw / 2)]

        self.data.qvel[:] = 0.0
        if self.randomize:
            self.data.qvel[0:3] = rng.uniform(-0.03, 0.03, size=3)
            self.data.qvel[3:6] = rng.uniform(-0.05, 0.05, size=3)
        self.data.eq_active[self.grasp_eq] = 0
        self.data.xfrc_applied[:] = 0.0
        self.prev_action[:] = 0.0
        self.grasped = False
        self.push_steps = 0
        self.push_force[:] = 0.0
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        action = action * self.active
        target = self.nominal + self.action_scale * action
        target = np.clip(target, self.joint_lo + 0.02, self.joint_hi - 0.02)
        self.data.ctrl[:] = target

        self._maybe_push()
        for _ in range(FRAME_SKIP):
            if self.gravity_comp:
                mujoco.mj_forward(self.model, self.data)
                bias = self.data.qfrc_bias[self.dof_adr]
                self.data.qfrc_applied[self.dof_adr] = np.clip(
                    bias, -self.act_effort, self.act_effort)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        self._update_grasp()

        up = self._up()
        root_z = float(self.data.xpos[self.torso, 2])
        reward = self._reward(action, up, root_z)
        terminated = bool(
            root_z < 0.70 or up < 0.50 or not np.isfinite(self.data.qpos).all())
        if terminated:
            reward -= 5.0
        truncated = self.steps >= MAX_EPISODE_STEPS
        self.prev_action[:] = action
        info = self._info(up, root_z)
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # ----------------------------------------------------------------- rewards
    def _up(self) -> float:
        return float(self.data.xmat[self.torso].reshape(3, 3)[2, 2])

    def _reward(self, action: np.ndarray, up: float, root_z: float) -> float:
        d = self.data
        vxy = d.qvel[0:2]
        w = d.qvel[3:6]
        r = 0.5
        r += 1.0 * math.exp(10.0 * (up - 1.0))
        r += 0.5 * math.exp(-((root_z - NOMINAL_ROOT_Z) / 0.12) ** 2)
        r += 0.2 * math.exp(-float(np.dot(vxy, vxy)) / 0.25)
        r += 0.1 * math.exp(-float(np.dot(w, w)) / 4.0)
        r -= 0.05 * float(np.mean((action - self.prev_action) ** 2))
        r -= 1e-6 * float(np.mean(d.actuator_force ** 2))
        margin = 0.05
        r -= 0.1 * float(np.sum(
            np.maximum(0.0, self.joint_lo + margin - d.qpos[self.qpos_adr]) +
            np.maximum(0.0, d.qpos[self.qpos_adr] - self.joint_hi + margin)))

        box = d.xpos[self.box_body]
        if self.stage >= 1:
            hl = d.site_xpos[self.left_hand]
            hr = d.site_xpos[self.right_hand]
            dL = float(np.linalg.norm(hl - box))
            dR = float(np.linalg.norm(hr - box))
            r += 0.8 * 0.5 * (math.exp(-2.0 * dL) + math.exp(-2.0 * dR))
        if self.stage >= 2:
            lift = float(np.clip((box[2] - BOX_Z0) / 0.30, 0.0, 1.0))
            r += 1.5 * lift
            if self.grasped:
                r += 0.5
        if self.stage >= 3:
            dgoal = float(np.linalg.norm(box[0:2] - np.array([GOAL_X, 0.0])))
            r += 2.0 * math.exp(-1.5 * dgoal)
        return r

    def _info(self, up: float, root_z: float) -> dict:
        d = self.data
        box = d.xpos[self.box_body]
        hl = d.site_xpos[self.left_hand]
        hr = d.site_xpos[self.right_hand]
        return {
            "root_z": root_z,
            "up": up,
            "box_z": float(box[2]),
            "box_goal_dist": float(np.linalg.norm(box[0:2] - np.array([GOAL_X, 0.0]))),
            "hand_dist": float(0.5 * (np.linalg.norm(hl - box) + np.linalg.norm(hr - box))),
            "grasped": float(self.grasped),
            "success": float(
                self.stage >= 2 and box[2] > BOX_Z0 + 0.15 and
                (self.stage < 3 or
                 np.linalg.norm(box[0:2] - np.array([GOAL_X, 0.0])) < 0.30)),
        }

    # ------------------------------------------------------------------ grasp
    def _update_grasp(self) -> None:
        if self.grasped or self.stage < 2:
            return
        d = self.data
        box = d.xpos[self.box_body]
        hl = d.site_xpos[self.left_hand]
        hr = d.site_xpos[self.right_hand]
        if max(np.linalg.norm(hl - box), np.linalg.norm(hr - box)) > GRASP_RADIUS:
            return
        rt = d.xmat[self.torso].reshape(3, 3)
        relpos = rt.T @ (box - d.xpos[self.torso])
        relquat = _quat_mul(_quat_inv(d.xquat[self.torso]), d.xquat[self.box_body])
        relquat /= np.linalg.norm(relquat)
        self.eq_data[self.grasp_eq, 3:6] = relpos
        self.eq_data[self.grasp_eq, 6:10] = relquat
        self.eq_data[self.grasp_eq, 10] = 1.0
        d.eq_active[self.grasp_eq] = 1
        self.grasped = True

    # ------------------------------------------------------------------ pushes
    def _maybe_push(self) -> None:
        rng = self.np_random
        d = self.data
        if self.push_steps > 0:
            self.push_steps -= 1
            d.xfrc_applied[self.torso, 0:3] = self.push_force
            if self.push_steps == 0:
                d.xfrc_applied[self.torso, 0:3] = 0.0
            return
        if rng.random() < self.push_prob:
            angle = rng.uniform(-math.pi, math.pi)
            magnitude = rng.uniform(120.0, 320.0)
            self.push_force = np.array([
                magnitude * math.cos(angle), magnitude * math.sin(angle), 0.0])
            self.push_steps = 6
            d.xfrc_applied[self.torso, 0:3] = self.push_force

    # -------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        d = self.data
        q = d.qpos
        root = q[0:3]
        quat = q[3:7]
        rt = _quat_rot(quat)
        box = d.xpos[self.box_body]
        hl = d.site_xpos[self.left_hand]
        hr = d.site_xpos[self.right_hand]
        goal = np.array([GOAL_X, 0.0, 0.0])
        obs = np.concatenate([
            [(root[2] - 1.0) / 0.5],
            quat,
            d.qvel[0:3] * 0.5,
            d.qvel[3:6] * 0.25,
            q[self.qpos_adr] - self.nominal,
            d.qvel[self.dof_adr] * 0.25,
            self.prev_action,
            (rt.T @ (box - hl)) * 2.0,
            (rt.T @ (box - hr)) * 2.0,
            [(box[2] - BOX_Z0) / 0.30],
            (rt.T @ (goal - box)) * 0.5,
            [float(self.grasped)],
        ])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="track")
            return self.renderer.render()
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 4.0
            self.viewer.cam.elevation = -15.0
            self.viewer.cam.lookat[:] = [0.6, 0.0, 0.8]
        self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
