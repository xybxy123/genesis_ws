import math
import torch
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from tensordict import TensorDict


def gs_rand(lower, upper, batch_shape):
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


class DogEnv:
    def __init__(
        self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False
    ):

        self.num_envs = num_envs

        # 单个仿真环境的观测向量维度
        self.num_obs = obs_cfg["num_obs"]

        # 特权观测维度(暂时不启用)
        self.num_privileged_obs = None

        # 动作指令维度(控制的关节数量)
        self.num_actions = env_cfg["num_actions"]

        self.num_commands = command_cfg["num_commands"]

        self.device = gs.device

        self.dt = 0.02

        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg

        self.obs_cfg = obs_cfg

        self.reward_cfg = reward_cfg

        self.command_cfg = command_cfg

        # 为 rsl_rl logger 提供 cfg 属性
        self.cfg = {
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
        }

        self.obs_scales = obs_cfg["obs_scales"]

        self.reward_scales = reward_cfg["reward_scales"]

        # 创建场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=30,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=gs.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="assets/xml_test.xml",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        self.scene.build(n_envs=num_envs)

        # 关节索引处理
        self.motor_dofs = torch.tensor(
            [
                self.robot.get_joint(name).dof_start
                for name in self.env_cfg["joint_names"]
            ],
            dtype=gs.tc_int,
            device=self.device,
        )

        # PD 参数设置
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # 缺少初始化奖励函数字典（存储各奖励项的计算函数）和单轮奖励累加字典（存储各奖励项的累计值）

        # 初始化各种 Buffer (逻辑同 Go2)
        self._init_buffers()

    def _init_buffers(self):
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][n]
                for n in self.env_cfg["joint_names"]
            ],
            device=self.device,
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device
        )

        # 奖励函数映射
        self.reward_functions = {
            name: getattr(self, "_reward_" + name) for name in self.reward_scales.keys()
        }
        self.extras = {"observations": {}}

    def step(self, actions):
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )

        target_dof_pos = (
            self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        )
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1

        # 获取状态
        self.base_quat = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)

        # 计算奖励
        self.rew_buf.zero_()
        for name, func in self.reward_functions.items():
            rew = func() * self.reward_scales[name] * self.dt
            self.rew_buf += rew

        # 终止判定 (掉下地面、翻滚过大或超时)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.projected_gravity[:, 2]) < 0.5  # 身体倾斜严重

        self._reset_idx(self.reset_buf)
        self._update_observation()
        self.last_actions.copy_(self.actions)

        return (
            TensorDict(
                {"policy": self.obs_buf}, batch_size=[self.num_envs], device=self.device
            ),
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _reset_idx(self, envs_idx):
        if envs_idx.any():
            # 这里简化处理，重置为初始姿态
            # 实际训练建议加入随机化（Randomization）
            self.robot.set_qpos(self.robot.get_qpos()[0], envs_idx=envs_idx)
            self.episode_length_buf[envs_idx] = 0
            # 重新采样指令
            self.commands[envs_idx, 0] = 0.5  # 默认给个前进 0.5m/s 的指令

    def _update_observation(self):
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 8
                self.dof_vel * self.obs_scales["dof_vel"],  # 8
                self.actions,  # 8
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._update_observation()
        return (
            TensorDict(
                {"policy": self.obs_buf}, batch_size=[self.num_envs], device=self.device
            ),
            None,
        )

    def get_observations(self):
        return TensorDict(
            {"policy": self.obs_buf}, batch_size=[self.num_envs], device=self.device
        )

    def get_privileged_observations(self):
        return None

    # --- 奖励函数区 ---
    def _reward_tracking_lin_vel(self):
        return torch.exp(
            -torch.sum(
                torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
            )
            / 0.25
        )

    def _reward_base_height(self):
        # 你的 XML 初始高度约 0.52，目标设为 0.35 比较稳
        return torch.square(self.robot.get_pos()[:, 2] - 0.35)
