import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from rc_wl_env import WLEnv  # 导入两轮两足机器人环境


def get_cfgs():
    # 环境配置
    env_cfg = {
        "num_actions": 6,
        "leg_joint_names": ["Lhleg", "Llleg", "Rhleg", "Rlleg"],
        "wheel_joint_names": ["Lwheel", "Rwheel"],
        "foot_link_names": ["Lleg_Lwheel", "Rlleg_Rwheel"],
        "default_leg_angles": {
            "Lhleg": 0.2,
            "Llleg": -1.0,
            "Rhleg": 0.2,
            "Rlleg": -1.0,
        },
        "default_wheel_angles": {"Lwheel": 0.0, "Rwheel": 0.0},
        "leg_kp": 40.0,  # 降低增益使控制更柔和
        "leg_kd": 1.0,
        "wheel_kp": 10.0,
        "wheel_kd": 0.5,
        "leg_max_torque": 20.0,
        "wheel_max_force": 10.0,
        "thigh_angle_min": -0.8,
        "thigh_angle_max": 1.2,
        "calf_angle_min": -2.0,
        "calf_angle_max": 0.65,
        "wheel_max_vel": 5.0,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 30,
        "termination_if_base_z_less_than": 0.30,  # 当基座高度低于0.15米时重置
        "base_init_pos": [0.0, 0.0, 0.55],  # 稍微提高初始高度
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "foot_ground_z_threshold": 0.06,
        "simulate_action_latency": True,
        "train_stage_1_steps": 24000,  # 第一阶段训练步数（速度0）
        "train_stage_2_speed": 0.5,   # 第二阶段速度范围（±0.3 m/s）
    }

    # 观测配置
    obs_cfg = {
        "num_obs": 29,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "projected_gravity": 1.0,
        },
    }

    # 奖励配置 - 重点是平衡奖励
    reward_cfg = {
        "reward_scales": {
            "tracking_lin_vel": 0.8,  # 初始阶段降低速度跟踪权重
            "wheel_vel_tracking": 0.8,  # 新增的轮子速度跟踪奖励
            "wheel_direction": 0.3,     # 轮子方向奖励
            "wheel_efficiency": 0.4,    # 轮子效率奖励  
            "gravity_projection": 3.0,  # 提高重力投影权重
            "action_symmetry": 5.0,  # 提高对称性权重
            "base_height": -3.0,
            "foot_contact": -0.8,     #轮子触地
            "jumping_penalty": -2.0
        },
        "tracking_sigma": 0.3,
        "gravity_sigma": 0.25,  # 添加重力投影sigma
        "symmetry_sigma": 0.25,  # 添加对称性sigma
        "base_height_target": 0.35,  # 与初始化高度一致
        "target_contact_count": 2,
    }

    # 命令配置 - 初始阶段只给零速度
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.0],  # 初始阶段不给速度
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        "ang_vel_max": 1.5,
        # 添加第二阶段命令范围
        "stage_2_lin_vel_x_range": [-1.0, 1.0],
        "stage_2_lin_vel_y_range": [0.0, 0.0],
        "stage_2_ang_vel_range": [0.0, 0.0],  # 0.3 * 3 = 0.9
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,  # 降低探索，专注于平衡
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 5e-4,  # 降低学习率，更稳定
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],  # 减小网络规模
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 0.5,  # 降低初始探索噪声
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24, #每次迭代是24step
        "save_interval": 50,  # 更频繁保存
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wl-walking")
    parser.add_argument(
        "-B", "--num_envs", type=int, default=8192
    )  # 两轮两足可适当减少并行环境数
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--show_viewer", action="store_true", help="Enable viewer")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # 清理并创建日志目录
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 保存配置文件
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # 创建两轮两足机器人环境
    env = WLEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,  # 传入可视化参数
    )

    # 初始化训练器并开始训练
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()
