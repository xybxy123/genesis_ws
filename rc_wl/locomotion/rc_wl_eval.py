import argparse
import os
import pickle
from importlib import metadata

import torch

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

# 导入两轮两足机器人环境
from rc_wl_env import WLEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_name", type=str, default="wl-robot-walking"
    )  # 实验名称默认值调整为两轮两足机器人
    parser.add_argument("--ckpt", type=int, default=100)  # 模型检查点编号
    args = parser.parse_args()

    # 初始化genesis仿真引擎
    gs.init()

    # 日志和配置文件路径
    log_dir = f"logs/{args.exp_name}"
    # 加载训练时保存的配置文件
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    # 评估时不需要奖励计算，清空奖励权重
    reward_cfg["reward_scales"] = {}

    # 创建两轮两足机器人环境实例
    env = WLEnv(
        num_envs=1,  # 评估时只需要一个环境
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,  # 启用可视化
    )

    # 初始化OnPolicyRunner用于加载模型
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    # 模型检查点路径
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    # 加载训练好的模型
    runner.load(resume_path)
    # 获取推理用的策略
    policy = runner.get_inference_policy(device=gs.device)

    # 重置环境获取初始观测
    obs, _ = env.reset()
    # 无梯度执行推理循环
    with torch.no_grad():
        while True:  # 持续运行直到手动终止
            # 策略根据观测生成动作
            actions = policy(obs)
            # 环境执行动作并获取新的观测、奖励、终止信号等
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
