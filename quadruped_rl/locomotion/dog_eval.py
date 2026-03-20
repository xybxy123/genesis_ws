import argparse
import os
import pickle
import torch
import genesis as gs
from envs.dog_env import DogEnv
from rsl_rl.runners import OnPolicyRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dog_walking")
    parser.add_argument("--ckpt", type=int, default=100) # 想要测试的检查点编号
    args = parser.parse_args()

    # 初始化 Genesis (评估时建议用 GPU，渲染更流畅)
    gs.init(backend=gs.gpu)

    log_dir = f"logs/{args.exp_name}"
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")

    # 1. 加载训练时保存的配置参数
    if not os.path.exists(cfgs_path):
        raise FileNotFoundError(f"找不到配置文件: {cfgs_path}，请先运行 train.py")
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))
    
    # 评估时不需要计算奖励逻辑，清空以节省开销
    reward_cfg["reward_scales"] = {}

    # 2. 创建单体评估环境 (开启渲染)
    env = DogEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # 3. 加载 PPO 策略
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    
    if not os.path.exists(resume_path):
        print(f"警告: 找不到模型文件 {resume_path}，将使用随机初始化的策略。")
    else:
        runner.load(resume_path)
    
    policy = runner.get_inference_policy(device=gs.device)

    # 4. 运行推理循环
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            # 神经网络根据当前观测输出动作
            actions = policy(obs)
            # 执行动作并获取下一帧观测
            obs, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    main()