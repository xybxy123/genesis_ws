import argparse
import os
import pickle
import genesis as gs
# 关键修改：适配你的目录结构
from locomotion.dog_env import DogEnv
from rsl_rl.runners import OnPolicyRunner

def get_cfgs():
    env_cfg = {
        "num_actions": 8,
        "joint_names": ["h_back_right", "l_back_right", "h_front_right", "l_front_right", "h_back_left", "l_back_left", "h_front_left", "l_front_left"],
        "default_joint_angles": {n: 0.0 for n in ["h_back_right", "l_back_right", "h_front_right", "l_front_right", "h_back_left", "l_back_left", "h_front_left", "l_front_left"]},
        "kp": 30.0, "kd": 1.0,
        "base_init_pos": [0, 0, 0.52], "base_init_quat": [1, 0, 0, 0],
        "episode_length_s": 20.0, "action_scale": 0.5, "clip_actions": 10.0,
    }
    obs_cfg = {"num_obs": 33, "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}}
    reward_cfg = {"reward_scales": {"tracking_lin_vel": 1.0, "base_height": -10.0}}
    command_cfg = {"num_commands": 3}
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    gs.init(backend=gs.gpu)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # 关键修改：日志目录适配你的路径（改为当前目录下的logs）
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs/dog_walking")
    os.makedirs(log_dir, exist_ok=True)

    # 可选：如果显存不足，减小num_envs（比如从4096改为1024）
    env = DogEnv(
        num_envs=1024,  # 优先推荐1024，显存足够再改回4096
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        show_viewer=True  # 训练时关闭可视化，调试时改为True
    )

    train_cfg = {
        "algorithm": {
            "class_name": "PPO", 
            "clip_param": 0.2, 
            "entropy_coef": 0.01, 
            "learning_rate": 1e-3, 
            "num_learning_epochs": 5, 
            "num_mini_batches": 4, 
            "value_loss_coef": 1.0
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution"
            }
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu"
        },
        "max_iterations": 200, 
        "save_interval": 50, 
        "experiment_name": "dog_test", 
        "run_name": "run1",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
    }

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=200, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
