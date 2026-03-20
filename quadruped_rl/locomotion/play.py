import torch
import genesis as gs
from envs.dog_env import DogEnv

def main():
    gs.init(backend=gs.gpu)
    # 获取与训练一致的配置
    from train import get_cfgs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env = DogEnv(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True
    )

    # 这里的路径需要指向你训练生成的 model_xxx.pt
    # policy = torch.load('logs/dog_walking/model_200.pt') 
    
    obs, _ = env.reset()
    while True:
        # 没模型时先用随机动作测试环境显示
        actions = torch.zeros((1, 8), device=gs.device) 
        obs, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    main()