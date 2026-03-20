# file: rc_wl_keyboard_eval_final.py
import argparse
import os
import pickle
import time
import sys

try:
    from importlib import metadata
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

import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from rc_wl_env import WLEnv


class KeyboardControl:
    """键盘控制类"""
    def __init__(self):
        try:
            from pynput import keyboard
            self.keyboard = keyboard
        except ImportError:
            print("警告：未安装pynput库，将使用简单的输入模式")
            print("请安装：pip install pynput")
            self.keyboard = None
        
        self.mode = "balance"  # 当前模式: balance, velocity
        self.target_vel_x = 0.0
        self.target_vel_y = 0.0
        self.target_ang_vel = 0.0
        
        # 控制参数
        self.vel_increment = 0.1  # 速度增量 (m/s)
        self.ang_vel_increment = 0.2  # 角速度增量 (rad/s)
        self.max_vel = 2.0  # 最大线速度
        self.max_ang_vel = 2.0  # 最大角速度
        
        # 按键状态
        self.key_states = {}
        self.special_keys_queue = []
        
        if self.keyboard:
            self.setup_keyboard_listener()
        else:
            print("键盘控制不可用，将使用预设速度")
        
        print("✓ 键盘控制初始化完成")
    
    def setup_keyboard_listener(self):
        """设置键盘监听器"""
        self.listener = self.keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
    
    def on_press(self, key):
        """按键按下"""
        try:
            key_char = key.char.lower()
            self.key_states[key_char] = True
            # 特殊键立即处理
            if key_char in ['1', '2', 'r', 'q']:
                self.special_keys_queue.append(key_char)
        except AttributeError:
            # 特殊键处理
            if key == self.keyboard.Key.up:
                self.key_states['up'] = True
            elif key == self.keyboard.Key.down:
                self.key_states['down'] = True
            elif key == self.keyboard.Key.left:
                self.key_states['left'] = True
            elif key == self.keyboard.Key.right:
                self.key_states['right'] = True
            elif key == self.keyboard.Key.space:
                self.key_states['space'] = True
            elif key == self.keyboard.Key.esc:
                self.special_keys_queue.append('q')
    
    def on_release(self, key):
        """按键释放"""
        try:
            key_char = key.char.lower()
            self.key_states[key_char] = False
        except AttributeError:
            if key == self.keyboard.Key.up:
                self.key_states['up'] = False
            elif key == self.keyboard.Key.down:
                self.key_states['down'] = False
            elif key == self.keyboard.Key.left:
                self.key_states['left'] = False
            elif key == self.keyboard.Key.right:
                self.key_states['right'] = False
            elif key == self.keyboard.Key.space:
                self.key_states['space'] = False
    
    def get_special_key(self):
        """获取特殊键"""
        if self.special_keys_queue:
            return self.special_keys_queue.pop(0)
        return None
    
    def is_pressed(self, key):
        """检查按键是否按下"""
        return self.key_states.get(key, False)
    
    def update(self):
        """更新控制状态"""
        # 检查特殊键
        special_key = self.get_special_key()
        if special_key:
            return self.handle_special_key(special_key)
        
        # 更新速度控制
        if self.mode == "balance":
            # 平衡模式：实时控制
            if self.is_pressed('up') or self.is_pressed('w'):
                self.target_vel_x += self.vel_increment
            if self.is_pressed('down') or self.is_pressed('s'):
                self.target_vel_x -= self.vel_increment
            if self.is_pressed('left') or self.is_pressed('a'):
                self.target_ang_vel += self.ang_vel_increment
            if self.is_pressed('right') or self.is_pressed('d'):
                self.target_ang_vel -= self.ang_vel_increment
            if self.is_pressed('space'):
                self.target_vel_x = 0.0
                self.target_ang_vel = 0.0
        else:
            # 速度跟踪模式：调整目标
            if self.is_pressed('up') or self.is_pressed('w'):
                self.target_vel_x += self.vel_increment
            if self.is_pressed('down') or self.is_pressed('s'):
                self.target_vel_x -= self.vel_increment
            if self.is_pressed('left') or self.is_pressed('a'):
                self.target_ang_vel += self.ang_vel_increment
            if self.is_pressed('right') or self.is_pressed('d'):
                self.target_ang_vel -= self.ang_vel_increment
            if self.is_pressed('space'):
                self.target_vel_x = 0.0
                self.target_ang_vel = 0.0
        
        # 限制速度范围
        self.target_vel_x = np.clip(self.target_vel_x, -self.max_vel, self.max_vel)
        self.target_ang_vel = np.clip(self.target_ang_vel, -self.max_ang_vel, self.max_ang_vel)
        
        # 更新信息文本
        self.info_text = f"模式: {self.mode} | 目标速度: {self.target_vel_x:.2f} m/s | 目标角速度: {self.target_ang_vel:.2f} rad/s"
        
        return None
    
    def handle_special_key(self, key):
        """处理特殊按键"""
        if key == '1':
            self.mode = "balance"
            self.target_vel_x = 0.0
            self.target_ang_vel = 0.0
            print("切换到平衡模式")
            return {'reset_commands': True}
        elif key == '2':
            self.mode = "velocity"
            print("切换到速度跟踪模式")
            return {'reset_commands': True}
        elif key == 'r':
            print("触发环境重置")
            return {'reset_env': True}
        elif key == 'q':
            print("退出程序")
            return {'quit': True}
        
        return None
    
    def print_controls(self):
        """打印控制说明"""
        print("\n" + "="*60)
        print("键盘控制说明")
        print("="*60)
        print("模式切换:")
        print("  [1] - 平衡模式")
        print("  [2] - 速度跟踪模式")
        print("  [r] - 重置环境")
        print("  [q] - 退出 (或ESC键)")
        print("\n速度控制:")
        print("  [↑]/[w] - 前进/增加速度")
        print("  [↓]/[s] - 后退/减少速度")
        print("  [←]/[a] - 左转/增加左转角速度")
        print("  [→]/[d] - 右转/增加右转角速度")
        print("  [空格]  - 停止/重置速度")
        print("="*60)
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'listener') and self.listener:
            self.listener.stop()


class WLRobotEvaluator:
    """两轮两足机器人评估器"""
    def __init__(self, model_path, config_path):
        """
        初始化评估器
        
        参数:
        - model_path: 模型文件路径
        - config_path: 配置文件路径
        """
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"错误：模型文件不存在: {model_path}")
            sys.exit(1)
        
        if not os.path.exists(config_path):
            print(f"错误：配置文件不存在: {config_path}")
            sys.exit(1)
        
        # 加载配置文件
        try:
            with open(config_path, 'rb') as f:
                cfgs = pickle.load(f)
                
            if len(cfgs) >= 4:
                env_cfg, obs_cfg, reward_cfg, command_cfg = cfgs[:4]
                if len(cfgs) >= 5:
                    train_cfg = cfgs[4]
                else:
                    train_cfg = None
            else:
                print("错误：配置文件格式不正确")
                sys.exit(1)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            sys.exit(1)
        
        # 评估时不需要奖励计算
        reward_cfg["reward_scales"] = {}
        
        # 创建环境
        self.env = WLEnv(
            num_envs=1,  # 评估时只需要一个环境
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,  # 启用可视化
        )
        
        # 键盘控制
        self.keyboard_control = KeyboardControl()
        
        # 加载模型
        self.model = self.load_model(model_path, train_cfg if train_cfg else None)
        
        # 数据记录
        self.telemetry = {
            'time': [],
            'actual_vel_x': [],
            'target_vel_x': [],
            'actual_ang_vel': [],
            'target_ang_vel': [],
            'pitch': [],
            'roll': [],
            'base_height': [],
            'mode': []
        }
        
        # 性能指标
        self.performance_metrics = {
            'balance_mode': {'total_time': 0, 'success_time': 0},
            'velocity_mode': {'total_time': 0, 'tracking_error': []}
        }
        
        # 运行状态
        self.running = True
        self.start_time = None
        
        print("✓ 评估器初始化完成")
    
    def load_model(self, model_path, train_cfg=None):
        """加载训练好的模型"""
        try:
            # 方法1：直接加载检查点并创建ActorCritic
            print("尝试方法1：直接加载模型检查点...")
            checkpoint = torch.load(model_path, map_location=gs.device)
            
            # 检查检查点结构
            print(f"检查点键: {checkpoint.keys() if isinstance(checkpoint, dict) else '不是字典'}")
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 尝试导入正确的ActorCritic类
                try:
                    from rsl_rl.modules import ActorCritic
                    
                    # 根据你的rsl_rl版本，参数可能不同
                    # 尝试不同的初始化方式
                    try:
                        # 新版本参数
                        model = ActorCritic(
                            num_actor_obs=self.env.num_obs,
                            num_critic_obs=self.env.num_obs,
                            num_actions=self.env.num_actions,
                            actor_hidden_dims=[256, 128, 64],
                            critic_hidden_dims=[256, 128, 64],
                            activation='elu',
                            init_noise_std=0.0
                        )
                    except TypeError as e:
                        print(f"新版本参数失败，尝试旧版本参数: {e}")
                        # 旧版本参数
                        model = ActorCritic(
                            num_obs=self.env.num_obs,
                            num_actions=self.env.num_actions,
                            actor_hidden_dims=[256, 128, 64],
                            critic_hidden_dims=[256, 128, 64],
                            activation='elu',
                            init_noise_std=0.0
                        )
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(gs.device)
                    model.eval()
                    
                    print(f"✓ 方法1成功: 直接加载模型")
                    return model
                    
                except Exception as e:
                    print(f"方法1失败: {e}")
            
            # 方法2：如果是完整的模型对象
            if hasattr(checkpoint, 'eval'):
                print("✓ 方法2：检查点本身就是模型")
                checkpoint.eval()
                return checkpoint.to(gs.device)
            
            # 方法3：使用训练器加载
            print("尝试方法3：使用训练器加载...")
            if train_cfg:
                log_dir = os.path.dirname(model_path)
                runner = OnPolicyRunner(self.env, train_cfg, log_dir, device=gs.device)
                runner.load(model_path)
                policy = runner.get_inference_policy(device=gs.device)
                print("✓ 方法3成功：从训练器加载")
                return policy
            
            print("所有方法都失败，将使用零动作")
            return None
            
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            print("将使用零动作进行演示")
            return None
    
    def run_evaluation(self):
        """运行评估"""
        # 打印控制说明
        self.keyboard_control.print_controls()
        
        # 初始化环境
        obs, _ = self.env.reset()
        self.start_time = time.time()
        last_telemetry_time = time.time()
        telemetry_interval = 0.1  # 10Hz记录数据
        last_display_time = time.time()
        display_interval = 0.1  # 10Hz更新显示
        
        print("\n开始评估，按 'q' 或 ESC 退出...\n")
        print("初始状态: 平衡模式，速度 0 m/s")
        
        try:
            while self.running:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # 更新键盘控制
                key_result = self.keyboard_control.update()
                
                # 处理键盘结果
                if key_result:
                    if 'quit' in key_result:
                        self.running = False
                        break
                    elif 'reset_env' in key_result:
                        obs, _ = self.env.reset()
                        print("环境已重置")
                        continue
                    elif 'reset_commands' in key_result:
                        # 重置命令范围
                        try:
                            envs_idx = torch.tensor([0], device=gs.device)
                            self.env._resample_commands(envs_idx)
                        except:
                            pass
                
                # 根据当前模式设置环境命令
                if self.keyboard_control.mode == "balance":
                    # 平衡模式：直接设置速度指令
                    self.env.lin_vel_x_range = [
                        self.keyboard_control.target_vel_x, 
                        self.keyboard_control.target_vel_x
                    ]
                    self.env.lin_vel_y_range = [0.0, 0.0]
                    self.env.ang_vel_range = [
                        self.keyboard_control.target_ang_vel, 
                        self.keyboard_control.target_ang_vel
                    ]
                    
                    # 更新性能指标
                    self.performance_metrics['balance_mode']['total_time'] += self.env.dt
                    if abs(self.env.base_euler[0, 1].item()) < 10:  # pitch角小于10度
                        self.performance_metrics['balance_mode']['success_time'] += self.env.dt
                        
                else:  # 速度跟踪模式
                    # 速度跟踪模式：设置目标速度
                    self.env.lin_vel_x_range = [
                        self.keyboard_control.target_vel_x, 
                        self.keyboard_control.target_vel_x
                    ]
                    self.env.lin_vel_y_range = [0.0, 0.0]
                    self.env.ang_vel_range = [
                        self.keyboard_control.target_ang_vel, 
                        self.keyboard_control.target_ang_vel
                    ]
                    
                    # 更新性能指标
                    self.performance_metrics['velocity_mode']['total_time'] += self.env.dt
                    if hasattr(self.env, 'base_lin_vel'):
                        vel_error = abs(self.keyboard_control.target_vel_x - self.env.base_lin_vel[0, 0].item())
                        self.performance_metrics['velocity_mode']['tracking_error'].append(vel_error)
                
                # 模型推理
                if self.model is not None:
                    try:
                        with torch.no_grad():
                            # 确保obs是tensor
                            if isinstance(obs, np.ndarray):
                                obs_tensor = torch.tensor(obs, device=gs.device, dtype=torch.float32)
                            else:
                                obs_tensor = obs
                            
                            # 调用模型
                            if hasattr(self.model, 'actor'):
                                actions = self.model.actor(obs_tensor)
                            elif hasattr(self.model, '__call__'):
                                actions = self.model(obs_tensor)
                            else:
                                actions = torch.zeros((1, self.env.num_actions), device=gs.device)
                            
                            # 确保动作是numpy数组
                            if torch.is_tensor(actions):
                                actions = actions.cpu().numpy()
                    except Exception as e:
                        print(f"模型推理错误: {e}")
                        actions = np.zeros((1, self.env.num_actions))
                else:
                    # 如果没有模型，使用零动作
                    actions = np.zeros((1, self.env.num_actions))
                
                # 环境步进 - 确保动作是numpy数组
                try:
                    obs, rews, dones, infos = self.env.step(actions)
                except TypeError as e:
                    # 如果env.step需要tensor，转换
                    if "clip() received an invalid combination of arguments" in str(e):
                        # 转换actions为tensor
                        actions_tensor = torch.tensor(actions, device=gs.device, dtype=torch.float32)
                        obs, rews, dones, infos = self.env.step(actions_tensor)
                    else:
                        raise
                
                # 记录遥测数据
                if current_time - last_telemetry_time >= telemetry_interval:
                    self.record_telemetry(elapsed_time)
                    last_telemetry_time = current_time
                
                # 显示状态信息
                if current_time - last_display_time >= display_interval:
                    self.display_status(elapsed_time)
                    last_display_time = current_time
                
                # 检查终止条件
                if dones[0]:
                    print("环境终止，正在重置...")
                    obs, _ = self.env.reset()
                
        except KeyboardInterrupt:
            print("\n评估被用户中断")
        except Exception as e:
            print(f"\n评估出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 显示最终性能统计
            if self.start_time:
                total_time = time.time() - self.start_time
                self.display_final_stats(total_time)
            
            # 保存遥测数据
            self.save_telemetry_data()
            
            # 清理资源
            self.cleanup()
    
    def record_telemetry(self, elapsed_time):
        """记录遥测数据"""
        if hasattr(self.env, 'base_lin_vel'):
            self.telemetry['time'].append(elapsed_time)
            self.telemetry['actual_vel_x'].append(self.env.base_lin_vel[0, 0].item())
            self.telemetry['target_vel_x'].append(self.keyboard_control.target_vel_x)
            self.telemetry['actual_ang_vel'].append(self.env.base_ang_vel[0, 2].item())
            self.telemetry['target_ang_vel'].append(self.keyboard_control.target_ang_vel)
            self.telemetry['pitch'].append(self.env.base_euler[0, 1].item())
            self.telemetry['roll'].append(self.env.base_euler[0, 0].item())
            self.telemetry['base_height'].append(self.env.base_pos[0, 2].item())
            self.telemetry['mode'].append(1 if self.keyboard_control.mode == "balance" else 2)
    
    def display_status(self, elapsed_time):
        """显示当前状态"""
        # 清屏
        print("\033[2J\033[H", end="")  # 清屏并移动光标到左上角
        
        # 显示标题
        print("="*60)
        print("两轮两足机器人 - 实时评估")
        print("="*60)
        
        # 显示模式
        mode_str = "平衡模式" if self.keyboard_control.mode == "balance" else "速度跟踪模式"
        print(f"模式: {mode_str}")
        
        if hasattr(self.env, 'base_lin_vel'):
            # 显示速度信息
            print(f"\n速度信息:")
            print(f"  目标速度: {self.keyboard_control.target_vel_x:.3f} m/s")
            print(f"  实际速度: {self.env.base_lin_vel[0, 0].item():.3f} m/s")
            print(f"  目标角速度: {self.keyboard_control.target_ang_vel:.3f} rad/s")
            print(f"  实际角速度: {self.env.base_ang_vel[0, 2].item():.3f} rad/s")
            
            # 显示姿态信息
            print(f"\n姿态信息:")
            print(f"  俯仰角(pitch): {self.env.base_euler[0, 1].item():.2f}°")
            print(f"  滚转角(roll): {self.env.base_euler[0, 0].item():.2f}°")
            print(f"  基座高度: {self.env.base_pos[0, 2].item():.3f} m")
            
            # 显示关节信息
            print(f"\n关节信息:")
            leg_names = ["左大腿", "左小腿", "右大腿", "右小腿"]
            wheel_names = ["左轮", "右轮"]
            
            for i in range(4):
                print(f"  {leg_names[i]}: {self.env.dof_pos[0, i].item():.3f} rad")
            
            for i in range(2):
                print(f"  {wheel_names[i]}: {self.env.dof_vel[0, 4+i].item():.3f} rad/s")
            
            # 显示性能指标
            print(f"\n性能指标:")
            if self.keyboard_control.mode == "balance":
                total_time = self.performance_metrics['balance_mode']['total_time']
                success_time = self.performance_metrics['balance_mode']['success_time']
                if total_time > 0:
                    success_rate = (success_time / total_time) * 100
                    print(f"  平衡成功率: {success_rate:.1f}%")
            else:
                if self.performance_metrics['velocity_mode']['tracking_error']:
                    avg_error = np.mean(self.performance_metrics['velocity_mode']['tracking_error'][-100:]) if len(self.performance_metrics['velocity_mode']['tracking_error']) > 0 else 0
                    print(f"  平均速度跟踪误差: {avg_error:.3f} m/s")
        else:
            print("\n环境状态不可用")
        
        # 显示运行时间
        print(f"\n运行时间: {elapsed_time:.1f} s")
        
        # 显示控制提示
        if hasattr(self.keyboard_control, 'info_text'):
            print(f"\n控制提示: {self.keyboard_control.info_text}")
        print("="*60)
    
    def display_final_stats(self, total_time):
        """显示最终统计信息"""
        print("\n" + "="*60)
        print("评估完成 - 性能统计")
        print("="*60)
        
        # 平衡模式统计
        balance_total = self.performance_metrics['balance_mode']['total_time']
        balance_success = self.performance_metrics['balance_mode']['success_time']
        
        if balance_total > 0:
            balance_rate = (balance_success / balance_total) * 100
            print(f"平衡模式:")
            print(f"  总时间: {balance_total:.1f} s")
            print(f"  成功时间: {balance_success:.1f} s")
            print(f"  成功率: {balance_rate:.1f} %")
        
        # 速度跟踪模式统计
        velocity_total = self.performance_metrics['velocity_mode']['total_time']
        velocity_errors = self.performance_metrics['velocity_mode']['tracking_error']
        
        if velocity_total > 0 and velocity_errors:
            avg_error = np.mean(velocity_errors)
            std_error = np.std(velocity_errors)
            print(f"\n速度跟踪模式:")
            print(f"  总时间: {velocity_total:.1f} s")
            print(f"  平均跟踪误差: {avg_error:.3f} m/s")
            print(f"  误差标准差: {std_error:.3f} m/s")
        
        print(f"\n总评估时间: {total_time:.1f} s")
        print("="*60)
    
    def save_telemetry_data(self):
        """保存遥测数据"""
        if not self.telemetry['time']:
            print("无遥测数据可保存")
            return
        
        # 创建保存目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"eval_telemetry/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为CSV
        import csv
        csv_file = f"{save_dir}/telemetry.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入标题
            headers = list(self.telemetry.keys())
            writer.writerow(headers)
            
            # 写入数据
            for i in range(len(self.telemetry['time'])):
                row = [self.telemetry[key][i] for key in headers]
                writer.writerow(row)
        
        print(f"\n遥测数据已保存到: {csv_file}")
        
        # 生成可视化图表
        self.generate_telemetry_plots(save_dir)
    
    def generate_telemetry_plots(self, save_dir):
        """生成遥测数据图表"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            fig.suptitle('两轮两足机器人遥测数据', fontsize=16)
            
            # 1. 速度跟踪
            axes[0, 0].plot(self.telemetry['time'], self.telemetry['target_vel_x'], 
                           'r--', label='目标速度', alpha=0.7)
            axes[0, 0].plot(self.telemetry['time'], self.telemetry['actual_vel_x'], 
                           'b-', label='实际速度', linewidth=1.5)
            axes[0, 0].set_xlabel('时间(s)')
            axes[0, 0].set_ylabel('速度(m/s)')
            axes[0, 0].set_title('速度跟踪')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 角速度跟踪
            axes[0, 1].plot(self.telemetry['time'], self.telemetry['target_ang_vel'], 
                           'r--', label='目标角速度', alpha=0.7)
            axes[0, 1].plot(self.telemetry['time'], self.telemetry['actual_ang_vel'], 
                           'b-', label='实际角速度', linewidth=1.5)
            axes[0, 1].set_xlabel('时间(s)')
            axes[0, 1].set_ylabel('角速度(rad/s)')
            axes[0, 1].set_title('角速度跟踪')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 姿态角度
            axes[1, 0].plot(self.telemetry['time'], self.telemetry['pitch'], 
                           'g-', label='俯仰角(pitch)', linewidth=1.5)
            axes[1, 0].set_xlabel('时间(s)')
            axes[1, 0].set_ylabel('角度(度)')
            axes[1, 0].set_title('俯仰角')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(self.telemetry['time'], self.telemetry['roll'], 
                           'r-', label='滚转角(roll)', linewidth=1.5)
            axes[1, 1].set_xlabel('时间(s)')
            axes[1, 1].set_ylabel('角度(度)')
            axes[1, 1].set_title('滚转角')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 4. 基座高度
            axes[2, 0].plot(self.telemetry['time'], self.telemetry['base_height'], 
                           'b-', linewidth=1.5)
            axes[2, 0].set_xlabel('时间(s)')
            axes[2, 0].set_ylabel('高度(m)')
            axes[2, 0].set_title('基座高度')
            axes[2, 0].grid(True, alpha=0.3)
            
            # 5. 模式变化
            axes[2, 1].step(self.telemetry['time'], self.telemetry['mode'], 
                           'k-', linewidth=1.5, where='post')
            axes[2, 1].set_xlabel('时间(s)')
            axes[2, 1].set_ylabel('模式')
            axes[2, 1].set_yticks([1, 2])
            axes[2, 1].set_yticklabels(['平衡', '速度'])
            axes[2, 1].set_title('控制模式')
            axes[2, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/telemetry_plots.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"遥测图表已保存到: {save_dir}/telemetry_plots.png")
            
        except ImportError:
            print("未安装matplotlib，跳过图表生成")
        except Exception as e:
            print(f"生成图表失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.keyboard_control.cleanup()


def main():
    parser = argparse.ArgumentParser(description='两轮两足机器人键盘控制评估')
    parser.add_argument('--model', type=str, required=True, 
                       help='模型文件路径 (如: logs/wl-walking/model_1999.pt)')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (如: logs/wl-walking/cfgs.pkl)')
    
    args = parser.parse_args()
    
    # 初始化genesis
    gs.init(logging_level="warning")
    
    try:
        # 创建评估器
        evaluator = WLRobotEvaluator(args.model, args.config)
        
        # 运行评估
        evaluator.run_evaluation()
        
    except Exception as e:
        print(f"评估出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()