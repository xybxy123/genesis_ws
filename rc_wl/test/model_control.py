import numpy as np
import os
import time
import argparse
from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE

# --- 1. 获取文件路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_relative_path = os.path.join("assets", "urdf", "wheel_leg.urdf")
urdf_path = os.path.join(current_dir, urdf_relative_path)
urdf_file_name = os.path.basename(urdf_path) 

print(f"尝试加载 URDF: {urdf_path}")

if not os.path.exists(urdf_path):
    print(f"❌ 错误: URDF 文件未找到于: {urdf_path}")
    exit()

# --- 添加命令行参数 ---
parser = argparse.ArgumentParser()
parser.add_argument("-dt", "--timestep", type=float, default=0.01, help="Simulation time step")
parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI", default=True)
parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument("-t", "--seconds", type=float, default=60, help="Number of seconds to simulate (default: 60)")
args = parser.parse_args()

########################## init ##########################
# 目标：GPU，回退到 CPU
try:
    gs.init(backend=gs.cpu if args.cpu else gs.cuda)
except Exception as e:
    print(f"⚠️ 无法初始化 CUDA 后端 ({e})。退回到 CPU 后端。")
    gs.init(backend=gs.cpu) # 回退到 CPU

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (1.0, 0, 0.8),
        camera_lookat = (0.0, 0.0, 0.0),
        camera_fov    = 45,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = args.timestep,
        gravity = np.array([0, 0, -9.8])
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=False,
    ),
    profiling_options=gs.options.ProfilingOptions(
        show_FPS=False,
    ),
    show_viewer = args.vis,
    renderer=gs.renderers.Rasterizer(),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
plane.pos = np.array([0.0, 0.0, -0.3]) 

wheel_leg = scene.add_entity(
    gs.morphs.URDF(
        file  = urdf_path,
    ),
)
wheel_leg.pos = np.array([0.0, 0.0, 0.5]) 
print(f"✅ 成功添加实体 ({urdf_file_name}). UID: {wheel_leg.uid}")

########################## 配置传感器 (IMU) ##########################
body_link = None
imu = None

try:
    # 1. 获取 body link
    all_links = scene.rigid_solver.links
    link_names = []
    for link in all_links:
        link_names.append(link.name)
        if link.name == 'body':
            body_link = link
            break
    
    if not body_link:
        print(f"⚠️ 未直接找到名称为'body'的 link，尝试使用机器人 base_link: {wheel_leg.base_link.name}")
        body_link = wheel_leg.base_link
        if not hasattr(body_link, 'name'):
            raise Exception(f"无法找到有效的 body link！可用 link 名称: {link_names}")
    
    print(f"✅ 成功获取目标 link: {body_link.name} (link uid: {body_link.uid})")

    # 2. 添加 IMU 传感器 (关键修正：设置 delay 和 jitter 为 0.0)
    print("\n📡 正在添加 IMU 传感器...")
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=wheel_leg.idx,
            link_idx_local=body_link.idx_local,
            pos_offset=(0.0, 0.0, 0.05),  # 在body上方5cm处
            # IMU噪声参数
            acc_cross_axis_coupling=(0.0, 0.01, 0.02),
            gyro_cross_axis_coupling=(0.03, 0.04, 0.05),
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.01, 0.01, 0.01),
            acc_random_walk=(0.001, 0.001, 0.001),
            gyro_random_walk=(0.001, 0.001, 0.001),
            # 关键修正：防止 division by zero 错误
            delay=0.0,
            jitter=0.0, 
            interpolate=True,
            # 可视化IMU位置
            draw_debug=True,
        )
    )
    print(f"✅ IMU 传感器已添加到 {body_link.name} link 上方5cm处")

except Exception as e:
    print(f"⚠️ IMU 配置失败！错误信息: {e}")
    imu = None




# ---------------------- 配置 IMU 数据记录和可视化 ----------------------
if imu is not None and args.vis:
    print("\n📊 配置 IMU 数据记录和可视化...")
    
    xyz = ("x", "y", "z")
    labels = {
        "lin_acc": xyz, "true_lin_acc": xyz,
        "ang_vel": xyz, "true_ang_vel": xyz
    }
    
    def data_func():
        if imu is None: return {}
        data = imu.read()
        true_data = imu.read_ground_truth()
        return {
            "lin_acc": data.lin_acc,
            "true_lin_acc": true_data.lin_acc,
            "ang_vel": data.ang_vel,
            "true_ang_vel": true_data.ang_vel,
        }
    
    if IS_PYQTGRAPH_AVAILABLE:
        scene.start_recording(
            data_func,
            gs.recorders.PyQtLinePlot(title="IMU Ground Truth vs Measured Data", labels=labels),
        )
        print("✅ 使用 pyqtgraph 进行实时 IMU 数据绘图")
    elif IS_MATPLOTLIB_AVAILABLE:
        gs.logger.info("pyqtgraph not found, falling back to matplotlib.")
        scene.start_recording(
            data_func,
            gs.recorders.MPLLinePlot(title="IMU Ground Truth vs Measured Data", labels=labels),
        )
        print("✅ 使用 matplotlib 进行实时 IMU 数据绘图")
    else:
        print("⚠️ matplotlib 或 pyqtgraph 未找到，跳过实时绘图")
    
    scene.start_recording(
        data_func=lambda: imu.read()._asdict() if imu else {},
        rec_options=gs.recorders.NPZFile(filename="wheel_leg_imu_data.npz"),
    )
    print("✅ IMU 数据将保存到: wheel_leg_imu_data.npz")

########################## build ##########################
scene.build()

# --- 关节控制配置 (这里已经使用了 dofs_idx_local，但警告可能仍在) ---
active_jnt_names = [
    'Lhleg', 'Llleg', 'Lwheel',
    'Rhleg', 'Rlleg', 'Rwheel'
]

try:
    active_dofs_idx = [wheel_leg.get_joint(name).dof_idx_local for name in active_jnt_names]
    num_active_dofs = len(active_dofs_idx)
    print(f"\n主动测试关节总数: {num_active_dofs}")
except Exception as e:
    print(f"❌ 错误: 无法获取主动关节索引。原始错误: {e}")
    exit()

# PD 增益和力限制
kp = np.array([3000, 3000, 500, 3000, 3000, 500])
kv = np.array([300, 300, 50, 300, 300, 50])
force_lower = np.array([-50, -50, -20, -50, -50, -20])
force_upper = np.array([50, 50, 20, 50, 50, 20])

wheel_leg.set_dofs_kp(kp=kp, dofs_idx_local=active_dofs_idx)
wheel_leg.set_dofs_kv(kv=kv, dofs_idx_local=active_dofs_idx)
wheel_leg.set_dofs_force_range(lower=force_lower, upper=force_upper, dofs_idx_local=active_dofs_idx)

# ---------------------- 1. 初始渐进归零 ----------------------
print("\n▶️ 初始渐进归零...")
for i in range(200):
    progress = min(1.0, i / 100.0)
    active_target_pos = np.array([0.2, 0.3, 0, 0.2, 0.3, 0]) * (1 - progress)
    wheel_leg.set_dofs_position(active_target_pos, dofs_idx_local=active_dofs_idx)
    scene.step()

print("⏳ 系统稳定中...")
for _ in range(100):
    wheel_leg.control_dofs_position(np.zeros(num_active_dofs), dofs_idx_local=active_dofs_idx)
    scene.step()

# ---------------------- 2. 定义主动测试关节的运动参数 ----------------------
test_joint_configs = [
    {"idx": 0, "name": "Lhleg", "min": -0.6, "max": 0.6, "duration": 200},
    {"idx": 1, "name": "Llleg", "min": -0.8, "max": 0.4, "duration": 200},
    {"idx": 2, "name": "Lwheel", "min": -6.0, "max": 6.0, "duration": 300},
    {"idx": 3, "name": "Rhleg", "min": -0.6, "max": 0.6, "duration": 200},
    {"idx": 4, "name": "Rlleg", "min": -0.8, "max": 0.4, "duration": 200},
    {"idx": 5, "name": "Rwheel", "min": -6.0, "max": 6.0, "duration": 300}
]

# ---------------------- 3. 核心逻辑：仅主动测试关节逐个运动 ----------------------
print("\n▶️ 启动6个主动测试关节独立运动演示...")
total_frame = 0

total_steps = sum([config["duration"] for config in test_joint_configs]) + 200 + 100 + 50 * 6 + 200
progress_bar = tqdm(total=total_steps, desc="Simulation Progress")

for _ in range(300):
    progress_bar.update(1)

try:
    for config in test_joint_configs:
        joint_idx = config["idx"]
        joint_name = config["name"]
        min_val = config["min"]
        max_val = config["max"]
        duration = config["duration"]
        
        print(f"\n📌 开始控制关节：{joint_name}（索引{joint_idx}）")
        
        for step in range(duration):
            progress = step / (duration - 1) if duration > 1 else 1.0
            current_target = min_val + progress * (max_val - min_val)
            
            active_target_pos = np.zeros(num_active_dofs)
            active_target_pos[joint_idx] = current_target
            
            wheel_leg.control_dofs_position(active_target_pos, dofs_idx_local=active_dofs_idx)
            
            if total_frame % 100 == 0:
                control_force = wheel_leg.get_dofs_control_force(dofs_idx_local=active_dofs_idx)
                internal_force = wheel_leg.get_dofs_force(dofs_idx_local=active_dofs_idx)
                
                print(f"\n=== 累计帧 {total_frame} ===")
                for act_idx, act_name, cf, if_ in zip(range(num_active_dofs), active_jnt_names, control_force, internal_force):
                    if act_idx == joint_idx:
                        print(f"{act_name:8s}: 控制力 = {cf:6.2f}, 实际力 = {if_:6.2f} 🟡（运动中）")
                    else:
                        print(f"{act_name:8s}: 控制力 = {cf:6.2f}, 实际力 = {if_:6.2f}（待命）")
                
                if imu is not None:
                    imu_data = imu.read()
                    imu_true = imu.read_ground_truth()
                    print("--- IMU 数据 (Body Link) ---")
                    # 修正后的代码 (使用 .cpu().numpy().round(3) 来确保兼容性)
                    print(f"线加速度 (测量): {imu_data.lin_acc.cpu().numpy().round(3)} m/s²")
                    print(f"线加速度 (真实): {imu_true.lin_acc.cpu().numpy().round(3)} m/s²")
                    print(f"角速度 (测量): {imu_data.ang_vel.cpu().numpy().round(3)} rad/s")
                    print(f"角速度 (真实): {imu_true.ang_vel.cpu().numpy().round(3)} rad/s")

            scene.step()
            total_frame += 1
            progress_bar.update(1)
        
        print(f"✅ 关节{joint_name}运动结束，停留0.5秒...")
        for _ in range(50):
            wheel_leg.control_dofs_position(active_target_pos, dofs_idx_local=active_dofs_idx)
            scene.step()
            total_frame += 1
            progress_bar.update(1)
    
    # ---------------------- 4. 演示结束，主动关节回归归零姿态 ----------------------
    print("\n🎯 所有测试关节运动完成，主动关节回归归零姿态...")
    for _ in range(200):
        wheel_leg.control_dofs_position(np.zeros(num_active_dofs), dofs_idx_local=active_dofs_idx)
        scene.step()
        total_frame += 1
        progress_bar.update(1)
    
    progress_bar.close()

except KeyboardInterrupt:
    progress_bar.close()
    print(f"\n⏸️ 用户中断程序，正在退出...")
finally:
    # 停止IMU数据记录
    if imu is not None:
        scene.stop_recording()
        print("\n📊 IMU 数据记录已停止")
        
        # 打印最终的IMU数据
        print("\n=== 最终 IMU 数据 ===")
        # 修正后的代码 (在 finally 块中)：
        # 打印最终 IMU 数据时，我们将 Tensor 移到 CPU 并转换为 NumPy 数组
        final_true = imu.read_ground_truth()
        final_measured = imu.read()

        print("Ground truth data:")
        print(f"Lin Acc: {final_true.lin_acc.cpu().numpy().round(3)}, Ang Vel: {final_true.ang_vel.cpu().numpy().round(3)}")
        print("Measured data:")
        print(f"Lin Acc: {final_measured.lin_acc.cpu().numpy().round(3)}, Ang Vel: {final_measured.ang_vel.cpu().numpy().round(3)}")

# --- 5. 保持Viewer窗口打开 ---
print("\n▶️ 仿真演示完毕。请手动关闭可视化窗口以退出程序。")
try:
    if hasattr(scene.visualizer, 'viewer') and scene.visualizer.viewer.is_open:
        viewer = scene.visualizer.viewer
        while viewer.is_open:
            scene.step()
            time.sleep(1.0/60.0)
    else:
        while True:
            scene.step()
            time.sleep(1.0/60.0)
except (AttributeError, KeyboardInterrupt):
    print(f"\n⏸️ 用户关闭窗口或中断程序，正在退出...")

# 清理资源
if imu is not None:
    scene.remove_sensor(imu)

print("\n✅ 程序正常退出。")