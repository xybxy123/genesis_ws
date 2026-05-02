import genesis as gs
import os
import numpy as np
import time

# --- 1. 获取文件路径 ---
mjcf_path = "/home/xybxy123/w_work/genesis_ws/quadruped_rl/assets/mujoco_menagerie/unitree_go2/go2_mjx.xml"
mjcf_file_name = os.path.basename(mjcf_path)

print(f"尝试加载 MJCF: {mjcf_path}")

if not os.path.exists(mjcf_path):
    print(f"❌ 错误: MJCF 文件未找到于: {mjcf_path}")
    print(f"📌 请确保在 genesis-env 环境中运行此脚本")
    exit()


# --- 2. 初始化 Genesis 和创建场景 ---
gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.0, 3.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=45,
        max_FPS=60,
        enable_default_keybinds=True,  # 开启所有默认快捷键
    ),
    renderer=gs.renderers.Rasterizer(),
)


# --- 3. 添加地面和机器人 ---

# 3.1 添加地面
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# 3.2 添加 Unitree Go2 机器人
go2_robot = scene.add_entity(
    gs.morphs.MJCF(file=mjcf_path),
)

# 设置初始位姿
initial_pos = [0.0, 0.0, 0.5]
go2_robot.pos = np.array(initial_pos)

print(f"✅ 成功添加实体 ({mjcf_file_name}). UID: {go2_robot.uid}")

# --- 4. 构建场景 ---
scene.build()

# --- 5. 设置关节初始态 ---
# 打印可用的自由度信息
print("\n📋 机器人关节详细信息:")
print(f"总自由度数: {go2_robot.n_dofs}")

# 遍历所有关节，获取索引信息
joint_info = {}
for joint in go2_robot.joints:
    print(
        f"  关节名: {joint.name:20s} | DoF索引: {joint.dofs_idx} | DoF数: {joint.n_dofs}"
    )
    joint_info[joint.name] = {
        "dofs_idx": joint.dofs_idx,
        "n_dofs": joint.n_dofs,
    }

# 只控制 12 个腿关节，不控制 base
control_joint_names = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

control_dofs_idx = [joint_info[name]["dofs_idx"][0] for name in control_joint_names]
target_dof_pos = np.array(
    [0.0, 0.0, 0.0, 0.0, 1.16, 1.16, 1.16, 1.16, -2.72, -2.72, -2.72, -2.72],
    dtype=np.float32,
)

print(f"\n🎯 目标关节角度设置(仅腿关节):")
print(f"  hip:   0.0 rad")
print(f"  thigh: 1.16 rad")
print(f"  calf:  -2.72 rad")

# 先把腿关节放到目标姿态，作为初始态
go2_robot.set_dofs_position(
    target_dof_pos, dofs_idx_local=control_dofs_idx, zero_velocity=True
)

# 再设置 PD 增益并启动位置控制，保持姿态
go2_robot.set_dofs_kp(
    np.full(len(control_dofs_idx), 80.0, dtype=np.float32),
    dofs_idx_local=control_dofs_idx,
)
go2_robot.set_dofs_kv(
    np.full(len(control_dofs_idx), 5.0, dtype=np.float32),
    dofs_idx_local=control_dofs_idx,
)
go2_robot.control_dofs_position(target_dof_pos, dofs_idx_local=control_dofs_idx)

print(f"\n✅ 关节控制已启动")

# --- 6. 运行仿真循环 ---
duration_seconds = 10
fps = 60
num_steps = duration_seconds * fps
print(f"\n▶️ 运行 {duration_seconds} 秒仿真 ({num_steps} 步)...")
print(f"运行频率: {fps} FPS")

for i in range(num_steps):
    scene.step()

    # 每秒打印一次关节状态
    if i % fps == 0:
        current_pos = go2_robot.get_dofs_position().detach().cpu().numpy().reshape(-1)
        print(
            f"[{i//fps}s] "
            f"FL=({current_pos[6]:.3f}, {current_pos[10]:.3f}, {current_pos[14]:.3f}) "
            f"FR=({current_pos[7]:.3f}, {current_pos[11]:.3f}, {current_pos[15]:.3f})"
        )
        print(
            f"      RL=({current_pos[8]:.3f}, {current_pos[12]:.3f}, {current_pos[16]:.3f}) "
            f"RR=({current_pos[9]:.3f}, {current_pos[13]:.3f}, {current_pos[17]:.3f})"
        )

# --- 7. 清理 ---
print("\n⏸️ 仿真结束。")
print("✅ 关节控制测试完成。")
