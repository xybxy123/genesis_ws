import genesis as gs
import os
import numpy as np
import time

# --- 1. 获取文件路径 ---
# 使用 genesis-env 环境加载 JQG12 机器人模型
mjcf_path = "/home/xybxy123/w_work/genesis_ws/quadruped_rl/assets/mujoco_menagerie/unitree_go2/go2_mjx.xml"
mjcf_file_name = os.path.basename(mjcf_path)  # 获取文件名 JQG12_straight.xml

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
    ),
    renderer=gs.renderers.Rasterizer(),
)


# --- 3. 添加地面和 JQG12 机器人 ---

# 3.1 添加地面
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# 3.2 添加 JQG12 机器人
jqg12_robot = scene.add_entity(
    gs.morphs.MJCF(file=mjcf_path),
)

# 设置初始位姿
initial_pos = [0.0, 0.0, 0.5]
jqg12_robot.pos = np.array(initial_pos)

# ✅ 成功添加实体
print(f"✅ 成功添加实体 ({mjcf_file_name}). UID: {jqg12_robot.uid}")

# --- 4. 构建场景并开始仿真循环 ---

scene.build()

# 运行 10 秒的模拟循环
duration_seconds = 10
fps = 60
num_steps = duration_seconds * fps
print(f"\n▶️ 运行 {duration_seconds} 秒仿真 ({num_steps} 步)...")

for i in range(num_steps):
    scene.step()
    time.sleep(1.0 / 60.0)

# --- 5. 清理 ---
print("\n⏸️ 仿真结束。")
