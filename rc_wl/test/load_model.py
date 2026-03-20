import genesis as gs
import os
import numpy as np
import time

# --- 1. 获取文件路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录: test/
# ✅ 关键修改：向上跳一级到 rc_wl 根目录
root_dir = os.path.dirname(current_dir)  # rc_wl 根目录
urdf_relative_path = os.path.join("assets", "urdf", "wheel_leg.urdf")
# ✅ 使用根目录拼接 URDF 路径，而非当前脚本目录
urdf_path = os.path.join(root_dir, urdf_relative_path)
urdf_file_name = os.path.basename(urdf_path)  # 获取文件名 wheel_leg.urdf

print(f"尝试加载 URDF: {urdf_path}")

if not os.path.exists(urdf_path):
    print(f"❌ 错误: URDF 文件未找到于: {urdf_path}")
    # 额外提示：帮助排查路径问题
    print(f"📌 当前脚本目录: {current_dir}")
    print(f"📌 期望的根目录: {root_dir}")
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


# --- 3. 添加地面和 URDF 实体 ---

# 3.1 添加地面
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# 3.2 添加轮腿机器人
wheel_leg_robot = scene.add_entity(
    gs.morphs.URDF(file=urdf_path),
)

# 设置初始位姿
initial_pos = [0.0, 0.0, 0.5]
wheel_leg_robot.pos = np.array(initial_pos)

# ✅ 修正行: 不再访问 wheel_leg_robot.name
print(f"✅ 成功添加实体 ({urdf_file_name}). UID: {wheel_leg_robot.uid}")

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
