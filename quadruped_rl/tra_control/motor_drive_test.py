import genesis as gs
import os
import numpy as np
import time

from motor_drive import apply_motor_commands

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MJCF_PATH = os.path.join(
    PROJECT_ROOT, "assets", "mujoco_menagerie", "unitree_go2", "go2_mjx.xml"
)

print(f"加载模型: {MJCF_PATH}")

if not os.path.exists(MJCF_PATH):
    print(f"❌ 错误: 模型文件不存在")
    exit()

# --- 初始化 Genesis ---
gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.0, 3.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=45,
        max_FPS=60,
        enable_default_keybinds=True,
    ),
    renderer=gs.renderers.Rasterizer(),
)

# --- 添加地面 & 机器人 ---
plane = scene.add_entity(gs.morphs.Plane())
go2_robot = scene.add_entity(gs.morphs.MJCF(file=MJCF_PATH))
go2_robot.pos = np.array([0.0, 0.0, 0.5])

# --- 构建场景 ---
scene.build()

# --- 获取关节索引 ---
joint_info = {joint.name: {"dofs_idx": joint.dofs_idx} for joint in go2_robot.joints}

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

# ==============================================
home_pos = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8], dtype=np.float32
)
stand_pos = np.array(
    [0.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.1, -2.3, -2.3, -2.3, -2.3], dtype=np.float32
)
squat_pos = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, -1.3, -1.3, -1.3, -1.3], dtype=np.float32
)
lift_fl_pos = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0, -0.8, -2.0, -2.0, -2.0], dtype=np.float32
)

action_list = [home_pos, stand_pos, squat_pos, lift_fl_pos]
action_name = ["Home姿态", "标准站立", "蹲伏", "抬左腿"]

# --- PD 参数 ---
go2_robot.set_dofs_kp(np.full(12, 50.0), control_dofs_idx)
go2_robot.set_dofs_kv(np.full(12, 4.0), control_dofs_idx)

# --- 循环动作 ---
hold_time = 3
fps = 60
step_per_action = hold_time * fps

print("\n🔄 开始循环动作...")

while True:
    for i, target in enumerate(action_list):
        print(f"→ {action_name[i]}")

        apply_motor_commands(
            robot=go2_robot,
            joint_indices=control_dofs_idx,
            target_positions=target,
            mode="position",
        )

        for _ in range(step_per_action):
            scene.step()
