"""
Go2 四足机器人 独立控制主程序
【修复：站立不会趴，只有走路才用IK】
"""

import os
import warnings
import genesis as gs
import numpy as np
import time
from keyboard_input import KeyboardInput
from foot_trajectory_generate import generate_foot_trajectory
from foot_ik import foot_ik
from motor_drive import apply_motor_commands

# ===================== 固定配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MJCF_PATH = os.path.join(
    PROJECT_ROOT, "assets", "mujoco_menagerie", "unitree_go2", "go2_mjx.xml"
)
MODEL_PATH = MJCF_PATH
CONTROL_DT = 1.0 / 60.0

warnings.filterwarnings("ignore", message=".*torch<2.8.0.*")
gs.init(backend=gs.gpu, logging_level="warning")

# 步态库
GAITS = {
    "stand": {
        "name": "站立",
        "phase_offsets": [0.0, 0.0, 0.0, 0.0],
        "step_len": 0.0,
        "step_h": 0.0,
        "period": 1.0,
    },
    "trot": {
        "name": "对角小跑",
        "phase_offsets": [0.0, 0.5, 0.5, 0.0],
        "step_len": 0.08,
        "step_h": 0.04,
        "period": 0.4,
    },
    "walk": {
        "name": "爬行",
        "phase_offsets": [0.0, 0.25, 0.5, 0.75],
        "step_len": 0.06,
        "step_h": 0.035,
        "period": 0.8,
    },
    "amble": {
        "name": "踱步",
        "phase_offsets": [0.0, 0.0, 0.5, 0.5],
        "step_len": 0.07,
        "step_h": 0.04,
        "period": 0.5,
    },
}

JOINT_NAMES = [
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

# 你给的绝对正确站立姿势
STAND_POS = np.array(
    [0.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.1, -2.3, -2.3, -2.3, -2.3], dtype=np.float32
)

HIP_RANGE = (-1.0472, 1.0472)
THIGH_RANGE = (-1.5708, 3.4907)
CALF_RANGE = (-2.7227, -0.83776)


# ===================== 主控制器 =====================
class Go2Control:
    def __init__(self):
        print(f"加载模型: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ 错误: 模型文件不存在")
            exit(1)

        self.scene = gs.Scene(
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

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.go2_robot = self.scene.add_entity(gs.morphs.MJCF(file=MODEL_PATH))
        self.go2_robot.pos = np.array([0.0, 0.0, 0.5])
        self.scene.build()

        joint_info = {
            joint.name: {"dofs_idx": joint.dofs_idx} for joint in self.go2_robot.joints
        }
        self.control_dofs_idx = np.array(
            [joint_info[name]["dofs_idx"][0] for name in JOINT_NAMES], dtype=int
        )

        self.go2_robot.set_dofs_kp(np.full(12, 55.0), self.control_dofs_idx)
        self.go2_robot.set_dofs_kv(np.full(12, 5.0), self.control_dofs_idx)

        self.kb = KeyboardInput()
        self.kb.start()

        self.gait_names = list(GAITS.keys())
        self.current_gait = "stand"
        self.phase = 0.0
        self.vx = self.vy = self.vyaw = 0.0
        self.should_exit = False
        self.debug_ik = True
        self.debug_step = 0

        self.stand_stable()
        if self.debug_ik:
            self.debug_ik_report()

        stand_sample = foot_ik(np.array([0.0, 0.0465, -0.2]), 0)
        stand_triplet = np.array(
            [STAND_POS[0], STAND_POS[4], STAND_POS[8]], dtype=np.float32
        )
        self.ik_joint_offset = stand_triplet - stand_sample
        print(f"IK offset: {self.ik_joint_offset}")
        print("===== Go2 已站稳 | 空格切换步态 | W/S/A/D/Q/E 控制 =====")

    def stand_stable(self):
        print("✅ 进入标准站立姿态")
        for _ in range(int(2.0 / CONTROL_DT)):
            apply_motor_commands(
                self.go2_robot, self.control_dofs_idx, STAND_POS, mode="position"
            )
            self.scene.step()

    def switch_gait(self):
        idx = self.gait_names.index(self.current_gait)
        idx = (idx + 1) % len(self.gait_names)
        self.current_gait = self.gait_names[idx]
        print(f"[步态] {GAITS[self.current_gait]['name']}")

    def debug_ik_report(self):
        print("\n===== IK Debug Report =====")
        print("STAND_POS:", STAND_POS)

        sample_targets = [
            ("FL", np.array([0.0, 0.0465, -0.2])),
            ("FR", np.array([0.0, -0.0465, -0.2])),
            ("RL", np.array([0.0, 0.0465, -0.2])),
            ("RR", np.array([0.0, -0.0465, -0.2])),
        ]
        for idx, (name, target) in enumerate(sample_targets):
            angles = foot_ik(target, idx)
            print(f"{name} target={target} -> ik={angles}")
        print("===========================\n")

    def update_input(self):
        key = self.kb.get_key()
        if not key:
            return

        if key == "\x1b":
            self.should_exit = True
            return

        if key == " ":
            self.switch_gait()

        elif key == "w":
            self.vx = 1.0
        elif key == "s":
            self.vx = -1.0
        elif key == "a":
            self.vy = 1.0
        elif key == "d":
            self.vy = -1.0
        elif key == "q":
            self.vyaw = -1.0
        elif key == "e":
            self.vyaw = 1.0
        else:
            self.vx = self.vy = self.vyaw = 0.0

    def step_control(self):
        # ===================== 关键修复 =====================
        if self.current_gait == "stand":
            # 站立时：强制用你的 STAND_POS，绝对不趴
            apply_motor_commands(self.go2_robot, self.control_dofs_idx, STAND_POS)
            return

        # 只有非站立步态才走 IK
        g = GAITS[self.current_gait]
        self.phase += CONTROL_DT / g["period"]
        self.phase %= 1.0

        foot_targets = np.zeros((4, 3))
        for leg in range(4):
            leg_phase = (self.phase + g["phase_offsets"][leg]) % 1.0
            side_sign = 1 if leg in (0, 2) else -1

            x, y, z = generate_foot_trajectory(
                phase=leg_phase,
                step_length=g["step_len"] * self.vx,
                step_height=g["step_h"],
                side_offset=0.0465 * side_sign + self.vy * 0.03,
                turn_offset=self.vyaw * 0.02 * (-1 if leg < 2 else 1),
                z_default=-0.2,
            )
            foot_targets[leg] = [x, y, z]

        q_target = np.zeros(12)
        for leg in range(4):
            angles = foot_ik(foot_targets[leg], leg)
            q_target[leg] = angles[0]
            q_target[leg + 4] = angles[1]
            q_target[leg + 8] = angles[2]

        # 用站立姿态做零偏标定，并限制到 XML 定义的关节范围
        for leg in range(4):
            q_target[leg] += self.ik_joint_offset[0]
            q_target[leg + 4] += self.ik_joint_offset[1]
            q_target[leg + 8] += self.ik_joint_offset[2]

        q_target[0:4] = np.clip(q_target[0:4], *HIP_RANGE)
        q_target[4:8] = np.clip(q_target[4:8], *THIGH_RANGE)
        q_target[8:12] = np.clip(q_target[8:12], *CALF_RANGE)

        apply_motor_commands(self.go2_robot, self.control_dofs_idx, q_target)

        if self.debug_ik and self.debug_step % 60 == 0:
            print(f"[IK] gait={self.current_gait} phase={self.phase:.3f}")
            for leg in range(4):
                print(
                    f"  leg{leg} target={foot_targets[leg]} angles={q_target[leg::4]}"
                )
            print(f"  q_target={q_target}")
        self.debug_step += 1

    def run(self):
        while not self.should_exit:
            step_start = time.time()
            self.update_input()
            self.step_control()
            self.scene.step()
            until = step_start + CONTROL_DT
            time.sleep(max(0, until - time.time()))
        self.kb.stop()


if __name__ == "__main__":
    controller = Go2Control()
    controller.run()
