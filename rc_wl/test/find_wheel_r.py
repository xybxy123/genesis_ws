#!/usr/bin/env python3
"""
【精简版 V10.0 - 绝对路径修复】计算 wheel_leg.urdf 的 foot_ground_z_threshold
- 目标：获取 URDF 轮子半径并计算阈值
- 修复：直接使用用户提供的绝对路径，解决文件找不到的问题。
"""
import os
import sys
import numpy as np
import torch
import time
import genesis as gs

# ===================== 仅需修改的配置项 =====================
# --- 路径配置 (【关键修改】：使用绝对路径) ---
# 使用用户提供的正确绝对路径
URDF_FILE_PATH = "/home/xybxy123/w_work/genesis_ws/rc_wl/assets/urdf/wheel_leg.urdf"

# --- 阈值配置 ---
WHEEL_RADIUS = 0.05  # 轮子半径（米） - 【请根据您的URDF核对】
TOLERANCE = 0.02  # 容错值（米）

# --- 仿真配置 ---
USE_CPU = True  # True=CPU运行，False=GPU运行
# =============================================================

# 初始化类型占位符
IdentityQuat = np.array(
    [1.0, 0.0, 0.0, 0.0], dtype=np.float32
)  # 使用 NumPy 四元数 (w, x, y, z)


def calculate_wheel_radius_and_threshold():
    """主执行逻辑：初始化 Genesis, 加载 URDF 实体，并计算阈值。"""

    print("\n" + "=" * 60)
    print("📢 提示：URDF轮子半径需手动核对和配置！")
    print("=" * 60)
    print(f"尝试加载 URDF: {URDF_FILE_PATH}")

    if not os.path.exists(URDF_FILE_PATH):
        # 如果文件仍未找到，可能是路径输入错误或权限问题
        print(f"❌ 严重错误: 即使使用绝对路径，URDF 文件仍未找到于: {URDF_FILE_PATH}")
        sys.exit(1)

    scene = None
    try:
        # --- 1. 初始化 Genesis 和创建场景 ---
        print("🔧 调用 gs.init() 初始化Genesis引擎...")

        # 使用最简化的 init 调用
        backend = gs.cpu if USE_CPU else gs.cuda
        try:
            gs.init(backend=backend)
        except TypeError as e:
            if "unexpected keyword argument 'backend'" in str(e):
                print("⚠️ 尝试不带任何参数调用 gs.init()。")
                gs.init()
            else:
                raise

        # 创建场景，使用最简化参数
        scene = gs.Scene(
            show_viewer=True,  # 开启 Viewer
        )

        # --- 2. 添加地面和 URDF 实体 ---

        # 2.1 添加地面
        if hasattr(gs.morphs, "Plane"):
            plane = scene.add_entity(gs.morphs.Plane())
        else:
            print("⚠️ 忽略添加地面：gs.morphs 缺少 Plane 接口。")

        # 2.2 添加轮腿机器人 (使用 gs.morphs.URDF 接口)
        if not hasattr(gs.morphs, "URDF"):
            raise AttributeError(
                "❌ Genesis模块缺少 gs.morphs.URDF 接口，无法加载模型。"
            )

        wheel_leg_robot = scene.add_entity(
            gs.morphs.URDF(file=URDF_FILE_PATH),
        )

        # 设置初始位姿
        initial_pos_z = WHEEL_RADIUS + TOLERANCE
        initial_pos = [0.0, 0.0, initial_pos_z]
        wheel_leg_robot.pos = np.array(initial_pos)

        urdf_file_name = os.path.basename(URDF_FILE_PATH)
        print(f"✅ 成功添加实体 ({urdf_file_name}). UID: {wheel_leg_robot.uid}")

        # --- 3. 构建场景并开始仿真循环（可选，用于验证加载） ---
        if hasattr(scene, "build"):
            scene.build()
        else:
            print("⚠️ Scene 对象缺少 build 方法。")

        duration_seconds = 2
        fps = 60
        num_steps = duration_seconds * fps
        print(f"\n▶️ 运行 {duration_seconds} 秒仿真 ({num_steps} 步)...")

        for i in range(num_steps):
            if hasattr(scene, "step"):
                scene.step()
                time.sleep(1.0 / 60.0)
            else:
                print("⚠️ Scene 对象缺少 step 方法，跳过仿真循环。")
                break

        # --- 4. 计算核心阈值 ---
        foot_ground_z_threshold = WHEEL_RADIUS + TOLERANCE

        # 5. 打印结果
        print("\n" + "=" * 70)
        print("✅ 成功加载 URDF 实体并计算阈值。")
        print(f"🔧 配置的轮子半径：{WHEEL_RADIUS:.6f} 米")
        print(f"🔧 配置的容错值：{TOLERANCE:.6f} 米")
        print("---")
        print(f"📌 最终推荐 foot_ground_z_threshold = {foot_ground_z_threshold:.6f} 米")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 脚本执行失败：{type(e).__name__} - {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # --- 6. 清理 ---
        print("\n⏸️ 仿真结束。")
        if scene and hasattr(scene, "destroy"):
            scene.destroy()

        if hasattr(gs, "destroy"):
            gs.destroy()
            print("✅ 仿真场景已清理。")
        else:
            print("✅ 脚本执行完成。")

    return WHEEL_RADIUS


if __name__ == "__main__":
    calculate_wheel_radius_and_threshold()
