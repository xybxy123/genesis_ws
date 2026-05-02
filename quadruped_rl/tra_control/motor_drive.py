"""
motor_drive.py
功能: 电机驱动层接口 — 将关节目标位置/速度转换为机器人实体的控制命令并下发。
"""

import numpy as np
from typing import Sequence, Optional


def apply_motor_commands(
    robot,
    joint_indices: Sequence[int],
    target_positions: Sequence[float],
    target_velocities: Optional[Sequence[float]] = None,
    mode: str = "position",
):
    jp = np.asarray(target_positions, dtype=float)
    if mode == "position":
        if hasattr(robot, "control_dofs_position"):
            robot.control_dofs_position(jp, joint_indices)
            return
    if mode == "velocity":
        if hasattr(robot, "control_dofs_velocity"):
            if target_velocities is None:
                raise ValueError("velocity 模式需要提供 target_velocities")
            robot.control_dofs_velocity(
                np.asarray(target_velocities, dtype=float), joint_indices
            )
            return

    if hasattr(robot, "control_dofs"):
        robot.control_dofs(jp, joint_indices)
        return

    # Genesis 机器人对象或 mujoco.MjData：直接设置目标位置
    if hasattr(robot, "qpos"):
        idx = np.asarray(joint_indices, dtype=int)
        joint_indices_np = np.asarray(joint_indices, dtype=int)
        # 设置目标位置到 qpos 属性
        if hasattr(robot, "dofs_target_pos"):
            # Genesis 机器人的 dofs_target_pos 属性
            robot.dofs_target_pos[joint_indices_np] = jp
        else:
            # 或直接写到 qpos
            if idx.ndim == 0:
                robot.qpos[int(idx)] = float(jp)
            else:
                if isinstance(jp, (int, float)):
                    robot.qpos[idx] = jp
                else:
                    robot.qpos[idx] = jp[: len(idx)]
        return

    raise NotImplementedError("robot 未提供已知的关节控制接口，请实现控制方法")


__all__ = ["apply_motor_commands"]
