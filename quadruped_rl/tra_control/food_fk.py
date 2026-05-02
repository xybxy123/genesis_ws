"""
food_fk.py
功能: 正运动学模块骨架 — 从关节角度计算足端位置（脚端位姿）。

说明:
- 提供 `foot_fk(joint_angles, leg_index, robot)`，优先调用 robot 的 FK 接口（如 `forward_kinematics` 或 `fk`）。
- 目前实现为包装函数，若 robot 未提供 FK 将抛出 NotImplementedError。

注意: 文件名沿用用户创建的 `food_fk.py`（可能为 `foot_fk.py` 的拼写变体）。
"""

import numpy as np
from typing import Sequence


def foot_fk(joint_angles: Sequence[float], leg_index: int, robot):
    """由关节角度计算足端位置，包装上层 robot 的 FK 方法。

    Returns:
            np.ndarray: 足端位置或位姿（取决于 robot 的实现）
    """
    if hasattr(robot, "forward_kinematics"):
        return np.asarray(robot.forward_kinematics(joint_angles, leg_index))
    if hasattr(robot, "fk"):
        return np.asarray(robot.fk(joint_angles, leg_index))
    raise NotImplementedError(
        "robot 未提供正运动学接口，请实现或传入支持 FK 的 robot 实例"
    )


__all__ = ["foot_fk"]
