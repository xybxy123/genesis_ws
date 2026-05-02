"""foot_trajectory_generate.py
功能: 生成单足端在步态周期内的轨迹（摆动/支撑相位）。

说明:
 - 提供简单的轨迹生成函数 `generate_foot_trajectory(phase, params)`，返回足端相对于基座的期望位置（3维数组）。
 - 当前提供一个最小化、可运行的示例：基于参数生成正弦/抬脚高度的轨迹。上层可根据步态相位调用并传入时序参数。

使用示例:
from tra_control.foot_trajectory_generate import generate_foot_trajectory
pos = generate_foot_trajectory(phase=0.25, params={"step_length":0.1, "step_height":0.03})
"""

import numpy as np


def generate_foot_trajectory(phase: float, params: dict = None, **kwargs) -> np.ndarray:
    """生成单足端轨迹示例。

    Args:
        phase: 步态相位，范围 [0,1)
        params: 字典，包含至少 `step_length` 和 `step_height`。
                也支持通过关键字参数直接传入，例如 `step_length=...`, `step_height=...`,
                `side_offset=...`, `turn_offset=...`, `z_default=...`。

    Returns:
        np.ndarray: 长度为3的足端位置向量 [x,y,z]
    """

    # 合并 params 和 kwargs（kwargs 优先）
    merged = {}
    if params:
        merged.update(params)
    merged.update(kwargs)

    step_length = float(merged.get("step_length", 0.1))
    step_height = float(merged.get("step_height", 0.03))
    side_offset = float(merged.get("side_offset", 0.0))
    turn_offset = float(merged.get("turn_offset", 0.0))

    # x: 前后摆动，y: 侧向固定偏移（加上转向偏置），z: 地面高度 + 抬脚
    x = (phase - 0.5) * step_length
    y = side_offset + turn_offset
    # 使用半周期抬脚（phase in [0,1])，峰值在 phase==0.5
    z_base = merged.get("z_base", merged.get("z_default", -0.2))
    z = z_base + step_height * max(0.0, 1.0 - abs((phase - 0.5) * 2))

    return np.array([x, y, z], dtype=float)


__all__ = ["generate_foot_trajectory"]
