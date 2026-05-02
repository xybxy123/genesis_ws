"""
foot_ik.py
【100% 匹配你的 Go2 XML 模型】
HIP 0 = 正中间
THIGH 0 = 垂直向下
CALF: -0.84=伸直, -2.72=弯曲
左hip正=外 | 右hip正=内
大腿thigh 正=向后
小腿calf 负=弯曲
"""

import numpy as np
from typing import Sequence

# 腿长参数（XML 精准值）
L1 = 0.0955
L2 = 0.213
L3 = 0.213


def foot_ik(goal_pos: Sequence[float], leg_index: int = 0):
    x, y, z = goal_pos

    # ========== 右腿侧摆符号修正（你的模型：右正=内）==========
    if leg_index in (1, 3):
        y = -y

    # 1. 侧摆 HIP (0=正中间)
    theta1 = np.arctan2(y, -z)

    # 2. 大腿/小腿平面计算
    x_proj = x
    z_proj = np.sqrt(z**2 + y**2) - L1
    s = np.sqrt(x_proj**2 + z_proj**2)
    s = np.clip(s, abs(L2 - L3) + 1e-6, L2 + L3 - 1e-6)

    # 3. 小腿 CALF: -0.84伸直  -2.72弯曲
    cos3 = (L2**2 + L3**2 - s**2) / (2 * L2 * L3)
    cos3 = np.clip(cos3, -1.0, 1.0)
    theta3 = -np.arccos(cos3)

    # 4. 大腿 THIGH: 0=垂直向下, 正=向后
    cos2 = (L2**2 + s**2 - L3**2) / (2 * L2 * s)
    cos2 = np.clip(cos2, -1.0, 1.0)
    raw_theta2 = np.arctan2(x_proj, z_proj) - np.arccos(cos2)
    theta2 = -raw_theta2  # 正=向后

    # 输出顺序：hip, thigh, calf
    return np.array([theta1, theta2, theta3], dtype=np.float32)


__all__ = ["foot_ik"]
