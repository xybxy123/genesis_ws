import torch
import math
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


# 辅助函数：随机浮点数生成
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


"""
两轮两足机器人仿真环境框架注释
适配硬件特征：共2条腿，每条腿包含2个关节+1个驱动轮子，总计4个关节DOF + 2个轮子DOF
"""


class WLEnv:
    def __init__(
        self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True
    ):
        """
        初始化两轮两足机器人仿真环境

        参数说明：
        - num_envs: 并行仿真环境数量
        - env_cfg: 环境核心配置（关节限位、PD参数、终止条件等）
        - obs_cfg: 观测配置（维度、缩放系数、噪声等）
        - reward_cfg: 奖励配置（各奖励项权重、目标值等）
        - command_cfg: 指令配置（速度指令范围、重采样周期等）
        - show_viewer: 是否启用可视化界面

        初始化流程：
        1. 初始化仿真与训练参数
           - 基础维度配置：环境数量、观测/动作/指令维度
           - 时间参数：仿真步长dt（适配轮子+腿部控制频率）、单轮次最大步数
           - 硬件适配：动作维度设为6（2腿×2关节 + 2轮子），指令维度适配轮足协同控制
           - 设备配置：绑定genesis计算设备，统一数据存储位置
           - 延迟模拟：启用真实机器人的1步动作延迟特性

        2. 创建仿真场景
           - 物理引擎配置：设置步长、子步数，选用Newton约束求解器
           - 可视化配置：设置相机位置/视角/帧率，仅渲染首个环境以节省资源
           - 刚体参数：启用碰撞检测、关节限位，限制最大碰撞对数（适配轮足简化碰撞）
           - 渲染控制：根据show_viewer决定是否启动可视化窗口

        3. 添加地面
           - 加载平面URDF模型，设置为固定刚体（无运动）
           - 适配轮足交互：地面摩擦系数适配轮子滚动特性

        4. 添加机器人
           - 模型加载：加载两轮两足机器人URDF模型
           - 初始位姿：设置机器人基座初始位置和四元数姿态
           - 位姿预处理：计算初始姿态的逆四元数，用于后续坐标变换
           - 碰撞配置：针对轮子-地面、腿部-地面碰撞做特殊配置

        5. 构建场景
           - 批量实例化：根据num_envs创建多并行仿真环境
           - 资源分配：为每个环境分配独立的物理计算资源

        6. 建立名称到索引的映射
           - 关节索引：映射腿部关节（2腿×2关节）、轮子关节名称到DOF索引
           - 分类管理：区分腿部关节（4个）和轮子关节（2个）索引
           - 足端索引：记录左右足端链接的索引，用于接触力检测

        7. 配置PD控制参数
           - 增益配置：为腿部关节和轮子分别设置kp（位置增益）、kv（速度增益）
           - 力限制：设置轮子驱动力和腿部关节力矩上限
           - 阻尼配置：为轮子添加滚动阻尼，适配真实物理特性
           - 惯量补偿：针对轮足协同运动做惯量参数调整

        8. 准备奖励函数系统
           - 奖励注册：遍历奖励配置，绑定对应奖励计算函数
           - 缩放归一：奖励系数乘以dt，消除步长对奖励值的影响
           - 累计缓存：初始化各奖励项的轮次累计值，用于训练监控

        9. 初始化各类缓冲区
           - 状态缓冲区：基座线速度/角速度、四元数姿态、欧拉角、投影重力
           - 关节缓冲区：腿部关节/轮子的位置、速度，记录上一帧值用于动作速率惩罚
           - 指令缓冲区：目标线速度（xy）、角速度（yaw）、腿长/躯干姿态指令
           - 动作缓冲区：当前/上一帧动作，考虑延迟的执行动作
           - 观测缓冲区：拼接所有观测维度，适配观测配置的维度要求
           - 奖励/终止缓冲区：存储每步奖励、终止标志、轮次长度
           - 额外信息缓冲区：记录超时标志、轮次奖励统计等监控信息
        """
        # 1. 初始化仿真与训练参数

        # 并行运行的仿真环境数量
        self.num_envs = num_envs

        # 单个仿真环境的观测向量维度
        self.num_obs = obs_cfg["num_obs"]

        # 特权观测维度(暂时不启用)
        self.num_privileged_obs = None

        # 动作指令维度(控制的关节数量)
        self.num_actions = env_cfg["num_actions"]

        # - 含义：机器人需跟踪的目标指令长度（如基础版3维：x/y线速度 + yaw角速度；扩展版可加腿长/躯干姿态指令）
        # - 作用：commands张量的第二维度，决定_resample_commands()中采样的指令数量
        self.num_commands = command_cfg["num_commands"]

        self.device = gs.device

        # 是否模拟真实机器人1步动作延迟
        self.simulate_action_latency = True

        # 仿真步长(秒)，对应50Hz控制频率（适配轮足协同）
        self.dt = 0.02

        # 单轮仿真最大步数（由总时长/步长向上取整）
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # 环境核心配置字典（关节/PD/终止条件等）
        self.env_cfg = env_cfg

        # 观测配置字典（维度/缩放系数/噪声等）
        self.obs_cfg = obs_cfg

        # 奖励配置字典（各奖励项权重/目标值等）
        self.reward_cfg = reward_cfg

        # 指令配置字典（速度范围/重采样周期等）
        self.command_cfg = command_cfg

        # 观测值缩放系数（统一不同维度量纲）
        self.obs_scales = obs_cfg["obs_scales"]

        # 奖励项缩放系数（平衡各奖励项权重）
        self.reward_scales = reward_cfg["reward_scales"]

        # 2. create scene
        # 创建genesis仿真场景实例
        self.scene = gs.Scene(
            # 仿真核心参数配置
            sim_options=gs.options.SimOptions(
                dt=self.dt,  # 仿真步长（与控制频率绑定，50Hz）
                substeps=2,  # 物理引擎子步数（单次step拆分为2个子步计算，提升轮足物理精度）
            ),
            # 可视化窗口参数配置
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(
                    0.5 / self.dt
                ),  # 最大渲染帧率（0.5/dt避免渲染过快，适配仿真步长）
                camera_pos=(2.0, 0.0, 2.5),  # 可视化相机位置（x,y,z），俯视机器人
                camera_lookat=(0.0, 0.0, 0.5),  # 相机朝向（聚焦机器人基座位置）
                camera_fov=40,  # 相机视场角（度数），控制视野范围
            ),
            # 渲染优化配置
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(
                    range(1)
                )  # 仅渲染第1个仿真环境（减少并行渲染算力消耗）
            ),
            # 刚体物理参数配置（适配轮足协同特性）
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,  # 刚体仿真步长（与全局仿真步长保持一致）
                constraint_solver=gs.constraint_solver.Newton,  # 约束求解器（Newton法适配轮足碰撞求解）
                enable_collision=True,  # 启用碰撞检测（轮子/腿部与地面碰撞）
                enable_joint_limit=True,  # 启用关节限位（防止腿部关节超程）
                max_collision_pairs=30,  # 最大碰撞对数量（轮足碰撞对少，优化内存占用）
            ),
            show_viewer=show_viewer,  # 是否显示可视化窗口（训练时关闭，调试时开启）
        )

        # 3. add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # 4. add robot
        # 机器人基座初始位置张量
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=gs.device
        )
        # 机器人基座初始姿态四元数
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=gs.device
        )
        # 计算初始姿态四元数的逆（用于后续机体坐标系与世界坐标系的变换）
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # 加载两轮两足机器人URDF模型并添加到仿真场景
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="../assets/urdf/wheel_leg.urdf",  # 机器人URDF文件相对路径
                pos=self.base_init_pos.cpu().numpy(),  # 初始位置（转numpy适配genesis接口）
                quat=self.base_init_quat.cpu().numpy(),  # 初始姿态（转numpy适配genesis接口）
            ),
        )

        # 5. build
        self.scene.build(n_envs=num_envs)

        # 6. names to indices
        # 从环境配置读取腿部关节名称列表（左右腿各2个关节）
        self.leg_joint_names = env_cfg["leg_joint_names"]
        # 从环境配置读取轮子关节名称列表（左右轮子各1个关节）
        self.wheel_joint_names = env_cfg["wheel_joint_names"]
        # 合并腿部和轮子关节名称，形成所有受控关节的名称列表
        self.all_joint_names = self.leg_joint_names + self.wheel_joint_names
        # 遍历所有关节名称，获取每个关节在仿真中的DOF驱动索引（用于PD控制）
        self.motors_dof_idx = [
            self.robot.get_joint(name).dof_start for name in self.all_joint_names
        ]
        # 提取前4个索引作为腿部关节的DOF索引（2腿×2关节）
        self.leg_dof_idx = self.motors_dof_idx[:4]
        # 提取后2个索引作为轮子关节的DOF索引（左右轮各1个）
        self.wheel_dof_idx = self.motors_dof_idx[4:]
        # 从环境配置读取足端链接名称，获取其在仿真中的链接索引（用于检测足端接触力）
        # ========== 足端 Link 定义 ==========
        # print("--- DEBUG: WLEnv 接收到的配置字典键值 ---")
        # print(self.env_cfg.keys())
        # 参考：足端位置
        self.left_foot = self.robot.get_link(self.env_cfg["foot_link_names"][0])
        self.right_foot = self.robot.get_link(self.env_cfg["foot_link_names"][1])

        # 初始化用于存储足端位置的张量
        self.left_foot_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.right_foot_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        # 初始化足端在基座坐标系下的位置张量
        self.left_foot_base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.right_foot_base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        # 首次获取足端位置并填充到张量中
        self.left_foot_pos[:] = self.left_foot.get_pos()
        self.right_foot_pos[:] = self.right_foot.get_pos()

        # 记录足端Link的索引（用于后续更简洁的访问，如果需要）
        # link_names = [l.name for l in self.robot.get_link_list_info()]

        # 备选方案：如果 get_link_list_info() 仍报错，请尝试：
        # rc_wl_env.py (修正后的代码)
        # 尝试使用 links 属性获取 Link 列表
        link_names = [l.name for l in self.robot.links]
        left_foot_name = self.env_cfg["foot_link_names"][0]
        right_foot_name = self.env_cfg["foot_link_names"][1]

        self.foot_link_idx = torch.tensor(
            [link_names.index(left_foot_name), link_names.index(right_foot_name)],
            device=gs.device,
            dtype=torch.long,
        )

        # 7. PD control parameters
        # 腿部关节和轮子分别设置PD位置增益（kp）
        # 配置逻辑：4个腿部关节用leg_kp，2个轮子关节用wheel_kp，按关节索引顺序赋值
        self.robot.set_dofs_kp(
            [self.env_cfg["leg_kp"]] * 4
            + [self.env_cfg["wheel_kp"]] * 2,  # PD控制器位置增益列表
            self.motors_dof_idx,  # 对应关节的DOF索引（腿部4个+轮子2个）
        )
        # 腿部关节和轮子分别设置PD速度增益（kv）
        # 配置逻辑：4个腿部关节用leg_kd，2个轮子关节用wheel_kd，按关节索引顺序赋值
        self.robot.set_dofs_kv(
            [self.env_cfg["leg_kd"]] * 4
            + [self.env_cfg["wheel_kd"]] * 2,  # PD控制器速度增益列表
            self.motors_dof_idx,  # 对应关节的DOF索引（腿部4个+轮子2个）
        )
        # 设置关节最大输出力/力矩（保护硬件，避免过载）
        # 配置逻辑：4个腿部关节限制最大力矩（leg_max_torque），2个轮子关节限制最大驱动力（wheel_max_force）
        # self.robot.set_dofs_max_force(
        #     [self.env_cfg["leg_max_torque"]] * 4
        #     + [self.env_cfg["wheel_max_force"]] * 2,  # 力/力矩限制列表
        #     self.motors_dof_idx,  # 对应关节的DOF索引（腿部4个+轮子2个）
        # )    没看到这个api

        # 8. prepare reward functions and multiply reward scales by dt
        # 初始化奖励函数字典（存储各奖励项的计算函数）和单轮奖励累加字典（存储各奖励项的累计值）
        self.reward_functions, self.episode_sums = dict(), dict()
        # 遍历所有奖励项名称（从奖励缩放系数字典的键获取，如速度跟踪、姿态稳定等）
        for name in self.reward_scales.keys():
            # 奖励缩放系数按仿真步长归一化（避免不同dt导致奖励值量级偏差，适配50Hz控制频率）
            self.reward_scales[name] *= self.dt
            # 动态绑定奖励项对应的计算函数（如name为vel_track则绑定self._reward_vel_track函数）
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            # 初始化每个奖励项在各并行环境中的单轮累加值（num_envs个环境，初始值为0，绑定仿真计算设备）
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        # 9. initialize buffers（初始化各类状态缓冲区，存储各并行环境的实时状态）
        # 基座状态（位置/速度/姿态/重力投影等）
        # 基座线速度缓冲区（num_envs个环境，每个环境3维[x,y,z]，初始0，绑定仿真计算设备）
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        # 基座角速度缓冲区（num_envs个环境，每个环境3维[roll/pitch/yaw角速度]，初始0）
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        # 机体坐标系下的重力投影缓冲区（num_envs个环境，每个环境3维，初始0，用于判断姿态倾斜）
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        # 世界坐标系下的重力向量（[0,0,-1]），扩展为num_envs个环境的维度（每个环境重力一致）
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        # 动作与指令（观测/动作/目标指令等）
        # 观测数据缓冲区（num_envs个环境，每个环境num_obs维，初始0，存储实时观测值）
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float
        )
        # 奖励与终止（单步奖励/重置标记/单轮时长等）
        # 单步奖励缓冲区（num_envs个环境，每个环境1维，初始0，存储当前步的总奖励）
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        # 环境重置标记缓冲区（num_envs个环境，初始1表示需要重置，1=重置/0=正常运行）
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        # 单轮仿真时长缓冲区（num_envs个环境，初始0，记录当前轮已运行的步数）
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        # 目标指令缓冲区（num_envs个环境，每个环境num_commands维，初始0，存储速度/姿态等目标指令）
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float
        )
        # 指令缩放系数张量（线速度x/y + 角速度z，按观测缩放系数配置，统一指令量纲）
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        # 动作指令缓冲区（num_envs个环境，每个环境num_actions维，初始0，存储待下发的动作）
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        # 上一步动作指令缓冲区（记录上一帧动作，用于动作平滑/惩罚动作突变）
        self.last_actions = torch.zeros_like(self.actions)
        # 关节/轮子实时位置缓冲区（num_envs个环境，每个环境num_actions维，初始0）
        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        # 关节/轮子实时速度缓冲区（维度与dof_pos一致，初始0）
        self.dof_vel = torch.zeros_like(self.dof_pos)
        # 关节/轮子上一步速度缓冲区（记录上一帧速度，用于速度平滑/差分计算）
        self.last_dof_vel = torch.zeros_like(self.dof_pos)
        # 基座位置缓冲区（num_envs个环境，每个环境3维[x,y,z]，初始0，记录基座在世界坐标系的位置）
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        # 基座姿态四元数缓冲区（num_envs个环境，每个环境4维[x,y,z,w]，初始0，存储实时姿态）
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )

        # 额外添加
        # 关节/轮子状态（默认位姿/实时位姿/速度等）
        # 腿部关节默认角度张量（按leg_joint_names顺序读取配置，绑定仿真设备）
        self.default_leg_pos = torch.tensor(
            [self.env_cfg["default_leg_angles"][name] for name in self.leg_joint_names],
            device=gs.device,
            dtype=gs.tc_float,
        )

        # 额外添加
        # 轮子关节默认角度张量（按wheel_joint_names顺序读取配置，绑定仿真设备）
        self.default_wheel_pos = torch.tensor(
            [
                self.env_cfg["default_wheel_angles"][name]
                for name in self.wheel_joint_names
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        # 合并腿部+轮子的默认关节位姿（用于动作指令的基准位姿）
        self.default_dof_pos = torch.cat(
            [self.default_leg_pos, self.default_wheel_pos]
        )  # 合并默认位姿
        # 额外信息（足端接触/自定义观测等）
        self.extras = dict()  # 存储训练/调试用的额外信息
        self.extras["observations"] = dict()  # 存储自定义观测数据的子字典

        # 额外添加
        # 足端接触状态缓冲区（num_envs个环境，每个环境2维[左足/右足]，0=未接触/1=接触）
        self.extras["foot_contact"] = torch.zeros(
            (self.num_envs, 2), device=gs.device, dtype=gs.tc_float
        )  # 足端接触状态
        # 基座欧拉角缓冲区（num_envs个环境，每个环境3维[roll/pitch/yaw]，初始0，用于倾覆判断）
        self.base_euler = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )  # 基座欧拉角

        self.train_stage = 1  # 1: 平衡阶段，2: 速度跟踪阶段
        self.stage_1_steps = self.env_cfg["train_stage_1_steps"]
        self.stage_2_speed = self.env_cfg["train_stage_2_speed"]
        self.total_training_steps = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
    
        self.lin_vel_x_range = command_cfg["lin_vel_x_range"]
        self.lin_vel_y_range = command_cfg["lin_vel_y_range"]
        self.ang_vel_range = command_cfg["ang_vel_range"]

    def _resample_commands(self, envs_idx):
        """
        重采样控制指令（针对指定环境索引）
        适配两轮两足机器人的轮足协同控制需求

        参数：
        - envs_idx: 需要重采样指令的环境索引列表

        采样逻辑：
        1. 线速度采样
           - 在预设x/y轴线速度范围内随机生成目标值
           - 考虑轮子最大转速限制，x轴速度不超过轮子额定速度
           - y轴速度适配腿部侧向移动能力，限制采样范围

        2. 角速度采样
           - 基于当前线速度动态调整角速度上限（防止高速转向倾覆）
           - 仅采样yaw轴角速度（两轮机器人主要绕z轴转向）
           - 低速时允许更大转向角速度，高速时限制转向幅度

        3. 腿长命令采样（可选）
           - 左右腿高度目标值，在预设范围内随机采样
           - 可配置左右腿高度的相关性（同步/异步调整）

        4. 躯干姿态命令（可选）
           - 躯干俯仰/滚转角目标，适配不平地面行走需求
           - 限制姿态范围，防止机器人失稳
        """

        """
        重采样控制指令（针对指定环境索引）
        """
        if len(envs_idx) == 0:
            return

        # 1. 线速度采样
        # 使用当前阶段设置的线速度范围
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.lin_vel_x_range, (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.lin_vel_y_range, (len(envs_idx),), gs.device
        )

        # 2. 角速度采样（动态上限）
        current_lin_vel_x = self.commands[envs_idx, 0].abs()
        
        # 避免除零错误
        if self.lin_vel_x_range[1] > 0:
            ang_vel_max = self.command_cfg["ang_vel_max"] * (
                1 - current_lin_vel_x / self.lin_vel_x_range[1]
            )
        else:
            ang_vel_max = self.command_cfg["ang_vel_max"]
        
        # 使用当前阶段设置的角速度范围
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.ang_vel_range, (len(envs_idx),), gs.device
        )

        # 3. 腿长命令采样（可选）
        # if self.num_commands > 3:
        #     self.commands[envs_idx, 3] = gs_rand_float(
        #         *self.command_cfg["leg_length_range"], (len(envs_idx),), gs.device
        #     )
        #     # 左右腿高度相关性
        #     if self.command_cfg["leg_correlation"] > 0:
        #         self.commands[envs_idx, 4] = self.commands[envs_idx, 3] + gs_rand_float(
        #             -self.command_cfg["leg_variation"],
        #             self.command_cfg["leg_variation"],
        #             (len(envs_idx),),
        #             gs.device,
        #         )
        #     else:
        #         self.commands[envs_idx, 4] = gs_rand_float(
        #             *self.command_cfg["leg_length_range"], (len(envs_idx),), gs.device
        #         )

        # 4. 躯干姿态命令（可选）
        # if self.num_commands > 5:
        #     self.commands[envs_idx, 5] = gs_rand_float(
        #         *self.command_cfg["pitch_range"], (len(envs_idx),), gs.device
        #     )
        #     self.commands[envs_idx, 6] = gs_rand_float(
        #         *self.command_cfg["roll_range"], (len(envs_idx),), gs.device
        #     )

    def step(self, actions):
        """
        仿真环境单步执行（核心逻辑）
        适配两轮两足机器人的轮足协同控制流程

        参数：
        - actions: 模型输出的动作张量，形状[num_envs, num_actions]

        执行流程：
        1. 动作处理
           - 动作限幅：限制腿部关节和轮子动作在安全范围内
           - 延迟模拟：使用上一帧动作作为执行动作（模拟真实机器人延迟）
           - 动作缩放：将归一化动作映射到关节/轮子的实际控制范围
           - PD指令：为腿部关节和轮子分别发送位置控制指令

        2. 仿真步进
           - 调用物理引擎执行一步仿真
           - 同步所有并行环境的物理状态
           - 检测轮子-地面、足端-地面接触状态

        3. 状态更新
           - 基座状态：更新位置、姿态、线速度/角速度（转换到机体坐标系）
           - 关节状态：更新腿部关节/轮子的位置和速度
           - 接触状态：检测左右足端的地面接触力
           - 坐标变换：计算投影重力、机体坐标系下的速度等

        4. 观测构建
           - 基础状态：基座角速度、投影重力、目标指令（缩放后）
           - 关节状态：腿部关节/轮子的位置偏差、速度（缩放后）
           - 动作信息：当前执行的动作值
           - 接触信息：足端接触状态（可选）
           - 历史帧：拼接历史观测（可选，增强时序信息）
           - 噪声添加：为观测添加高斯噪声（可选，提高鲁棒性）

        5. 奖励计算
           - 遍历所有奖励项，调用对应计算函数
           - 按权重加权求和得到总奖励
           - 累计各奖励项的轮次总和，用于训练监控

        6. 终止条件检查
           - 超时终止：轮次步数超过最大长度
           - 倾覆终止：基座俯仰/滚转角超过安全阈值
           - 越界终止：机器人位置超出预设范围（可选）
           - 异常终止：轮子/关节速度超过安全阈值（可选）

        7. 环境重置
           - 筛选需要重置的环境索引
           - 调用reset_idx重置对应环境的状态
           - 重采样新的控制指令

        8. 信息收集
           - 记录超时标志、接触状态、奖励分解等信息
           - 用于训练过程中的监控和分析

        9. 返回标准四元组
           - 观测：当前环境观测张量
           - 奖励：单步奖励张量
           - 终止：环境终止标志张量
           - 额外信息：包含监控信息的字典
        """

        # 更新总训练步数
        self.total_training_steps += 1
        # print(f"训练step:{self.total_training_steps}")
        
        # 检查是否需要切换训练阶段
        self._check_train_stage()

        # 1. 动作处理：分「大腿/小腿/轮子」精细化裁剪（替代原统一裁剪逻辑）
        # 拆分动作张量：按关节类型拆分（需匹配关节顺序：左大腿、左小腿、右大腿、右小腿、左轮、右轮）
        thigh_actions = actions[:, [0, 2]]  # 左右大腿动作：(num_envs, 2)
        calf_actions = actions[:, [1, 3]]  # 左右小腿动作：(num_envs, 2)
        wheel_actions = actions[:, 4:]  # 左右轮子动作：(num_envs, 2)

        # # 2. 分别裁剪：按各自约束推导归一化裁剪阈值（核心：把实际角度范围映射到-1~1的归一化动作）
        # ## 2.1 大腿动作裁剪（映射实际角度范围[-0.8,1.2]到归一化动作[-1,1]）
        # # 推导：归一化动作的裁剪阈值=1.0（固定，因为策略输出是-1~1，裁剪仅兜底）
        # thigh_actions_clipped = torch.clip(
        #     thigh_actions,
        #     -1.0,  # 归一化动作最小值（兜底，防止策略输出超范围）
        #     1.0,  # 归一化动作最大值
        # )
        # ## 2.2 小腿动作裁剪（同理，映射[-2.0,0.65]到-1~1）
        # calf_actions_clipped = torch.clip(calf_actions, -1.0, 1.0)
        # ## 2.3 轮子动作裁剪（映射最大速度[-5.0,5.0]到-1~1）
        # wheel_actions_clipped = torch.clip(wheel_actions, -1.0, 1.0)

        # # 3. 拼接裁剪后的动作：恢复(num_envs, num_actions)维度（按原顺序拼接）
        # clipped_actions = torch.cat(
        #     [
        #         thigh_actions_clipped[:, [0]],  # 左大腿
        #         calf_actions_clipped[:, [0]],  # 左小腿
        #         thigh_actions_clipped[:, [1]],  # 右大腿
        #         calf_actions_clipped[:, [1]],  # 右小腿
        #         wheel_actions_clipped,  # 左右轮
        #     ],
        #     dim=1,
        # )
        # self.actions = clipped_actions

        # 直接对整个actions进行裁剪
        clipped_actions = torch.clip(actions, -1.0, 1.0)
        self.actions = clipped_actions

        # 4. 模拟真实机器人的1步动作延迟
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )

        # 5. 动作缩放：分「大腿/小腿（角度）」「轮子（速度）」转换为实际控制量
        ## 5.1 大腿关节：归一化动作→实际角度（映射[-1,1]到[-0.8,1.2]）
        # 推导公式：实际角度 = 归一化动作 × 缩放系数 + 基准角度
        # 缩放系数 = (最大角度 - 最小角度) / 2 = (1.2 - (-0.8))/2 = 1.0
        # 基准角度 = (最大角度 + 最小角度) / 2 = (1.2 + (-0.8))/2 = 0.2
        thigh_scale = (
            self.env_cfg["thigh_angle_max"] - self.env_cfg["thigh_angle_min"]
        ) / 2
        thigh_base = (
            self.env_cfg["thigh_angle_max"] + self.env_cfg["thigh_angle_min"]
        ) / 2
        thigh_target_pos = (
            exec_actions[:, [0, 2]] * thigh_scale + thigh_base
        )  # (num_envs, 2)

        ## 5.2 小腿关节：归一化动作→实际角度（映射[-1,1]到[-2.0,0.65]）
        calf_scale = (
            self.env_cfg["calf_angle_max"] - self.env_cfg["calf_angle_min"]
        ) / 2
        calf_base = (
            self.env_cfg["calf_angle_max"] + self.env_cfg["calf_angle_min"]
        ) / 2
        calf_target_pos = (
            exec_actions[:, [1, 3]] * calf_scale + calf_base
        )  # (num_envs, 2)

        ## 5.3 轮子：归一化动作→实际速度（限制最大速度5.0rad/s）
        wheel_target_vel = (
            exec_actions[:, 4:] * self.env_cfg["wheel_max_vel"]
        )  # (num_envs, 2)

        # 6. 合并腿部目标角度（位置控制），下发执行
        ## 6.1 拼接大腿+小腿目标角度（按关节顺序：左大腿、左小腿、右大腿、右小腿）
        leg_target_pos = torch.cat(
            [
                thigh_target_pos[:, [0]],  # 左大腿
                calf_target_pos[:, [0]],  # 左小腿
                thigh_target_pos[:, [1]],  # 右大腿
                calf_target_pos[:, [1]],  # 右小腿
            ],
            dim=1,
        )  # (num_envs, 4)

        ## 6.2 下发腿部关节位置指令（PD位置控制）
        self.robot.control_dofs_position(
            leg_target_pos, self.leg_dof_idx  # 腿部4个关节的DOF索引：[0,1,2,3]
        )

        ## 6.3 下发轮子速度指令（替换原位置控制，改为速度控制）
        self.robot.control_dofs_velocity(  # 速度控制API（替代control_dofs_position）
            wheel_target_vel, self.wheel_dof_idx  # 轮子2个关节的DOF索引：[4,5]
        )

        # 2. 仿真步进
        self.scene.step()

        # 3. 状态更新
        # 累加单轮仿真时长：每个并行环境的已运行步数+1（记录当前轮次的持续步数）
        self.episode_length_buf += 1

        # ========== 基座状态更新（世界坐标系→机体坐标系，核心用于观测/奖励计算） ==========
        # 更新基座在世界坐标系下的位置（x/y/z），覆盖原有缓冲区数据
        self.base_pos[:] = self.robot.get_pos()
        # 更新基座在世界坐标系下的姿态四元数（x/y/z/w），覆盖原有缓冲区数据
        self.base_quat[:] = self.robot.get_quat()
        # 计算基座相对于初始姿态的欧拉角（roll/pitch/yaw，单位：度）
        # 核心逻辑：用初始姿态的逆四元数抵消初始偏差，仅计算相对初始姿态的旋转角度（避免初始摆放偏移干扰）
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat)
                * self.inv_base_init_quat,  # 扩展初始逆四元数到环境维度
                self.base_quat,  # 当前基座姿态四元数（世界坐标系）
            ),
            rpy=True,  # 输出顺序为roll(x)/pitch(y)/yaw(z)
            degrees=True,  # 角度单位为度（而非弧度）
        )
        # 计算基座当前姿态的逆四元数（用于后续将世界坐标系物理量转换为机体坐标系）
        inv_base_quat = inv_quat(self.base_quat)
        # 更新基座线速度（转换为机体坐标系）：从仿真引擎读取世界坐标系线速度，通过逆四元数转换为机器人自身视角的速度
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        # 更新基座角速度（转换为机体坐标系）：从仿真引擎读取世界坐标系角速度，转换为机器人自身视角的角速度
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        # 更新机体坐标系下的重力投影：将世界坐标系重力[0,0,-1]转换为机器人自身视角，用于判断姿态倾斜（如pitch角过大则重力投影异常）
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        # ========== 关节/轮子状态更新（读取实时位姿和速度） ==========
        # 更新关节/轮子的实时位置：按motors_dof_idx索引从仿真引擎读取腿部+轮子的实际角度（弧度）
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        # 更新关节/轮子的实时速度：按motors_dof_idx索引读取腿部+轮子的实际角速度（弧度/秒）
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # ========== 足端接触状态更新（通过足端位置z坐标判断是否触地，替代接触力API） ==========
        # 先定义地面基准高度（假设地面z=0，可根据机器人初始姿态调整）
        ground_z = 0.0
        # 遍历左右足端的链接索引（i=0左足，i=1右足）
        for i, link_idx in enumerate(self.foot_link_idx):
            # 方式1：若已提前定义left_foot/right_foot（参考用户提供的代码）
            # 适配：根据索引区分左右足，获取足端位置
            if i == 0:  # 左足
                foot_pos = self.left_foot.get_pos()  # 获取左足端世界坐标系位置（x/y/z）
            else:  # 右足
                foot_pos = (
                    self.right_foot.get_pos()
                )  # 获取右足端世界坐标系位置（x/y/z）

            # 方式2：若未定义left_foot/right_foot，直接通过link_idx获取足端位置（通用方式）
            # foot_pos = self.robot.get_link_pos(link_idx)  # 需确认引擎是否有该API，优先方式1

            # 提取足端z轴坐标（垂直地面方向）
            foot_z = foot_pos[:, 2]
            # 判断足端是否接触地面：z坐标≤地面基准+阈值则为触地（1.0），否则未接触（0.0）
            # 逻辑：足端z越低越接近地面，≤阈值说明踩在地面上
            # foot_ground_z_threshold为0.05(车轮半径)+0.01(容忍)
            self.extras["foot_contact"][:, i] = (
                foot_z <= (ground_z + self.env_cfg["foot_ground_z_threshold"])
            ).float()

        # 4. 重采样指令
        # 1. 计算重采样的周期步数
        # self.env_cfg["resampling_time_s"]：配置中设置的重采样时间间隔（秒）。
        # self.dt：仿真环境的时间步长（秒）。
        # (resampling_time_s / self.dt) = 多少个仿真步长 (steps) 构成一个重采样周期。
        resampling_period = int(self.env_cfg["resampling_time_s"] / self.dt)
        # 2. 识别需要重采样的环境索引
        envs_idx = (
            (
                # self.episode_length_buf: 形状为 (num_envs,) 的 Tensor，记录每个环境当前回合已经运行的步数。
                self.episode_length_buf
                # 使用取模运算 (%) 检查当前步数是否为重采样周期的整数倍。
                # 结果是一个形状为 (num_envs,) 的布尔 Tensor (True/False)。
                % resampling_period
                == 0
            )
            # .nonzero(): 返回所有 True 元素的索引（即满足重采样条件的环境索引）。
            # as_tuple=False: 返回一个形状为 (N, 1) 的 Tensor，其中 N 是需要重采样的环境数量。
            .nonzero(as_tuple=False)
            # .reshape((-1,)): 将索引 Tensor 展平为形状 (N,) 的一维 Tensor。
            .reshape((-1,))
        )
        # 3. 对识别出的环境执行命令重采样
        # _resample_commands() 是一个内部方法，负责为 envs_idx 中的环境生成新的运动命令（如速度目标）。
        self._resample_commands(envs_idx)

        # 5. 终止条件检查：判断是否触发环境重置（多条件“或”逻辑，任一条件满足则重置）
        # 5.1 基础终止条件：单轮仿真步数超过最大允许步数（超时终止）
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # reset_buf为布尔张量，shape=(num_envs,)，True表示需重置
        # 5.2 倾覆检测：机器人基座俯仰角/滚转角超限（判定为倾覆，强制重置）
        # 基座欧拉角base_euler[:,0]=roll（滚转，绕x轴）、[:,1]=pitch（俯仰，绕y轴）、[:,2]=yaw（偏航，绕z轴）
        # 俯仰角（pitch）超限检测：绝对值超过配置阈值则触发重置
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])  # 取所有环境基座俯仰角的绝对值
            > self.env_cfg[
                "termination_if_pitch_greater_than"
            ]  # 俯仰角最大容忍值（单位：度/弧度，需与base_euler一致）
        )
        # 滚转角（roll）超限检测：绝对值超过配置阈值则触发重置
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])  # 取所有环境基座滚转角的绝对值
            > self.env_cfg["termination_if_roll_greater_than"]  # 滚转角最大容忍值
        )
        # 在 step 方法中的“终止条件检查”部分（第5步）添加以下代码：
        # 5.3 基座高度过低检测：防止机器人趴在地上
        self.reset_buf |= (
            self.base_pos[:, 2]  # 获取所有环境的基座Z坐标
            < self.env_cfg["termination_if_base_z_less_than"]  # 配置阈值，例如0.15米
        )
        # 5.3 可选检测：轮子/关节速度超限（防止关节转速过高导致仿真异常）
        # if self.env_cfg.get(
        #     "check_dof_vel", False
        # ):  # 从配置读取是否开启该检测，默认关闭
        #     # 计算每个环境下所有关节速度的最大绝对值（dim=1按环境维度取最大值，shape=(num_envs,)）
        #     dof_vel_max = torch.max(torch.abs(self.dof_vel), dim=1)[0]
        #     # 关节最大速度超过配置阈值则触发重置
        #     self.reset_buf |= (
        #         dof_vel_max > self.env_cfg["max_dof_vel"]
        #     )  # max_dof_vel为关节最大允许速度（弧度/秒）
        # 5.4 可选检测：基座位置越界（防止机器人跑出仿真区域）
        # if self.env_cfg.get(
        #     "check_base_pos", False
        # ):  # 从配置读取是否开启该检测，默认关闭
        #     # x轴越界检测：基座x坐标绝对值超过x轴最大范围
        #     self.reset_buf |= (
        #         torch.abs(self.base_pos[:, 0])
        #         > self.env_cfg["max_x_range"]  # max_x_range为x轴允许最大范围（米）
        #     )
        #     # y轴越界检测：基座y坐标绝对值超过y轴最大范围
        #     self.reset_buf |= (
        #         torch.abs(self.base_pos[:, 1])
        #         > self.env_cfg["max_y_range"]  # max_y_range为y轴允许最大范围（米）
        #     )
        # 5.5 超时标志记录：区分“超时终止”和“异常终止（倾覆/越界等）”
        # 第一步：找到所有因超时触发重置的环境索引（nonzero返回非零元素索引，reshape展平为一维）
        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)  # 仅判断超时条件
            .nonzero(as_tuple=False)  # 返回shape=(n,1)的索引张量（n为超时环境数）
            .reshape((-1,))  # 展平为shape=(n,)的一维张量，便于后续赋值
        )
        # 第二步：初始化超时标志张量（与reset_buf同维度，默认全0，0=非超时终止，1=超时终止）
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf,  # 匹配reset_buf的shape和device
            device=gs.device,  # 绑定Genesis框架的计算设备（CPU/GPU）
            dtype=gs.tc_float,  # 采用Genesis定义的浮点类型（兼容框架张量类型）
        )
        # 第三步：给超时环境的超时标志赋值为1.0
        self.extras["time_outs"][time_out_idx] = 1.0

        # 6. 执行环境重置：对所有触发重置的环境执行重置逻辑
        # reset_buf.nonzero(...)获取所有需重置的环境索引，传入reset_idx函数完成重置（包括位姿、速度、步数清零等）
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # 7. 奖励计算：按多维度奖励项加权累加，最终输出每个环境的单步奖励
        # 7.1 初始化奖励缓冲区：将所有环境的单步奖励清零（避免上一步奖励残留）
        self.rew_buf[:] = 0.0  # rew_buf shape=(num_envs,)，存储每个环境的单步总奖励

        # 7.2 遍历所有奖励函数，计算并累加各维度奖励
        # self.reward_functions：字典，key=奖励项名称（如"velocity_tracking"），value=该奖励的计算函数
        # self.reward_scales：字典，对应每个奖励项的权重（控制不同奖励的重要性）
        for name, reward_func in self.reward_functions.items():
            # 计算当前奖励项的原始值，并乘以对应权重（缩放奖励幅度）
            rew = reward_func() * self.reward_scales[name]  # rew shape=(num_envs,)

            # 根据训练阶段调整某些奖励的权重
            if self.train_stage == 1 and name == "tracking_lin_x_vel":
                # 第一阶段降低速度跟踪权重，专注于平衡
                rew *= 0.1
            elif self.train_stage == 2 and name == "tracking_lin_x_vel":
                # 第二阶段提高速度跟踪权重
                rew *= 2.0

            # 累加至总奖励缓冲区（多奖励项加权求和）
            self.rew_buf += rew
            # 累计该奖励项在当前轮次的总收益（用于日志分析/奖励项效果评估）
            self.episode_sums[name] += rew

        # 8. 观测构建：拼接多维度观测特征，形成强化学习模型的输入（状态表示）
        # 8.1 定义观测组件列表（每个组件为张量，需保证维度匹配，最后一维拼接）
        obs_components = [
            self.base_ang_vel
            * self.obs_scales[
                "ang_vel"
            ],  # 基座角速度（3维：roll/pitch/yaw轴），按配置缩放（统一量纲）
            self.projected_gravity,  # 机体坐标系下的重力投影（3维），反映基座倾斜姿态
            self.commands
            * self.commands_scale,  # 控制指令（如速度指令，num_commands维，如2维：x/y速度），缩放后匹配观测范围
            (self.dof_pos - self.default_dof_pos)
            * self.obs_scales[
                "dof_pos"
            ],  # 关节/轮子位置偏差（6维：左右大腿/小腿/轮子），相对默认位姿的偏移，缩放统一量纲
            self.dof_vel
            * self.obs_scales[
                "dof_vel"
            ],  # 关节/轮子角速度（6维），缩放后避免数值过大/过小
            self.actions,  # 当前执行的动作（6维：对应6个关节的控制量），范围通常[-1,1]
            self.extras[
                "foot_contact"
            ],  # 足端接触状态（2维：左/右轮子触地标记，0/1），反映机器人与地面的交互状态
        ]
        # 8.2 拼接所有观测组件：按最后一维（-1）拼接，最终obs_buf shape=(num_envs, obs_dim)
        # obs_dim = 3(角速度)+3(重力)+num_commands(指令)+6(位置偏差)+6(速度)+6(动作)+2(触地)
        self.obs_buf = torch.cat(obs_components, axis=-1)

        # 9. 缓存更新：保存当前步的关键状态，供下一时间步计算使用
        # 9.1 保存当前动作到上一步动作缓存（用于后续动作平滑/差分计算）
        self.last_actions[:] = self.actions[:]  # last_actions shape=(num_envs, 6)
        # 9.2 保存当前关节速度到上一步速度缓存（用于速度差分/奖励计算）
        self.last_dof_vel[:] = self.dof_vel[:]  # last_dof_vel shape=(num_envs, 6)
        # 9.3 更新Critic网络的观测缓存（Actor-Critic架构中，Critic使用相同观测评估状态价值）
        self.extras["observations"]["critic"] = self.obs_buf

        # 10. 返回标准四元组：符合强化学习环境step()函数的通用输出格式
        # 返回值说明：
        # - self.obs_buf：当前观测（num_envs, obs_dim），供Actor网络决策下一步动作
        # - self.rew_buf：单步奖励（num_envs,），用于更新策略网络
        # - self.reset_buf：终止标志（num_envs,），True表示环境需重置，False继续
        # - self.extras：额外信息字典（含超时标志、critic观测等），用于日志/调试/评估
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _check_train_stage(self):
        """检查并切换训练阶段"""
        if self.train_stage == 1 and self.total_training_steps[0] >= self.stage_1_steps:
            print(f"\033[1;32m 切换到第二阶段训练: 线速度范围 ±{self.stage_2_speed} m/s \033[0m")
            self.train_stage = 2
            # 重置命令范围
            self._setup_stage_2_commands()
            
    def _setup_stage_2_commands(self):
        """设置第二阶段的速度命令范围"""
        # 使用配置中的第二阶段命令范围
        self.lin_vel_x_range = self.command_cfg.get("stage_2_lin_vel_x_range", 
                                                [-self.stage_2_speed, self.stage_2_speed])
        self.lin_vel_y_range = self.command_cfg.get("stage_2_lin_vel_y_range", [0.0, 0.0])
        self.ang_vel_range = self.command_cfg.get("stage_2_ang_vel_range", 
                                                [-self.stage_2_speed * 3, self.stage_2_speed * 3])
        
        # 重置所有环境的命令
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        
        print(f"第二阶段命令范围设置完成: lin_vel_x_range={self.lin_vel_x_range}, "
            f"ang_vel_range={self.ang_vel_range}")

    def reset_idx(self, envs_idx):
        """
        重置指定索引的环境
        """
        if len(envs_idx) == 0:
            return

        # 重置关节/轮子状态
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # 重置基座状态
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # 重置缓冲区
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # 奖励统计
        # 初始化episode级别的统计信息字典：用于存储本轮结束的episode中各奖励项的标准化收益（平均每秒奖励）
        self.extras["episode"] = {}

        # 遍历所有奖励项的累计键（如"velocity_tracking"、"pose_stability"、"energy_penalty"等）
        for key in self.episode_sums.keys():
            # 计算该奖励项的「平均每秒奖励」并存入extras，用于日志/可视化分析
            # 核心逻辑：累计奖励和 → 求均值 → 转Python数值 → 除以episode总时长（标准化）
            self.extras["episode"]["rew_" + key] = (
                # 1. self.episode_sums[key][envs_idx]：取本次重置的环境（envs_idx）在本轮episode中该奖励项的累计值
                # 2. torch.mean(...)：计算这些环境的累计奖励均值（消除多环境差异，反映整体表现）
                # 3. .item()：将张量转为Python标量（便于后续打印/保存日志，避免张量类型）
                torch.mean(self.episode_sums[key][envs_idx]).item()
                # 4. 除以episode总时长（秒）：标准化为「平均每秒奖励」，统一不同episode时长的对比基准
                / self.env_cfg["episode_length_s"]
            )
            # 重置该奖励项的累计值：将本次重置的环境的累计奖励清零，为下一轮episode重新累计
            self.episode_sums[key][envs_idx] = 0.0

        # 重采样指令
        self._resample_commands(envs_idx)

    def reset(self):
        """
        重置所有环境
        """
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def get_observations(self):
        return self.obs_buf, self.extras

    def _check_joint_limits(self):
        """检查关节是否超出安全范围"""
        # 大腿关节限制
        thigh_pos = self.dof_pos[:, [0, 2]]
        thigh_violation = torch.any(
            (thigh_pos < self.env_cfg["thigh_angle_min"])
            | (thigh_pos > self.env_cfg["thigh_angle_max"]),
            dim=1,
        )

        # 小腿关节限制
        calf_pos = self.dof_pos[:, [1, 3]]
        calf_violation = torch.any(
            (calf_pos < self.env_cfg["calf_angle_min"])
            | (calf_pos > self.env_cfg["calf_angle_max"]),
            dim=1,
        )

        return thigh_violation | calf_violation

    # ------------ 奖励函数 ------------
    def _reward_tracking_lin_vel(self):
        """线速度跟踪奖励（xy轴）"""
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_gravity_projection(self):
        """
        重力投影奖励 - 保持机器人姿态水平
        理想的重力投影在机体坐标系下应该是 [0, 0, -1]
        惩罚重力在x和y方向的分量
        """
        # 惩罚重力在x和y方向的分量（理想是0）
        # 使用平方误差，鼓励重力投影接近 [0, 0, -1]
        gravity_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-gravity_error / self.reward_cfg.get("gravity_sigma", 0.25))

    def _reward_action_symmetry(self):
        """
        动作对称奖励 - 鼓励左右两侧动作对称
        这对于保持平衡很重要
        """
        # 获取腿部动作（前4个是腿部关节：左大腿、左小腿、右大腿、右小腿）
        leg_actions = self.actions[:, :4]
        
        # 计算左右对称性误差
        # 左大腿 vs 右大腿
        thigh_diff = torch.square(leg_actions[:, 0] - leg_actions[:, 2])
        # 左小腿 vs 右小腿  
        calf_diff = torch.square(leg_actions[:, 1] - leg_actions[:, 3])
        
        # 对称性误差总和
        symmetry_error = thigh_diff + calf_diff
        
        return torch.exp(-symmetry_error / self.reward_cfg.get("symmetry_sigma", 0.25))

    def _reward_wheel_vel_tracking(self):
        """轮子速度跟踪奖励 - 鼓励轮子转动以实现滑行"""
        # 获取轮子速度（假设后两个关节是轮子）
        wheel_vel = self.dof_vel[:, 4:]  # 假设索引4和5是轮子关节
        
        # 根据线速度命令计算期望轮子速度
        # 公式：轮子线速度 = 角速度 * 轮子半径
        # 这里我们简化处理，直接用线速度命令乘以一个系数
        wheel_radius = 0.05  # 轮子半径，根据你的机器人实际情况调整
        target_wheel_vel = self.commands[:, 0] / wheel_radius  # x方向线速度转换为角速度
        
        # 左右轮期望速度（考虑转向）
        # 对于两轮差动机器人：左轮 = (v - w*d/2)/r，右轮 = (v + w*d/2)/r
        wheel_distance = 0.2  # 两轮距离，根据你的机器人实际情况调整
        
        # 计算左右轮期望速度
        target_left_wheel_vel = (self.commands[:, 0] - self.commands[:, 2] * wheel_distance / 2) / wheel_radius
        target_right_wheel_vel = (self.commands[:, 0] + self.commands[:, 2] * wheel_distance / 2) / wheel_radius
        
        # 计算轮子速度误差
        wheel_vel_error = (
            torch.square(wheel_vel[:, 0] - target_left_wheel_vel) + 
            torch.square(wheel_vel[:, 1] - target_right_wheel_vel)
        )
        
        # 使用指数衰减函数，误差越小奖励越高
        return torch.exp(-wheel_vel_error / self.reward_cfg.get("wheel_tracking_sigma", 0.5))
    
    def _reward_wheel_direction(self):
        """轮子转动方向奖励 - 鼓励轮子朝正确方向转动"""
        # 获取轮子速度
        wheel_vel = self.dof_vel[:, 4:]  # 假设索引4和5是轮子关节
        
        # 期望的轮子速度方向（根据命令）
        target_wheel_sign = torch.sign(self.commands[:, 0])  # 根据x方向速度确定期望方向
        
        # 计算实际轮子速度的方向
        wheel_vel_avg = torch.mean(wheel_vel, dim=1)  # 平均轮子速度
        actual_wheel_sign = torch.sign(wheel_vel_avg)
        
        # 方向一致奖励：如果方向一致则奖励，不一致则惩罚
        direction_match = (target_wheel_sign * actual_wheel_sign > 0).float()
        
        return direction_match

    def _reward_wheel_efficiency(self):
        """轮子滑动效率奖励 - 鼓励用轮子而不是踏步"""
        # 计算底座线速度（实际运动速度）
        actual_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
        
        # 计算轮子理论速度（轮子转速 * 半径）
        wheel_radius = 0.05
        wheel_vel = torch.mean(torch.abs(self.dof_vel[:, 4:]), dim=1)  # 平均轮子速度绝对值
        wheel_theoretical_vel = wheel_vel * wheel_radius
        
        # 计算滑动效率：实际速度/理论速度
        # 理想情况下，如果完全用轮子滑动，效率接近1
        # 如果是踏步走，效率会较低
        efficiency = actual_vel / (wheel_theoretical_vel + 1e-6)  # 防止除零
        
        # 限制效率在合理范围内
        efficiency = torch.clamp(efficiency, 0, 1)
        
        return efficiency

    def _reward_jumping_penalty(self):
        """惩罚不必要的跳跃"""
        # 检测基座垂直速度过大
        vertical_vel = torch.abs(self.base_lin_vel[:, 2])
        # 检测足端接触状态突然变化（可能表示跳跃）
        contact_diff = torch.abs(
            torch.diff(self.extras["foot_contact"], dim=1)
        ).sum(dim=1)
        
        jump_penalty = vertical_vel + contact_diff * 0.5
        return jump_penalty  # 负奖励，惩罚跳跃

    # def _reward_tracking_ang_vel(self):
    #     """角速度跟踪奖励（yaw轴）"""
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    # def _reward_lin_vel_z(self):
    #     """惩罚z轴基座线速度"""
    #     return torch.square(self.base_lin_vel[:, 2])

    # def _reward_action_rate(self):
    #     """惩罚动作突变"""
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_similar_to_default(self):
    #     """惩罚关节/轮子偏离默认位姿（使用平方误差）"""
    #     return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        """惩罚基座高度偏离目标值"""
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])



    def _reward_foot_contact(self):
        """奖励稳定的足端接触（防止腾空）"""
        contact_sum = torch.sum(self.extras["foot_contact"], dim=1)
        return torch.square(contact_sum - self.reward_cfg["target_contact_count"])
