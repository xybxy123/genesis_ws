import os

import numpy as np

import genesis as gs


MJCF_PATH = "/home/xybxy123/w_work/genesis_ws/quadruped_rl/assets/mujoco_menagerie/unitree_go2/go2_mjx.xml"

CONTROL_JOINT_NAMES = [
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

DEFAULT_DOF_POS = np.array(
    [0.0, 0.0, 0.0, 0.0, 1.16, 1.16, 1.16, 1.16, -2.72, -2.72, -2.72, -2.72],
    dtype=np.float32,
)


def main():
    print(f"尝试加载 MJCF: {MJCF_PATH}")
    if not os.path.exists(MJCF_PATH):
        raise FileNotFoundError(f"MJCF 文件未找到: {MJCF_PATH}")

    gs.init(backend=gs.gpu)

    show_viewer = os.environ.get("GENESIS_SHOW_VIEWER", "1") != "0"

    scene = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 3.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            max_FPS=60,
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    scene.add_entity(gs.morphs.Plane())
    go2_robot = scene.add_entity(gs.morphs.MJCF(file=MJCF_PATH))
    go2_robot.pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)

    joint_info = {joint.name: joint for joint in go2_robot.joints}
    control_dofs_idx = [joint_info[name].dofs_idx[0] for name in CONTROL_JOINT_NAMES]

    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=go2_robot.idx,
            link_idx_local=go2_robot.base_link.idx_local,
            pos_offset=(-0.02557, 0.0, 0.04232),
        )
    )

    scene.build()

    go2_robot.set_dofs_position(
        DEFAULT_DOF_POS, dofs_idx_local=control_dofs_idx, zero_velocity=True
    )
    go2_robot.set_dofs_kp(
        np.full(len(control_dofs_idx), 80.0, dtype=np.float32),
        dofs_idx_local=control_dofs_idx,
    )
    go2_robot.set_dofs_kv(
        np.full(len(control_dofs_idx), 5.0, dtype=np.float32),
        dofs_idx_local=control_dofs_idx,
    )
    go2_robot.control_dofs_position(DEFAULT_DOF_POS, dofs_idx_local=control_dofs_idx)

    print("开始读取 IMU 和关节传感器数据")
    print(
        f"IMU link: {go2_robot.base_link.name}, local idx: {go2_robot.base_link.idx_local}"
    )
    print(f"关节控制索引: {control_dofs_idx}")

    fps = 60
    num_steps = 10 * fps
    for step in range(num_steps):
        scene.step()

        imu_data = imu.read()
        joint_pos = (
            go2_robot.get_dofs_position(control_dofs_idx)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        joint_vel = (
            go2_robot.get_dofs_velocity(control_dofs_idx)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        if step % fps == 0:
            sec = step // fps
            print(
                f"[{sec}s] IMU lin_acc = {imu_data.lin_acc.detach().cpu().numpy().reshape(-1)}"
            )
            print(
                f"      IMU ang_vel = {imu_data.ang_vel.detach().cpu().numpy().reshape(-1)}"
            )
            print(
                f"      IMU mag     = {imu_data.mag.detach().cpu().numpy().reshape(-1)}"
            )
            print(f"      joint pos   = {np.round(joint_pos, 4)}")
            print(f"      joint vel   = {np.round(joint_vel, 4)}")

    print("仿真结束")


if __name__ == "__main__":
    main()
