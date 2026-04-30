import genesis as gs
import numpy as np


def main():
    # 初始化Genesis，使用GPU后端
    gs.init(backend=gs.gpu)

    # 创建场景
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.002,  # 匹配XML中的timestep
        ),
    )

    # 添加地面和机器人实体
    plane = scene.add_entity(morph=gs.morphs.Plane())
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(file="../assets/JQG12/MJCF/JQG12_straight.xml"),
    )

    # 获取IMU安装的link（对应XML中的flat_cube body）
    # 先找到flat_cube link
    imu_link = robot.get_link("flat_cube")

    # 添加Genesis原生IMU传感器（推荐用法）
    imu_sensor = scene.add_sensor(
        gs.sensors.IMU(
            # 传感器挂载位置
            entity_idx=robot.idx,
            link_idx_local=imu_link.idx_local,
            pos_offset=(0.0, 0.0, 0.0),  # 匹配XML中的imu_site位置
            euler_offset=(0.0, 0.0, 0.0),  # 匹配XML中的imu_site朝向
            # 传感器特性配置（模拟真实IMU的噪声和误差）
            acc_cross_axis_coupling=(0.0, 0.01, 0.02),  # 加速度计轴耦合
            gyro_cross_axis_coupling=(0.03, 0.04, 0.05),  # 陀螺仪轴耦合
            acc_noise=(0.01, 0.01, 0.01),  # 加速度计噪声
            gyro_noise=(0.01, 0.01, 0.01),  # 陀螺仪噪声
            acc_random_walk=(0.001, 0.001, 0.001),  # 加速度计随机游走
            gyro_random_walk=(0.001, 0.001, 0.001),  # 陀螺仪随机游走
            delay=0.002,  # 传感器延迟，需要为仿真dt的整数倍
            jitter=0.001,  # 时间抖动
            interpolate=True,  # 插值延迟数据
            draw_debug=True,  # 在视窗中显示传感器坐标系
        )
    )

    # 构建场景
    scene.build()

    # 打印传感器信息
    print("=" * 60)
    print("IMU传感器已初始化")
    print(f"挂载实体ID: {robot.idx}")
    print(f"挂载Link: {imu_link.name} (local index: {imu_link.idx_local})")
    print("=" * 60)

    # 仿真主循环
    step_count = 0
    try:
        while True:
            # 执行仿真步
            scene.step()

            # 每20帧读取并打印一次数据（避免刷屏）
            if step_count % 20 == 0:
                # 读取带噪声的测量数据（模拟真实传感器）
                measured_data = imu_sensor.read()
                # 读取无噪声的真值数据（用于对比/调试）
                ground_truth_data = imu_sensor.read_ground_truth()

                # 打印数据
                print(f"\n=== 仿真步数: {step_count} ===")
                print("📏 测量数据（带噪声）:")
                print(f"   线性加速度: {np.round(measured_data.lin_acc, 4)} m/s²")
                print(f"   角速度:     {np.round(measured_data.ang_vel, 4)} rad/s")

                print("🎯 真值数据（无噪声）:")
                print(f"   线性加速度: {np.round(ground_truth_data.lin_acc, 4)} m/s²")
                print(f"   角速度:     {np.round(ground_truth_data.ang_vel, 4)} rad/s")

            step_count += 1

    except KeyboardInterrupt:
        # 捕获Ctrl+C优雅退出
        print("\n\n仿真结束！")
        scene.close()


if __name__ == "__main__":
    main()
