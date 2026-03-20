import genesis as gs
import os
import sys 

# 1. 先初始化 Genesis（全局 GPU 后端）
gs.init(logging_level="warning", backend=gs.cuda)

# 2. 使用用户提供的绝对 URDF 路径
URDF_PATH = "/home/xybxy123/w_work/genesis_ws/rc_wl/assets/urdf/wheel_leg.urdf"

def load_and_inspect_robot():
    print(f"尝试加载 URDF 文件: {URDF_PATH}")
    
    # 验证文件是否存在
    if not os.path.exists(URDF_PATH):
        print(f"❌ 错误：URDF 文件未找到！")
        print(f"  请确认路径是否正确，或文件是否存在于该位置")
        return

    # 3. 动态加载 RigidSolver（无需手动指定 device）
    try:
        RigidSolver = sys.modules['genesis.engine.solvers.rigid.rigid_solver_decomp'].RigidSolver
    except KeyError:
        from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
        
    # 4. 创建场景（仅保留必要配置）
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0, 1.0),  # 调整相机位置，适配可能更大的机器人
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        sim_options=gs.options.SimOptions(dt=1/60),
        show_viewer=False,  # 如需可视化机器人，改为 True
    )
    
    # 5. 加载 URDF（使用用户提供的绝对路径）
    try:
        morph = gs.morphs.URDF(file=URDF_PATH)
    except Exception as e:
        print(f"❌ URDF 加载失败：{e}")
        print(f"  可能原因：URDF 文件格式错误、依赖文件缺失或权限问题")
        return
    
    # 6. 添加机器人实体
    robot = scene.add_entity(morph=morph, visualize_contact=False)
    
    # 7. 构建场景（必须调用，否则连杆数据未初始化）
    scene.build()
    
    print("\n" + "="*80)
    print("✅ 机器人加载成功！")
    print("🔍 正在获取所有连杆信息（包含所有存在的连杆）...")
    print("="*80)
    
    # -------------------------- 核心功能：打印所有连杆 --------------------------
    all_links = []
    try:
        # 从 rigid_solver 中获取所有连杆
        solver_links = scene.rigid_solver.links
        print(f"\n📊 系统中所有连杆总数：{len(solver_links)}")
        print("\n" + "-"*50)
        print("所有连杆的完整信息：")
        print("-"*50)
        
        for idx, link in enumerate(solver_links):
            # 提取连杆关键信息（容错处理）
            link_name = getattr(link, "name", "未知名称")
            link_uid = getattr(link, "uid", "无UID")
            link_idx_local = getattr(link, "idx_local", "无本地索引")
            link_entity_idx = getattr(link, "entity_idx", "无实体索引")
            
            # 标记当前机器人的连杆（通过 entity_idx 匹配）
            is_current_robot = (link_entity_idx == robot.idx)
            robot_mark = " 🤖（当前机器人）" if is_current_robot else " 🚧（其他实体）"
            
            # 打印连杆信息
            print(f"[{idx}] 名称: '{link_name}', UID: {link_uid}, 本地索引: {link_idx_local}, 实体索引: {link_entity_idx}{robot_mark}")
            
            # 收集当前机器人的连杆（后续分析用）
            if is_current_robot:
                all_links.append({
                    "name": link_name,
                    "idx_local": link_idx_local,
                    "uid": link_uid,
                    # 扩展基座候选名称（适配工业机器人常见命名）
                    "is_root_candidate": link_name in ["base_link", "body", "base", "root", "link0", "CJ-003_base"]
                })
    except Exception as e:
        print(f"⚠️ 获取所有连杆失败：{e}")
        return
    
    # -------------------------- 分析当前机器人的连杆 --------------------------
    print("\n" + "="*80)
    print("🤖 当前机器人（CJ-003）的所有连杆（过滤后）：")
    print("="*80)
    
    if all_links:
        for link in all_links:
            root_mark = " 🟢（基座候选）" if link["is_root_candidate"] else ""
            print(f"  - 名称: '{link['name']}', 本地索引: {link['idx_local']}, UID: {link['uid']}{root_mark}")
    else:
        print("❌ 未找到属于当前机器人（CJ-003）的连杆！")
        return
    
    # -------------------------- 给出 CJ-003 机器人的配置建议 --------------------------
    print("\n" + "="*80)
    print("🎯 CJ-003 机器人配置建议（针对你的主程序）：")
    print("="*80)
    
    # 优先推荐工业机器人常见的基座连杆
    root_candidates = [link for link in all_links if link["is_root_candidate"]]
    if root_candidates:
        recommended_root = root_candidates[0]
        print(f"✅ 推荐基座连杆：'{recommended_root['name']}'")
        print(f"   本地索引：{recommended_root['idx_local']}")
        print(f"   说明：该名称符合工业机器人基座命名规范（base_link/link0 等）")
    else:
        # 无明确基座时，推荐第一个连杆（通常是根连杆）
        recommended_root = all_links[0]
        print(f"⚠️ 未找到标准基座名称（base_link/body 等），推荐使用第一个连杆：'{recommended_root['name']}'")
        print(f"   本地索引：{recommended_root['idx_local']}")
        print(f"   说明：URDF 中第一个连杆通常是根连杆（基座）")
    
    # 明确主程序修改步骤
    print(f"\n📌 主程序修改步骤（IMU 配置）：")
    print(f"  1. 打开你的主程序文件")
    print(f"  2. 找到 IMU 配置部分，替换连杆名称：")
    print(f"     原代码：if link.name == 'body':")
    print(f"     修改后：if link.name == '{recommended_root['name']}':")
    print(f"  3. 备选方案（通过本地索引获取，更稳定）：")
    print(f"     body_link = wheel_leg.get_link_by_idx_local({recommended_root['idx_local']})")
    print(f"  4. 注意：确保主程序中 URDF 路径也同步修改为当前路径")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    load_and_inspect_robot()
