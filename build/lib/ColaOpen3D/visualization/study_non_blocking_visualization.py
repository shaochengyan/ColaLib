# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/visualization/non_blocking_visualization.py

import open3d as o3d
import numpy as np

"""
总结: 通过下面函数来不断更新 vis 里面的动态几何体, 还可以据选择性的更新
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcds[i])
def visualize_non_blocking(vis, pcds):
    for pcd in pcds:
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

"""

def test1():
    # 这是官方给出的源码
    # 目的: 非阻塞式展示ICP配准的过程, 动图展示点云如何一步一步配准过来的

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcd_data = o3d.data.DemoICPPointClouds()
    source_raw = o3d.io.read_point_cloud(pcd_data.paths[0])
    target_raw = o3d.io.read_point_cloud(pcd_data.paths[1])

    source = source_raw.voxel_down_sample(voxel_size=0.02)
    target = target_raw.voxel_down_sample(voxel_size=0.02)
    trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7],
             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans)

    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)

    """ 步骤
    - 创建可视化对象 -> 创建window
    - 给可视化对象关联几何体
    - 循环: 对几何体做transform -> poll_events -> 更新渲染器
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=500, height=500)
    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.05
    icp_iteration = 100
    save_image = False

    for i in range(icp_iteration):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        if i == 0:
            input()
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    input()
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

def test2():
    # 这是为了测试我的非阻塞可视化
    # 目的: 非阻塞式展示ICP配准的过程, 动图展示点云如何一步一步配准过来的

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcd_data = o3d.data.DemoICPPointClouds()
    source_raw = o3d.io.read_point_cloud(pcd_data.paths[0])
    target_raw = o3d.io.read_point_cloud(pcd_data.paths[1])

    source = source_raw.voxel_down_sample(voxel_size=0.02)
    target = target_raw.voxel_down_sample(voxel_size=0.02)
    trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7],
             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans)

    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)

    """ 步骤
    - 创建可视化对象 -> 创建window
    - 给可视化对象关联几何体
    - 循环: 对几何体做transform -> poll_events -> 更新渲染器
    """
    from ColaOpen3D import o4d_vis
    vis = o4d_vis.VisDynamic([source, target])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(source)
    # vis.add_geometry(target)
    threshold = 0.05
    icp_iteration = 100
    save_image = False

    for i in range(icp_iteration):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        vis.update_renderer()
        # vis.update_geometry(source)
        # vis.poll_events()
        # vis.update_renderer()
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

if __name__ == "__main__":
    test1()
    # test2()


"""
python -m ColaOpen3D.visualization.study_non_blocking_visualization

"""