# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def high_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Points", points)
    for idx in range(0, len(points.points)):
        vis.add_3d_label(points.points[idx], "{}".format(idx))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()


def low_level():
    app = gui.Application.instance
    app.initialize()

    # 创建点云
    points = make_point_cloud(100, (0, 0, 0), 1.0)

    # 创建指定大小的window
    w = app.create_window("Open3D - 3D Text", 1024, 768)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)

    # 创建材料渲染器
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling

    # 给3d窗口添加几何体
    widget3d.scene.add_geometry("Points", points, mat)

    # 每一个点绘制label
    for idx in range(0, len(points.points)):
        # 给点添加lable
        l = widget3d.add_3d_label(points.points[idx], "{}".format(idx))
        # 指定颜色
        l.color = gui.Color(points.colors[idx][0], points.colors[idx][1],
                            points.colors[idx][2])
        # 指定大小
        l.scale = np.random.uniform(0.5, 3.0)
    # 边界框
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()


if __name__ == "__main__":
    high_level()
    low_level()