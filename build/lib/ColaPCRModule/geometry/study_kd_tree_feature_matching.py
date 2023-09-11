# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Load two aligned point clouds.")
    demo_data = o3d.data.DemoFeatureMatchingPointClouds()
    pcd0 = o3d.io.read_point_cloud(demo_data.point_cloud_paths[0])
    pcd1 = o3d.io.read_point_cloud(demo_data.point_cloud_paths[1])

    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd0, pcd1])

    """
    给定两个匹配后的点云, 计算匹配对之间描述符的相似度, 若距离为0则可是画出来(白点), 否则是黑点
    pipeline: pcd1, pcd2 -> feauture1, feature2 -> 取出feauture1[i]并计算其匹配对象feature2[j] -> 根据 pcd1[i] pcd2[j] 距离判断是否匹配正确
    note: 点可以也被认为是特征, 这里如果以feature为主体,是为了说明feature更加具备区分度
    """
    print("Load their FPFH feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = o3d.io.read_feature(demo_data.fpfh_feature_paths[0])
    feature1 = o3d.io.read_feature(demo_data.fpfh_feature_paths[1])

    fpfh_tree = o3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.colors[i] = [c, c, c]
    o3d.visualization.draw_geometries([pcd0])
    print("")

    print("Load their L32D feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = o3d.io.read_feature(demo_data.l32d_feature_paths[0])
    feature1 = o3d.io.read_feature(demo_data.l32d_feature_paths[1])

    fpfh_tree = o3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.colors[i] = [c, c, c]
    o3d.visualization.draw_geometries([pcd0])
    print("")
