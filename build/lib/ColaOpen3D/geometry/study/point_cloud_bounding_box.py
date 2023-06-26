# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)

    # 获取包围盒
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()  # 沿着坐标轴的包围盒
    axis_aligned_bounding_box.color = (1, 0, 0)  
    oriented_bounding_box = pcd.get_oriented_bounding_box()  # 沿着点云方向的包围盒
    oriented_bounding_box.color = (0, 1, 0)
    print(
        "Displaying axis_aligned_bounding_box in red and oriented bounding box in green ..."
    )
    o3d.visualization.draw(
        [pcd, axis_aligned_bounding_box, oriented_bounding_box])
