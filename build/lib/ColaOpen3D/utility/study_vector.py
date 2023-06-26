# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import copy
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    """
    总结: 
        - utility 下面的vector指的是 std::vector, IntVecotr: std::vector<int>, Vector3dVector: std::vector<Eigen::Vector3d>
        - 通过 asarray 可以将其映射到 numpy, 从而被当做 numpy 来处理
        - 由于是std::vector -> 可以调用append来添加元素, extend 来扩展元素
        - 访问方式: a[:], a[0][0]
    """

    print("Testing vector in open3d ...")

    # 构造vector的方式: Python list | numpy array | copy | deepcopy | 片索引
    print("")
    print("Testing o3d.utility.IntVector ...")
    vi = o3d.utility.IntVector([1, 2, 3, 4, 5])  # made from python list
    vi1 = o3d.utility.IntVector(np.asarray(
        [1, 2, 3, 4, 5], dtype=np.int32))  # made from numpy array
    vi2 = copy.copy(vi)  # valid copy
    vi3 = copy.deepcopy(vi)  # valid copy
    vi4 = vi[:]  # valid copy
    print(vi)
    print(np.asarray(vi))
    vi[0] = 10
    np.asarray(vi)[1] = 22
    vi1[0] *= 5
    vi2[0] += 1
    vi3[0:2] = o3d.utility.IntVector([40, 50])
    print(vi)
    print(vi1)
    print(vi2)
    print(vi3)
    print(vi4)

    print("")
    print("Testing o3d.utility.DoubleVector ...")
    vd = o3d.utility.DoubleVector([1, 2, 3])
    vd1 = o3d.utility.DoubleVector([1.1, 1.2])
    vd2 = o3d.utility.DoubleVector(np.asarray([0.1, 0.2]))
    print(vd)
    print(vd1)
    print(vd2)
    vd1.append(1.3)  # 添加一个数据
    vd1.extend(vd2)  # 扩展同类数据
    print(vd1)

    print("")
    print("Testing o3d.utility.Vector3dVector ...")
    vv3d = o3d.utility.Vector3dVector([[1, 2, 3], [0.1, 0.2, 0.3]])
    vv3d1 = o3d.utility.Vector3dVector(vv3d)
    vv3d2 = o3d.utility.Vector3dVector(np.asarray(vv3d))
    vv3d3 = copy.deepcopy(vv3d)
    print(vv3d)
    print(np.asarray(vv3d))

    # 访问方式
    vv3d[0] = [4, 5, 6]  # 可以直接赋值
    print(np.asarray(vv3d))
    # bad practice, the second [] will not support slice
    vv3d[0][0] = -1  # 需要用多层访问
    print(np.asarray(vv3d))
    # good practice, use [] after converting to numpy.array
    # 转为numpy后赋值(引用)
    np.asarray(vv3d)[0][0] = 0  
    print(np.asarray(vv3d))
    np.asarray(vv3d1)[:2, :2] = [[10, 11], [12, 13]]  # 
    print(np.asarray(vv3d1))
    vv3d2.append([30, 31, 32])
    print(np.asarray(vv3d2))

    vv3d3.extend(vv3d)
    print(np.asarray(vv3d3))

    print("")
    print("Testing o3d.utility.Vector3iVector ...")
    vv3i = o3d.utility.Vector3iVector([[1, 2, 3], [4, 5, 6]])
    print(vv3i)
    print(np.asarray(vv3i))

    print("")