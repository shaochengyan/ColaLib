# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import ColaOpen3D as o4d
import random

NUM_LINES = 10


def random_point():
    return [5 * random.random(), 5 * random.random(), 5 * random.random()]


def main():
    pts = [random_point() for _ in range(0, 2 * NUM_LINES)]
    line_indices = [[2 * i, 2 * i + 1] for i in range(0, NUM_LINES)]
    colors = [[0.0, 0.0, 0.0] for _ in range(0, NUM_LINES)]

    # 创建线条并初始化数据|颜色
    lines_cola = o4d.geometry.ColaLineSet()
    lines = lines_cola.data
    lines_cola.cola_init_lines(pts, line_indices)
    lines_cola.cola_init_colors(colors)

    # 可视化线条
    # 部分设备不需要 OpenGL 来支持 wide lines, 所以通过指定 shader=unlitLine 来定制(只有line可以用,其他shaders将会被忽略)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 10  # note that this is scaled with respect to pixels,
    # so will give different results depending on the
    # scaling values of your system
    o3d.visualization.draw({
        "name": "lines",
        "geometry": lines,
        "material": mat
    })


if __name__ == "__main__":
    main()