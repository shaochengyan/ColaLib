import open3d
import numpy

"""
http://www.open3d.org/docs/release/python_api/open3d.geometry.LineSet.html
"""

class ColaLineSet:
    def __init__(self, *args, **kwargs):
        """
        Overloaded function.
            __init__(self: open3d.cpu.pybind.geometry.LineSet) -> None
        Default constructor
            __init__(self: open3d.cpu.pybind.geometry.LineSet, arg0: open3d.cpu.pybind.geometry.LineSet) -> None
        Copy constructor
            __init__(self: open3d.cpu.pybind.geometry.LineSet, points: open3d.cpu.pybind.utility.Vector3dVector, lines: open3d.cpu.pybind.utility.Vector2iVector) -> None
        Create a LineSet from given points and line indices
        """
        self.data = open3d.geometry.LineSet(*args, **kwargs)

    
    def cola_init_lines(self, points, line_indices):
        """ 绘制M条线
        points: ndarray (N, 3) float
        line_indices: ndarray (M, 2) int  points[idx[i, 0]] and points[idx[i, 1]] 构成一条线
        """
        self.data.points = open3d.utility.Vector3dVector(points)
        self.data.lines = open3d.utility.Vector2iVector(line_indices)

    def cola_init_colors(self, colors):
        """
        args:
            colors: ndarray (M, 3) float
        example:
            默认是白色
            colors = [[0.0, 0.0, 0.0] for _ in range(0, NUM_LINES)] -> black
        """
        self.data.colors = open3d.utility.Vector3dVector(colors)


"""
python -m ColaOpen3D.visualization.study_line_width
"""
if __name__=="__main__":
    pass