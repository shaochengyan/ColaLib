import open3d
import numpy
import ColaOpen3D as o4d

"""
http://www.open3d.org/docs/release/python_api/open3d.geometry.AxisAlignedBoundingBox.html#
"""

class ColaAxisAlignedBoundingBox():
    def __init__(self, *args, **kwargs) -> None:
        self.data = open3d.geometry.AxisAlignedBoundingBox(*args, **kwargs)

    @staticmethod
    def create_from_points(points):
        """
        points (open3d.utility.Vector3dVector) A list of points. ndarray | list of points
        """
        if isinstance(points, numpy.ndarray):
            points = o4d.utility.Vector3dVector(points)
        return open3d.geometry.AxisAlignedBoundingBox(points)

    def clear(self):
        """ 删除几何体里面的所有几何体
        """
        return self.data.clear()
    
    def dimension(self):
        """ 返回2D or 3D
        """
        return self.data.dimension()

    ## Get function
    def get_axis_aligned_bounding_box(self):
        """ 返回一个该box对应的与坐标轴平行的几何体
        """
        return self.data.get_axis_aligned_bounding_box()

    def get_oriented_bounding_box(self):
        """ 返回该box对应的有方向的box
        """
        return self.data.get_oriented_bounding_box()

    def get_box_points(self):
        """
        return: 构成该包围盒的8个点 Vector3dVector
        """
        return self.data.get_box_points()

    def get_center(self):
        """
        return: 中心点坐标 ndarray (3, 1) float
        """
        return self.data.get_center()

    def get_extent(self):
        """
        return: 返回沿着xyz轴的延伸范围(包围盒长度) ndarray (3, 1) double
        """
        return self.data.get_extent()

    ## Others
    def is_empty(self):
        return self.data.is_empty()

    ## 变换
    def rotate(self, *args, **kwargs):
        """ 旋转
        NOTE: 必须先转换为OrientedBoundingBox才可以
        args:
            - R (numpy.ndarray[numpy.float64[3, 3]]) The rotation matrix
            - center (option) (numpy.ndarray[numpy.float64[3, 1]])  Rotation center used for transformation.
        """
        if not isinstance(self.data, open3d.geometry.OrientedBoundingBox):
            self.data = self.get_oriented_bounding_box()
        return self.data.rotate(*args, **kwargs)


if __name__=="__main__":
    points = numpy.asarray([[1, 2, 3], [2, 3, 4.0]], dtype=numpy.float64)
    box = ColaAxisAlignedBoundingBox(points[0], points[1])
    print(box.data)

    # 旋转
    from .get_rotation_matrix import get_rotation_matrix_from_xyz
    rot = get_rotation_matrix_from_xyz([1.1, 2, 3])
    print(box.rotate(rot)) 


