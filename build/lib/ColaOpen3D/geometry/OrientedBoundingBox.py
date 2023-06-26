import open3d
import numpy
import ColaOpen3D as o4d

class ColaOrientedBoundingBox():
    def __init__(self, *args, **kwargs) -> None:
        self.data = open3d.geometry.OrientedBoundingBox(*args, **kwargs)

    ## 变换
    def rotate(self, *args, **kwargs):
        """
        args:
            - R (numpy.ndarray[numpy.float64[3, 3]]) The rotation matrix
            - center (option) (numpy.ndarray[numpy.float64[3, 1]])  Rotation center used for transformation.
        """
        return self.data.rotate(*args, **kwargs)


if __name__=="__main__":
    pass