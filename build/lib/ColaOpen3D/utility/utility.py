import open3d


"""
http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#
"""


class Vector3dVector(open3d.utility.Vector3dVector):
    def __init__(self, *args, **kwargs) -> None:
        """ 从 numpy.ndarray: float 中构建vector3d, 返回结果可以当做vector来用
        Convert float64 numpy array of shape (n, 3) to Open3D format.
        Overloaded function.
            __init__(self: open3d.cpu.pybind.utility.Vector3dVector) -> None
            __init__(self: open3d.cpu.pybind.utility.Vector3dVector, arg0: numpy.ndarray[numpy.float64]) -> None
            __init__(self: open3d.cpu.pybind.utility.Vector3dVector, arg0: open3d.cpu.pybind.utility.Vector3dVector) -> None
        Copy constructor
            __init__(self: open3d.cpu.pybind.utility.Vector3dVector, arg0: Iterable) -> None
        """
        super().__init__(*args, **kwargs)


class Vector2iVector(open3d.utility.Vector2iVector):
    def __init__(self, *args, **kwargs) -> None:
        """ 从 numpy.ndarray: int 中构建vector2i, 返回结果可以当做vector来用
        Overloaded function.
        NOTE: 
            - input: (N, 2) int ndarray | list of list
        __init__(self: open3d.cpu.pybind.utility.Vector2iVector) -> None
        __init__(self: open3d.cpu.pybind.utility.Vector2iVector, arg0: numpy.ndarray[numpy.int32]) -> None
        __init__(self: open3d.cpu.pybind.utility.Vector2iVector, arg0: open3d.cpu.pybind.utility.Vector2iVector) -> None
        Copy constructor
        __init__(self: open3d.cpu.pybind.utility.Vector2iVector, arg0: Iterable) -> None
        """
        super().__init__(*args, **kwargs)



if __name__=="__main__":
    import numpy as np
    data = np.random.randn(100, 3)
    print(Vector3dVector(data))

    data = np.arange(12).reshape(-1, 2)
    print(Vector2iVector(data))