import open3d
import numpy


def get_rotation_matrix_from_axis_angle(rotation):
    """ [theta1, theta2, theta2]
    NOTE: 单位是度 -> 需要乘以 pi
    numpy.ndarray[numpy.float64[3, 1]] -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    return open3d.geometry.get_rotation_matrix_from_axis_angle(rotation)


def get_rotation_matrix_from_quaternion(rotation):
    """
    rotation: numpy.ndarray[numpy.float64[4, 1]]) → numpy.ndarray[numpy.float64[3, 3]]
    """
    return open3d.geometry.get_rotation_matrix_from_quaternion(rotation)


def get_rotation_matrix_from_xyz(rotation):
    """
    numpy.ndarray[numpy.float64[3, 1]] -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    return open3d.geometry.get_rotation_matrix_from_xyz(rotation)

def get_rotation_matrix_from_xzy(rotation):
    """
    numpy.ndarray[numpy.float64[3, 1]] -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    return open3d.geometry.get_rotation_matrix_from_xzy(rotation)


def get_rotation_matrix_from_yxz(rotation):
    """
    numpy.ndarray[numpy.float64[3, 1]] -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    return open3d.geometry.get_rotation_matrix_from_yxz(rotation)


def get_rotation_matrix_from_yzx(rotation):
    """
    numpy.ndarray[numpy.float64[3, 1]] -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    return open3d.geometry.get_rotation_matrix_from_yzx(rotation)


def get_rotation_matrix_from_zxy(rotation):
    """
    numpy.ndarray[numpy.float64[3, 1]] -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    return open3d.geometry.get_rotation_matrix_from_zxy(rotation)


def get_rotation_matrix_from_zyx(rotation):
    """
    numpy.ndarray[numpy.float64[3, 1]] -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    return open3d.geometry.get_rotation_matrix_from_zyx(rotation)
    



    
    