import open3d
import numpy
import torch
from .KDTreeFlann import KDTreeFlann

from ColaUtils import pcdvis_utils
from ColaUtils.sequence_utils import normalize
"""
http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
"""


class ColaPointCloud():
    def __init__(self, obj=None) -> None:
        """
        Overloaded function.
            __init__(self: open3d.cpu.pybind.geometry.PointCloud) -> None
        Default constructor
            __init__(self: open3d.cpu.pybind.geometry.PointCloud, arg0: open3d.cpu.pybind.geometry.PointCloud) -> None
        Copy constructor
            __init__(self: open3d.cpu.pybind.geometry.PointCloud, points: open3d.cpu.pybind.utility.Vector3dVector) -> None
        ColaDO:
            args[0]: numpy.darray [N, 3] float
        """
        
        if obj is not None:
            # 已有的点云来初始化
            if isinstance(obj, open3d.geometry.PointCloud):
                self.data = obj
            # 从 array 来初始化
            elif isinstance(obj, numpy.ndarray):
                assert obj.ndim == 2 and obj.shape[1] == 3
                self.data = open3d.geometry.PointCloud()
                self.cola_init_points(obj)
            else:
                self.data = open3d.geometry.PointCloud()
        else:
            self.data = open3d.geometry.PointCloud()
        
        # array
        self.init_torchnp()

    """
    Cola do
    """
    def cola_init_points(self, arr):
        """
        用 arr 来初始化点, 此时是深度拷贝
        arr: ndarray float (N, 2|3)
        """
        self.data.points = open3d.utility.Vector3dVector(arr)
        self.init_torchnp()
    
    def cola_resort_point(self, idx_new):
        """
        idx_new: ndarray (N, ) int
        """
        assert len(idx_new) == len(self.arr)
        self.arr[:, :] = self.arr[idx_new, :]

    def init_torchnp(self):
        self.arr = numpy.asarray(self.data.points)
        self.tensor = torch.from_numpy(self.arr)
        self.normals = None

    def cola_set_colors(self, arr):
        """
        arr: list | ndarray
        """
        return pcdvis_utils.set_geometry_color(self.data, arr)

    def __len__(self):
        """
        return number of point (int)
        """
        return len(self.arr)

    def cola_set_color_with_label(self, label, color_map=None):
        """
        label: ndarray (N, ) or list of it
        color_map: dict label as key, color rgb as value, label:[r, g, b]
        """
        pcdvis_utils.set_pcd_with_semantic_label(self.data, label=label, cmap=color_map) 

    def cola_change_points_value(self, idx_start, idx_end, arr):
        """
        idx_start, idx_end: int, <
        arr: ndarray
        """
        self.arr[idx_start:idx_end, :] = arr

    def estimate_normals_from_other(self, pcd, radius=0.1, max_nn=30, fast_normal_computation=True):
        """
        input: pcd ColaPointCloud
        从pts中估计法向: 先计算pts的法向 -> 在pts中搜搜距离this pcd 的点最近的k个点 -> 平均作为该点的法向
        NOTE: 由于需要计算pts的全部点 -> 不适合大量使用, PCL中有直接从另一个点云中计算法向的函数
        """
        pcd.estimate_normals(radius=radius, max_nn=max_nn, fast_normal_computation=fast_normal_computation)
        tree = KDTreeFlann()
        tree.set_geometry(pcd.data)
        normals = numpy.zeros_like(self.arr)
        for i, query in enumerate(self.arr):
            [_, idx, _] = tree.search_knn_vector_3d(query, knn=3)
            normals[i] = normalize(numpy.sum(pcd.normals[idx], axis=0))
        self.normals = normals
    
    """
    Some compute function
    """
    def estimate_normals(self, radius=0.1, max_nn=30, fast_normal_computation=True):
        """
        For pcd.normals (ndarray Nx3)
        """
        self.data.estimate_normals(
            open3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        self.normals = numpy.asarray(self.data.normals)
    

    """
    Others
    """

    def dimension(self):
        """
        return: int 2 or 3
        """
        return self.data.dimension()

    def is_empty(self):
        return self.data.is_empty()

    def paint_uniform_color(self, color):
        """
        color (numpy.ndarray[numpy.float64[3, 1]]) RGB color for the PointCloud.
        """
        if not isinstance(color, numpy.ndarray):
            color = numpy.asarray(color).reshape(-1, 1)[:3, 0]
        return self.data.paint_uniform_color(color)

    """
    Point Cloud pre-process
    """
    def farthest_point_down_sample(self, num_samples:int):
        """ FPS 采样
        """
        pcd = self.data.farthest_point_down_sample(num_samples)
        return ColaPointCloud(pcd)

    def random_down_sample(self, sampling_ratio):
        """ 产生随机采样下标 -> 随机降采样
        sampling_ratio (float) Sampling ratio, the ratio of number of selected points to total number of points[0-1]
        return: pcd
        """
        return self.data.random_down_sample(sampling_ratio)

    def voxel_down_sample(self, voxel_size):
        """ 体素降采样: 给定体素大小进行体素降采样, norams/colors将会取平均
        """
        pcd = self.data.voxel_down_sample(voxel_size)
        return ColaPointCloud(pcd)

    def remove_non_finite_points(self, remove_nan=True, remove_infinite=True):
        """ remove nan entry, or infinite entries, and  associated with the non-finite point such as normals
        remove_nan (bool, optional, default=True) Remove NaN values from the PointCloud
        remove_infinite (bool, optional, default=True) Remove infinite values from the PointCloud
        """
        return self.data.remove_non_finite_points(remove_nan, remove_infinite)

    def remove_radius_outlier(self, nb_points, radius, print_progress=False):
        """ 一个点周围一定半径内的邻居少于一定值 -> 异常值 -> 去除
        nb_points (int) – Number of points within the radius
        radius (float) – Radius of the sphere.
        print_progress (bool, optional, default=False) – Set to True to print progress bar.
        """
        return self.data.remove_radius_outlier(nb_points, radius, print_progress=False)

    def crop(self, *args, **kwargs):
        """ 根据包围盒(两种)来进行裁剪
        args:
            bounding_box:
        """
        return self.data(*args, **kwargs)

    ## 变换
    def rotate(self, *args, **kwargs):
        """ 旋转
        args:
            - R (numpy.ndarray[numpy.float64[3, 3]]) The rotation matrix
            - center (option) (numpy.ndarray[numpy.float64[3, 1]])  Rotation center used for transformation.
        """
        return self.data.rotate(*args, **kwargs)

    def translate(self, *args, **kwargs):
        """ 平移变换
        NOTE: 
        args:
            - translation (numpy.ndarray[numpy.float64[3, 1]]) A 3D vector to transform the geometry
            - relative (bool, optional, default=True) If true, the translation vector is directly added to the geometry coordinates. Otherwise, the center is moved to the translation vector.center used for transformation.
        """
        return self.data.translate(*args, **kwargs)

    def transform(self, *args, **kwargs):
        """ 旋转平移变换
        args:
            - arg0 (numpy.ndarray[numpy.float64[4, 4]])
        """
        return self.data.transform(*args, **kwargs)

    


if __name__=="__main__":
    import numpy as np
    pcd = ColaPointCloud()
    pcd.cola_init_points(np.random.randn(100, 3))
    print(pcd.data)
    
    pcd.cola_get_ndarray_points()[:10, :3] = 0
    pcd2 = ColaPointCloud(pcd.data)

    print(pcd2.cola_get_ndarray_points()[:11, :3])
    

