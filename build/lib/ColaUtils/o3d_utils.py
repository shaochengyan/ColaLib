import numpy as np
import open3d as o3d

from ColaUtils.torchnp_utils import tensor_to_ndarray
from ColaUtils.wrapper import wrapper_data_trans



@wrapper_data_trans
def ndarray_to_pcd(pts):
    """
    ndarray(N, 3) 转为 pcd 支持多个
    """
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) 

@wrapper_data_trans
def ndarray_to_Feature(feature):
    """
    feature: ndarray (dim, N) N个长度为dim的列特征向量 or list of it
    """
    if isinstance(feature, o3d.pipelines.registration.Feature):
        return feature
    feature_o3d = o3d.pipelines.registration.Feature()
    feature_o3d.data = tensor_to_ndarray(feature)
    return feature_o3d


@wrapper_data_trans
def any_to_PointCloud(pts):
    """
    任意Tensor|Ndarray 转为 PointCloud
    pts: Nx3 Tensor|ndarray|PointCloud or list of it
    """
    # print(pts)
    if isinstance(pts, o3d.geometry.PointCloud):
        return pts
    return ndarray_to_pcd(tensor_to_ndarray(pts))


@wrapper_data_trans
def pcd_to_ndarray(pcd):
    return np.asarray(pcd.points)
    
@wrapper_data_trans
def feature_to_ndarray(feature):
    return np.asarray(feature.data)


import numpy as np
import torch
import ColaOpen3D as o4d
import open3d as o3d
from ColaUtils.wrapper import wrapper_data_trans


@wrapper_data_trans
def trans_to_array_tensor(data):
    return PCDArrayTensor(data)

"""
For convinient convert pcd and ndarray and tensor 
"""
class PCDArrayTensor(np.ndarray):
    def __new__(cls, input_array):
        """
        input: o3d/o4d.PointCloud | ndarray |  Tensor 
        """
        if isinstance(input_array, o4d.geometry.ColaPointCloud):
            obj = input_array.arr
        elif isinstance(input_array, o3d.geometry.PointCloud):
            obj = np.asarray(input_array.points)
        elif isinstance(input_array, torch.Tensor):
            obj = input_array.numpy()
        else:
            obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        
        self.tensor = torch.from_numpy(obj)
        self.arr = np.asarray(self)
        self.create_pcd()
    
    # cola do
    def create_pcd(self):
        """
        return ColaPointCloud
        """
        self.pcd = o4d.geometry.ColaPointCloud(self)


if __name__=="__main__":  
    a = ArrayTensor([1, 2, 3.1])
    print(a)
    # 使用示例
    arr = np.array([1, 2, 3])
    arr2 = ArrayTensor(arr)

    arr[0] = -222
    print(arr)
    print(arr2)
    print(type(arr2))
    print(arr2.dtype)
    print(arr2.tensor)

    arr[1] = 100 
    print(arr2.arr)
    print(arr2)

"""
python -m ColaDType.numpy_tensor
"""