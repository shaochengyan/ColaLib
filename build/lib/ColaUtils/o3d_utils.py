import numpy as np
import open3d as o3d

import ColaOpen3D as o4d
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