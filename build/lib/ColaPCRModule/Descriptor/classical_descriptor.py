import open3d as o3d
import numpy as np

def fpfh(keypts, downsample):
    """ 输入两个 PCD, 输出keypts的FPFH特征
    Returns:
        (N, 33) 单位化 & ndarray
    """
    keypts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=downsample * 2, max_nn=30))
    features = o3d.pipelines.registration.compute_fpfh_feature(keypts, o3d.geometry.KDTreeSearchParamHybrid(
        radius=downsample * 5, max_nn=100))
    features = np.array(features.data).T
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
    return features

def fpfh2(src_kpts, tgt_kpts, drat=0.05):
    src_desc = fpfh(src_kpts, drat)
    tgt_desc = fpfh(tgt_kpts, drat)
    return src_desc, tgt_desc