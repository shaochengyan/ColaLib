import time
import numpy as np
import PCLKeypoint as pcl

"""
https://github.com/lijx10/PCLKeypoints.git
https://github.com/lijx10/PCLKeypoints/blob/master/src/keypoints.cpp

install PCLKeypoint

"""

def run_fpfh33(points:np.ndarray, keypoints, compute_normal_k=10, feature_radius=1.0):
    """
    points: Nx3
    Keypoints: Mx3
    compute_noraml_k: int
    feature_radius: float
    return: ndarray M x 33
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert keypoints.ndim == 2 and keypoints.shape[1] == 3 
    return pcl.featureFPFH33(points, keypoints, compute_normal_k, feature_radius)

def run_fpfh33_with_normal(points:np.ndarray, normals, keypoints,feature_radius=1.0):
    """
    points: Nx3
    normals: ndarray Nx3 normal for the points
    Keypoints: Mx3
    compute_noraml_k: int
    feature_radius: float
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert keypoints.ndim == 2 and keypoints.shape[1] == 3 
    return pcl.featureFPFH33(points, normals, keypoints, feature_radius)


# @record_duration
def run_shot352(points:np.ndarray, keypoints, compute_normal_k=10, feature_radius=1.0):
    """
    points: Nx3
    Keypoints: Mx3
    compute_noraml_k: int
    feature_radius: float
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert keypoints.ndim == 2 and keypoints.shape[1] == 3 
    return pcl.featureSHOT352(points, keypoints, compute_normal_k, feature_radius)



# @record_duration
def run_shot352_with_normal(points:np.ndarray, normals, keypoints,feature_radius=1.0):
    """
    points: Nx3
    Keypoints: Mx3
    compute_noraml_k: int
    feature_radius: float
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert keypoints.ndim == 2 and keypoints.shape[1] == 3 
    return pcl.featureSHOT352(points, keypoints, normals, feature_radius)


if __name__=="__main__":
    # 1. load data
    i = 10
    filebase = "/home/cola/GP/pcr2/Assets/data_debug_semantic/{:06d}.npy"
    (
        pts_rawxyz, labels_sem, src_pc, dst_pc, R, shift, 
        kpts_src_i, kpts_dst_i,  pcd_src_i, pcd_dst_i,  
        dsc_src, dsc_dst
    ) = np.load(filebase.format(i), allow_pickle=True)

    # 2. compute descriptor
    desc = run_fpfh33(src_pc[0].T, kpts_src_i)