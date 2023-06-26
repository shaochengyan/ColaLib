import time
import PCLKeypoint as pcl
import numpy as np


"""
https://github.com/lijx10/PCLKeypoints
https://github.com/lijx10/PCLKeypoints/blob/master/src/keypoints.cpp
"""

def random_select(pts, num):
    idx = np.random.permutation(len(pts))[:num]
    return pts[idx, :]

def check_if_full_num(pts, kpts, target_num):
    num = len(kpts)
    if num >= target_num:
        return kpts
    
    # fill
    pts_random = random_select(pts, target_num - num)
    return np.vstack([kpts, pts_random])


def record_duration(func):
    def wraper(*args, **kwargs):
        t1 = time.time()
        rst = func(*args, **kwargs)
        t2 = time.time()
        print("Duration: {} ms".format((t2 - t1)*1000))
        return rst
    return wraper


@record_duration
def run_iss(points, iss_salient_radius=3.0, iss_non_max_radius=2.0, iss_gamma_21=0.975, iss_gamma_32=0.975, iss_min_neighbors=5, threads=0, target_num=None):
    """
    points: ndarray Nx3
    return: kpts Ndarra Mx3
    """
    kpts = pcl.keypointIss(points, iss_salient_radius, iss_non_max_radius, iss_gamma_21, iss_gamma_32, iss_min_neighbors, threads)

    if target_num is None:
        return kpts
    else:
        return check_if_full_num(points, kpts, target_num)

@record_duration
def run_Harris3D(points, radius=0.5, nms_threshold=0.001, threads=0, is_nms=False, is_refine=False, target_num=None):
    """
    points: ndarray Nx3
    return: kpts Ndarra Mx3
    """
    kpts = pcl.keypointHarris3D(points, radius, nms_threshold, threads, is_nms, is_refine)
 
    if target_num is None:
        return kpts
    else:
        return check_if_full_num(points, kpts, target_num)

@record_duration
def run_Harris6D(points, radius=0.5, nms_threshold=0.001, threads=0, is_nms=False, is_refine=False, target_num=None):
    """
    points: ndarray Nx3
    return: kpts Ndarra Mx3
    """
    kpts = pcl.keypointHarris6D(points, radius, nms_threshold, threads, is_nms, is_refine)
    if target_num is None:
        return kpts
    else:
        return check_if_full_num(points, kpts, target_num)


@record_duration
def run_Sift(points, min_scale=0.1, n_octaves=6, n_scales_per_octave=10, min_contrast=0.05, target_num=None):
    """
    points: ndarray Nx3
    return: kpts Ndarra Mx3
    """
    kpts = pcl.keypointSift(points, min_scale, n_octaves, n_scales_per_octave, min_contrast)

    if target_num is None:
        return kpts
    else:
        return check_if_full_num(points, kpts, target_num)

