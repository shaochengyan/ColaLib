import open3d
import numpy

import ColaUtils.o3d_utils as utils_o3d
import ColaOpen3D.utility.utility as utility_o4d


def registration_colored_icp(
    source, target, max_correspondence_distance, 
    init=numpy.eye(4, dtype=numpy.float32), 
    estimation_method=open3d.pipelines.registration.TransformationEstimationForColoredICP(), 
    criteria=open3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, 
        relative_rmse=1e-6,
        max_iteration=30)):
    """ 对有颜色的点云进行配准
    source, target: 源/目标点云
    max_correspondence_distance: 两个对应点匹配后的距离阈值,小于该阈值才算成功
    init: 4x4 的初始配准矩阵
    estimation_methd: 计算配准的方法 TODO: 研究不同的方案是什么意思
    criteria: 评判最终成功的标准
    NOTE: ColaOpen3D/pipelines/study/colored_icp_registration.py
    """
    return open3d.pipelines.registration.registration_colored_icp(source, target, max_correspondence_distance, 
    init=init, estimation_method=estimation_method, criteria=criteria)


def registration_icp(
    source, target, max_correspondence_distance, 
    init=numpy.eye(4, dtype=numpy.float32), 
    # estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(), 
    estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPlane(), 
    criteria=open3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, 
        relative_rmse=1e-6,
        max_iteration=30)):
    """ 只需要俩点云即可配准的ICP算法
    NOTE: ColaOpen3D/pipelines/study/icp_registration.py
    args:
        estimation_method: 
            - TransformationEstimationPointToPoint: 点到点
            - TransformationEstimationPointToPlane: 点到面
    """
    return open3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance, 
    init=init, estimation_method=estimation_method, criteria=criteria)


def registration_ransac_based_on_correspondence(
    source, target, corres, max_correspondence_distance, 
    estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
    ransac_n=3, 
    checkers=[open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                      open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)], 
    criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(
                50000, 1000)):
    """ 全局匹配: 基于匹配对进行配准
    corres (open3d.utility.Vector2iVector): o3d.utility.Vector2iVector that stores indices of corresponding point or feature arrays.
    checkers (List[open3d.pipelines.registration.CorrespondenceChecker], optional, default=[]): Vector of Checker class to check if two point clouds can be aligned. One of (CorrespondenceCheckerBasedOnEdgeLength, CorrespondenceCheckerBasedOnDistance, CorrespondenceCheckerBasedOnNormal)
    criteria (open3d.pipelines.registration.RANSACConvergenceCriteria, optional, default=RANSACConvergenceCriteria class with max_iteration=100000, and confidence=9.990000e-01): Convergence criteria
    """
    # if source, target is ndarray
    source, target = utils_o3d.any_to_PointCloud([source, target])
    if isinstance(corres, numpy.ndarray):
        corres = utility_o4d.Vector2iVector(corres)
    return open3d.pipelines.registration.registration_ransac_based_on_correspondence(source, target, corres, max_correspondence_distance, estimation_method=estimation_method, ransac_n=ransac_n, checkers=checkers, criteria=criteria)


def registration_ransac_based_on_feature_matching(
    source, target, source_feature, target_feature, 
    mutual_filter, 
    max_correspondence_distance, 
    estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
    ransac_n=3, 
    checkers=[open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                      open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)], 
    criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(
                50000, 1000)):
    """ 全局匹配: 根据关键点创建特征 -> 构成匹配对 -> 全局匹配
    NOTE: pcd, feature可以是ndarray
        - pcd: ndarray shape=(N, 3)
        - feature: ndarray shape(dim, N) 每一列是一个特征向量 
        - 测试函数: ColaOpen3D/pipelines/study/registration_ransac.py
    source_feature (open3d.pipelines.registration.Feature):Source point cloud feature.
    target_feature (open3d.pipelines.registration.Feature): Target point cloud feature.
    checkers (List[open3d.pipelines.registration.CorrespondenceChecker], optional, default=[]): Vector of Checker class to check if two point clouds can be aligned. One of (CorrespondenceCheckerBasedOnEdgeLength, CorrespondenceCheckerBasedOnDistance, CorrespondenceCheckerBasedOnNormal)
    criteria (open3d.pipelines.registration.RANSACConvergenceCriteria, optional, default=RANSACConvergenceCriteria class with max_iteration=100000, and confidence=9.990000e-01): Convergence criteria
    """
    # if source, target is ndarray
    source, target = utils_o3d.any_to_PointCloud([source, target])
    source_feature, target_feature = utils_o3d.ndarray_to_Feature([source_feature, target_feature])

    return open3d.pipelines.registration.registration_ransac_based_on_feature_matching(source, target, source_feature, target_feature, mutual_filter, max_correspondence_distance, estimation_method=estimation_method, ransac_n=ransac_n, checkers=checkers, criteria=criteria)

