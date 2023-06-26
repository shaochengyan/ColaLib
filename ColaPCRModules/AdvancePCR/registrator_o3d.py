import torch
import numpy as np
import torch.nn as nn
import open3d as o3d

from ColaUtils.torchnp_utils import tensor_to_ndarray, transpose
from ColaUtils.o3d_utils import ndarray_to_pcd



class O3dRegistratorRansac(nn.Module):
    """
    输入关键点及其描述符->创建匹配对->ransac
    """
    def __init__(self) -> None:
        super(O3dRegistratorRansac, self).__init__()

    def to_feature(self, data_dsc):
        """
        data_dsc: N x dim ndarray or tensor
        """
        feature = o3d.pipelines.registration.Feature()
        feature.data = tensor_to_ndarray(data_dsc)
        return feature

    def to_pcd(self, pts):
        """
        pts: N x dim, dim may = 3, ndarray or tensor
        """
        pts = tensor_to_ndarray(pts)
        return ndarray_to_pcd(pts)

    def forward(self, kpts_src, kpts_dst, dsc_src, dsc_dst):
        """
        kpts_src: N1 x 3, ndarray or tensor
        kpts_dst: N1 x dim, ndarray or tensor
        dsc_src: N2 x 3, ndarray or tensor
        dsc_dst: N2 x dim, ndarray or tensor
        return:
            reg_rslt
        """
        pcd_src = self.to_pcd(kpts_src)
        pcd_dst = self.to_pcd(kpts_dst)
        feature_src = self.to_feature(dsc_src)
        feature_dst = self.to_feature(dsc_dst)
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_src,
            pcd_dst,
            feature_src,
            feature_dst,
            mutual_filter=True,
            max_correspondence_distance=2.0,  # 0.02 -> 1
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False),
            ransac_n=3,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                50000, 1000)
        )


class O3dRegistratorBasedCorreRansac(O3dRegistratorRansac):
    """
    输入关键点及其匹配关系(即建立了匹配对之后)
    """
    def __init__(self) -> None:
        super().__init__()

    def to_Vector2iVector(self, arr):
        """
        arr: ndarray Nx2 -> Vector2iVector 
        """
        return o3d.utility.Vector2iVector(arr)

    def forward(self, kpts_src, kpts_dst, corres):
        """
        kpts_src: N1 x 3, ndarray or tensor
        kpts_dst: N1 x dim, ndarray or tensor
        return:
            reg_rslt
        """
        pcd_src = self.to_pcd(kpts_src)
        pcd_dst = self.to_pcd(kpts_dst)
        corres = self.to_Vector2iVector(corres)
        return o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd_src,
            pcd_dst,
            corres,
            max_correspondence_distance=2.0,  # 0.02 -> 1
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False),
            ransac_n=3,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                50000, 1000)
        )
