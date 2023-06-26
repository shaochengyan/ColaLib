import numpy as np
import torch
import ColaOpen3D as o4d
import open3d as o3d

from .o3d_utils import pcd_to_ndarray

def integrate_trans(R, t):
    """ (R, t) -> trans
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def transform(pts, trans):
    """rslt = R @ pts + shift
    transform pts, support batch or just one
    Input
        - pts: [num_points, dim] or [bs, num_points, dim] dim >= 3
        - trans:  [4, 4] or [bs, 4, 4]
    Output
        - trans_pts: [num_points, dim] or [bs, num_points, dim]
    """
    if len(pts.shape) == 3:
        trans_pts = torch.einsum('bnm,bmk->bnk', trans[:, :3, :3],
                                 pts.permute(0, 2, 1)) + trans[:, :3, 3:4]  
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = torch.einsum('nm,mk->nk', trans[:3, :3],
                                 pts.T) + trans[:3, 3:4]
        return trans_pts.T


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """ 根据匹配对求解刚体变换(可带权重)
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)  # 升维度，然后变为对角阵
    H = Am.permute(0, 2, 1) @ Weight @ Bm  # permute : tensor中的每一块做转置

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    return integrate_trans(R, t)


def post_refinement(initial_trans, src_kpts, tgt_kpts, iters, weights=None):
    """给定初始变换后，在改变换的基础上，计算所有内点 -> 若内点数更多则通过这些内点重新计算一个新的变换 -> 不断迭代
    Args:
        initial_trans: 初始变换 [4, 4] torch.Tensor
        src_kpts: (N, 3)
        tgt_kpts: (N, 3)
        iters: 迭代次数
        weights: _description_. Defaults to None.
    Returns:
        trans after refinment
    """
    inlier_threshold = 0.1
    pre_inlier_count = 0
    for i in range(iters):
        pred_tgt = transform(src_kpts, initial_trans)
        L2_dis = torch.norm(pred_tgt - src_kpts, dim=-1)
        pred_inlier = L2_dis < inlier_threshold
        inlier_count = torch.sum(pred_inlier)
        if inlier_count <= pre_inlier_count:
            break
        pre_inlier_count = inlier_count
        initial_trans = rigid_transform_3d(
            A=src_kpts[pred_inlier, :],
            B=tgt_kpts[pred_inlier, :],
            weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[pred_inlier]
        )
    return initial_trans


def estimate_normal(pcd, radius=0.06, max_nn=30):
    """estimate normal by neighbor
    Args:
        pcd: o3d point cloud
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


def transformation_error(pred_trans, gt_trans):
    """计算变换误差

    Args:
        pred_trans: 求解的变换
        gt_trans: 真实变换 groud true

    Returns:
        _description_
    """
    pred_R = pred_trans[:3, :3]
    gt_R = gt_trans[:3, :3]
    pred_t = pred_trans[:3, 3:4]
    gt_t = gt_trans[:3, 3:4]

    tr = torch.trace(pred_R.T @ gt_R)
    RE = torch.acos(torch.clamp((tr - 1) / 2.0, min=-1, max=1)) * 180 / np.pi
    TE = torch.norm(pred_t - gt_t) * 100
    return RE, TE


def visualization_2phase(src_pcd, tgt_pcd, pred_trans):
    """vis origin and after trans
    NOTE: press k will change background color
    """
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    key_to_callback = {ord("K"): change_background_to_black}
    if not src_pcd.has_normals():
        estimate_normal(src_pcd)
        estimate_normal(tgt_pcd)
    src_pcd.paint_uniform_color([1, 0.706, 0])
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries_with_key_callbacks([src_pcd, tgt_pcd], key_to_callback)
    src_pcd.transform(pred_trans)
    o3d.visualization.draw_geometries_with_key_callbacks([src_pcd, tgt_pcd], key_to_callback)

def create_corr_via_descriptor_sample(src_desc, tgt_desc):
    """ 直接计算各个描述符之间的距离然后返回匹配对下标
    input: (N, dim) (M, dim) ndarray
    return corr ndarray (N, 2) int
    """
    distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)  # DONE:计算单位向量之间的欧氏距离 
    source_idx = np.argmin(distance, axis=1)  # for each row save the index of minimun
    # feature matching
    corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]],
                        axis=-1)  # n to 1  
    return corr


## 降采样函数
def downsample_random(pts, ratio=None, num=None):
    """
    按照一定比例|数量均匀随机降采样
    pts: ndarray NXdim, dim=3
    """
    N = len(pts)
    # get num of downsample
    if ratio is not None:
        assert ratio > 0 and ratio <= 1
        num = int(N * ratio)
        
    idxs = np.random.permutation(N)[:num]
    return pts[idxs, ...], idxs

def downsample_voxel(pts, voxel: float):
    """
    体素降采样
    """
    pcd_ = o4d.geometry.ColaPointCloud(pts)
    pcd_down = pcd_.voxel_down_sample(voxel)
    return pcd_to_ndarray(pcd_down)


def downsample_fps(pts, n_node=None):
    """
    最远点采样
    """
    pcd_ = o4d.geometry.ColaPointCloud(pts)
    pcd_down = pcd_.farthest_point_down_sample(n_node)
    return pcd_to_ndarray(pcd_down)

def downsample_fps_batch(pts, n_node):
    """
    批量FPS降采样
    pts: BxNx3 ndarray
    return: BxMx3 ndarray
    """
    B, N, _ = pts.shape
    assert n_node <= N
    pts_node = np.zeros(shape=(B, n_node, 3))
    for i in range(B):
        pts_node[i, :, :] = downsample_fps(pts[i], n_node)

    return pts_node


# 计算匹配对的正确匹配率
def calculate_right_correspondence_mask(kpts_src, kpts_dst, corres, radiu, R, shift):
    """
    给定匹配关系和正确变换 -> 正确匹配对的mask
    kpts_src: ndarray Nx3
    kpts_dst: ndarray Mx3
    corres: Kx2 int ndarray
    radiu: float 
    R: ndarray 3x3
    shift: ndarray (3,)
    return: float [0, 1] 匹配对的正确匹配率
    """
    K = len(corres)
    pts_src = kpts_src[corres[:, 0], :]
    pts_dst = kpts_dst[corres[:, 1], :]

    pts_src_transformed = pts_src @ R.T + shift

    dis = np.linalg.norm(pts_src_transformed - pts_dst, axis=1)  # (K,)
    mask = dis < radiu

    return mask

def calculate_right_correspondences_rate(kpts_src, kpts_dst, corres, radiu, R, shift):
    """
    计算正确匹配率
    kpts_src: ndarray Nx3
    kpts_dst: ndarray Mx3
    corres: Kx2 int ndarray
    radiu: float 
    R: ndarray 3x3
    shift: ndarray (3,)
    return: float [0, 1] 匹配对的正确匹配率
    """
    mask = calculate_right_correspondence_mask(kpts_src, kpts_dst, corres, radiu, R, shift)

    return np.sum(mask) / float(len(corres)), np.sum(mask)

