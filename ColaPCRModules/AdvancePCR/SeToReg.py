import numpy as np
import ColaOpen3D as o4d
import open3d as o3d
import matplotlib.pyplot as plt

# cola import
from ColaUtils import torchnp_utils
from ColaOpen3D import o4d_utils, o4d_vis


def elimilate_corres_via_semantics(
    kpts_src, kpts_dst, corres1, 
    pcd_src, labels_src, pcd_dst, labels_dst):
    """
    kpts_src kpts_dst: ndarray | tensor (N|M, 3)
    corres1: ndarray (K, 2) int
    pcd_src pcd_dst: (NN|MM, 3) float ndarray | tensor
    labels_src labels_dst: (NN,) (MM, ) int 
    """
    # to ndarray 
    kpts_src, kpts_dst, pcd_src, labels_src, pcd_dst, labels_dst = torchnp_utils.tensor_to_ndarray([kpts_src, kpts_dst, pcd_src, labels_src, pcd_dst, labels_dst])

    # tree_src
    pcd_src_ = o4d_utils.any_to_PointCloud(pcd_src)
    tree_src = o4d.geometry.KDTreeFlann()
    tree_src.set_geometry(pcd_src_)

    # tree dst
    pcd_dst_ = o4d_utils.any_to_PointCloud(pcd_dst)
    tree_dst = o4d.geometry.KDTreeFlann(o4d_utils.any_to_PointCloud(pcd_dst))
    tree_dst.set_geometry(pcd_dst_)

    # check
    corres2 = []
    info_dict = {
        "dynamic": 0, 
        "all_equal": 0, 
        "same": 0, 
        "bad": 0, 
        "not_salience": 0 
    }
    label_dynamic = np.asarray([11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259])
    label_not_salience = np.asarray([40])  # road:40
    label_undfine = np.asarray([0])
    # ax = vis_pcd_plt(pcd_src, color=[1.0, 0.0, 0.0, 1.0], config={"size": 0.5})
    # vis_pcd_plt(kpts_src, ax=ax, color=[0.0, 0.0, 1.0, 1.0], config={"size":20})
    # plt.show()
    for i in range(corres1.shape[0]):
        # query
        query_src = kpts_src[corres1[i, 0]].reshape(3, 1)
        query_dst = kpts_dst[corres1[i, 1]].reshape(3, 1)

        # get label of i-th corrres
        
        [k1, idx1, _] = tree_src.search_radius_vector_3d(query_src, 0.3)
        l_src = np.unique(labels_src[idx1])
        [k2, idx2, _] = tree_dst.search_radius_vector_3d(query_dst, 0.3)
        l_dst = np.unique(labels_dst[idx2])
        # print(l_src, l_dst)

        # visulization
        if False:
            o4d_vis.set_geometry_color(pcd_src_, [0.1, 0.1, 0.1])
            o4d_vis.set_geometry_idx_color(pcd_src_, idx1, [1, 0, 0])

            o4d_vis.set_geometry_color(pcd_dst_, [0.1, 0.1, 0.1])
            o4d_vis.set_geometry_idx_color(pcd_dst_, idx2, [1, 0, 0])

            o4d_vis.vis_geo_static([pcd_src_, pcd_dst_])
            

        # check is pefect corres
        is_perfect = False
        func_is_very_bad = lambda l: l.size == 1 and l[0] in label_not_salience
        if func_is_very_bad(l_src) or func_is_very_bad(l_dst):
            # TODO: 如果不剔除路面
            # is_perfect = False
            is_perfect = True
            info_dict["not_salience"] += 1
        # 1. 检查是否有动态目标: 若有则不匹配
        elif np.intersect1d(l_src, label_dynamic).size != 0 or np.intersect1d(l_dst, label_dynamic) != 0: 
            info_dict["dynamic"] += 1
            is_perfect = False
        else:  # 2. 既然没有动态目标, 则看标签是否类似
            if np.all(l_src == l_dst):  # 完全相等
                info_dict["all_equal"] += 1
                is_perfect = True
            else:  # 3. 不完全相同则要求对称差集合的大小为1(比较少)
                l_xor = np.setxor1d(l_src, l_dst)
                if np.intersect1d(l_xor, label_undfine).size > 0:
                # if l_xor.size == 1:
                    is_perfect = True
                    info_dict["same"] += 1
        
        # store
        if is_perfect:
            corres2.append(corres1[i])
        else:
            info_dict["bad"] += 1

        # print("Y: " if is_perfect else "N ", l_src, '\t', l_dst)
    
    # print(info_dict)
    corres2 = np.asarray(corres2)
    return corres2, info_dict

def elimilate_corres_via_topo(kpts_src, kpts_dst, corres, thresh_dis=10, topo_p=0.05):
    """
    kpts_src kpts_dst: ndarray | tensor (N|M, 3)
    corres1: ndarray (K, 2) int
    """
    N = corres.shape[0]
    grades = np.zeros(shape=(N, ), dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            xi = kpts_src[corres[i, 0]]
            xj = kpts_src[corres[j, 0]]
            
            yi = kpts_dst[corres[i, 1]]
            yj = kpts_dst[corres[j, 1]]

            dis_X = np.sum((xi - xj)**2)
            dis_Y = np.sum((yi - yj)**2)
            # print(dis_X, dis_Y)

            if np.abs(dis_X - dis_Y) < thresh_dis:
                grades[i] += 1
                grades[j] += 1

    # check: 去除分数少于 N * 0.1 的
    thresh_N = int(topo_p * N) - 1
    corres3 = []
    for i in range(N):
        if grades[i] >= thresh_N:
            corres3.append(corres[i, :])
    # print("原始有 {} 个, 剩余 {} 个, 剔除了 {} 个".format(N, len(corres3), N - len(corres3)))
    return np.asarray(corres3)


import open3d as o3d
import ColaOpen3D as o4d
from ColaOpen3D import o4d_utils
import numpy as np

from ColaUtils.torchnp_utils import tensor_to_ndarray

""" test thresh_dis
dsc_src = np.asarray([
    0.0, 0.0, 
    1.0, 1.0, 
    3.0, 3.0
]).reshape(3, 2)

dsc_dst = np.asarray([
    0.0, 0.0, 
    1.0, 1.0, 
    3.0, 3.0
]).reshape(3, 2) + 0.1 * np.random.rand(3, 2)
ColaPyRansac.create_correspondeces(dsc_src.T, dsc_dst.T, 0, 0.01)
"""
def create_correspondences(feature_source, feature_target, is_col_feature=True, is_mutual=False, thresh_dis=1.0) -> np.ndarray:
    """
    args: all double type
        source: ndarray or torch.Tensor (N, 3)
        target: ndarray or torch.Tensor or torch.Tensor (M, 3)
        feature_source: ndarray or torch.Tensor (dim, N) 
        feature_target: ndarray or torch.Tensor (dim, M)
        is_col_feature: is feature_source[:, i] == feautre_i, otherwise feature_source[i, :] == feature_i
    return: ndarray or torch.Tensor (N, 2) int
    """
    is_mutual = False
    feature_source, feature_target = tensor_to_ndarray([feature_source, feature_target])
    if not is_col_feature:
        feature_source, feature_target = feature_source.T, feature_target.T
    corres = ColaPyRansac.create_correspondeces(feature_source, feature_target, is_mutual, thresh_dis)
    return corres.T

# create label for each kpts
def create_group_labels(kpts, tree, labels_sem):
    labels = []
    groups = dict()
    
    # label setting
    label_dynamic = np.asarray([11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259])
    label_not_salience = np.asarray([40])  # road:40
    label_undfine = np.asarray([0])


    # query each keypoint's label and group it
    for i in range(len(kpts)):
        query = kpts[i]
        [k, idx, _] = tree.search_radius_vector_3d(query, 1)
        label = np.unique(labels_sem[idx])
        labels.append(label)

        # print(labels)

        if labels is None:
            continue

        # group via group_label
        group_labels = np.setdiff1d(label, label_undfine) # 去除label中的未定义标签

        if np.intersect1d(group_labels, label_dynamic).size > 0:  # 有动态目标
            continue

        if group_labels.size == 1 and np.intersect1d(group_labels, label_not_salience).size == 1:  # 在路面等直接排除的地方
            # TODO: 不剔除不显著的地方
            pass
            # continue
        
        # create group key
        group_labels.sort()
        group_labels_str = "_".join([str(int(item)) for item in group_labels])
        if group_labels_str not in groups.keys():
            groups[group_labels_str] = []  # 初始化为 list

        # group add the key 
        groups[group_labels_str].append(i)

    return labels, groups


def coreate_correspondences_via_labels(pts_src, pts_dst, label_src, label_dst, kpts_src, kpts_dst, dsc_src, dsc_dst, thresh_dis=0.8, is_mutual=False):
    """
    通过点云的语义信息进行分类匹配
    """
    tree_src = o4d.geometry.KDTreeFlann()
    pcd_src_ = o4d_utils.any_to_PointCloud(pts_src)
    tree_src.set_geometry(pcd_src_)

    pcd_dst_ = o4d.geometry.ColaPointCloud(pts_dst)
    tree_dst = o4d.geometry.KDTreeFlann()
    tree_dst.set_geometry(pcd_dst_.data)

    # group kpts
    _, groups_src = create_group_labels(kpts_src, tree_src, label_src)
    _, groups_dst = create_group_labels(kpts_dst, tree_dst, label_dst)

    corres = []
    # for each group with the same label
    for key in groups_src.keys():
        if key in groups_dst.keys():
            # print(key)
            idxs_src = groups_src[key]
            idxs_dst = groups_dst[key]

            # create correspondeces
            dsc_src_tmp = dsc_src[:, idxs_src]
            dsc_dst_tmp = dsc_dst[:, idxs_dst]
            corres_tmp = create_correspondences(dsc_src_tmp, dsc_dst_tmp, is_col_feature=True, thresh_dis=thresh_dis, is_mutual=is_mutual)
            # print(corres_tmp)
            
            for i in range(len(corres_tmp)):
                idx_src = corres_tmp[i, 0]
                idx_dst = corres_tmp[i, 1]

                # # TODO: 根据描述符阈值来粗略判断是否可以！
                # dis = np.linalg.norm(dsc_src_tmp[..., idx_src] - dsc_dst_tmp[..., idx_dst])
                # if np.all(dis >= thresh_dis):
                #     continue
                    # pass
                # print(dis)
                corres.append([
                    idxs_src[idx_src], 
                    idxs_dst[idx_dst], 
                ])
    corres = np.asarray(corres)
    # corres1 = o4d.utility.Vector2iVector(np.asarray(corres1))
    # print(corres)
    return corres

"""
python -m PCRModules.utils_registration
"""

if __name__=="__main__":
    def test1():
        feature_source, feature_target = np.random.rand(32, 1000), np.random.rand(32, 1100)
        rslt = create_correspondences(feature_source, feature_target)
        print(rslt)
    
    def test2():
        feature1 = np.random.rand(32, 100)
        idx = np.random.permutation(100)
        feature2 = feature1[:, idx]
        rslt = create_correspondences(feature1, feature2)
        print(rslt[idx, 1])  # rslt第二列中中 idx[i] 对应的 是 i，所以打印出来的是顺序数组 -> 可以证明算法的正确性
        print(rslt)
    test2()