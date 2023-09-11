import numpy as np
import torch
import time
import open3d as o3d
import igraph

from ColaUtils import pcr_utils as putils
from ..Assets.ExampleData import load_data
from ..Descriptor.classical_descriptor import fpfh2

def graph_constructe(src_pts, tgt_pts, thresh_d=0.7, thresh_score=0.95):
    src_dist = ((src_pts[:, None, :] - src_pts[None, :, :]) ** 2).sum(-1) ** 0.5 
    tgt_dist = ((tgt_pts[:, None, :] - tgt_pts[None, :, :]) ** 2).sum(-1) ** 0.5
    cross_dis = torch.abs(src_dist - tgt_dist)
    FCG = torch.clamp(1 - cross_dis ** 2 / (thresh_d) ** 2, min=0)  # 区间保持超过则截断函数
    FCG = FCG - torch.diag_embed(torch.diag(FCG))
    FCG[FCG < thresh_score] = 0  # 相容分数小于一定阈值则直接删除
    SCG = torch.matmul(FCG, FCG) * FCG
    return SCG



def max_clique(G_mat, SCG, src_pts, tgt_pts):
    """
    G_mat: NxN ndarray bool adjacent matrix, is compatibal for corr_i and corr_j
    SCG: NxN float ndarray adjacent matrix, grade between corr_i and corr_j
    src_pts, tgt_pts: Nx3 ndarray
    """
    if isinstance(G_mat, torch.Tensor):
        from ColaUtils.torchnp_utils import tensor_to_ndarray
        G_mat, SCG = tensor_to_ndarray([G_mat.cpu(), SCG.cpu()])
    
    N = len(SCG)
    # search cliques
    # input: 对称矩阵SCG, 权重大于0的地方建边，否则不建立边
    graph = igraph.Graph.Adjacency(G_mat.tolist())  # 根据对称矩阵创建图(无权重，1的地方则是边)
    graph.es['weight'] = SCG[SCG.nonzero() ]  # 为所有边添加权重 list of number
    graph.vs['label'] = range(0, N)  # 为所有顶点标号 list 
    graph.to_undirected()  # 无向图
    macs = graph.maximal_cliques(min=3)  # 求解极大团 最小顶点数为3
    print(f'Total: %d' % len(macs))
    
    # 计算每个团的分数 = sum of all node
    # calique score = summ of correspondences SCG score
    clique_weight = np.zeros(len(macs), dtype=float)
    for ind in range(len(macs)):  # for each clique
        mac = list(macs[ind])  # list of node for this clique
        if len(mac) >= 3:  
            # 遍历这里clique的所有权重
            # 上三角
            for i in range(len(mac)):  
                for j in range(i + 1, len(mac)): 
                    clique_weight[ind] = clique_weight[ind] + SCG[mac[i], mac[j]]

    # 每个节点的clique下标
    clique_ind_of_node = np.ones(N, dtype=int) * -1  # TODO: For what?
    # 每个节点权重最大的团的权重
    max_clique_weight = np.zeros(N, dtype=float)
    max_size = 3
    for ind in range(len(macs)):  # for each clique
        mac = list(macs[ind])
        weight = clique_weight[ind]
        if weight > 0:
            for i in range(len(mac)):  # for each node
                # 若该节点的权重小于该团的权重则将该节点的权重归给改团
                if weight > max_clique_weight[mac[i]]: 
                    max_clique_weight[mac[i]] = weight
                    clique_ind_of_node[mac[i]] = ind
                    max_size = len(mac) > max_size and len(mac) or max_size  # 最大团的size

    filtered_clique_ind = list(set(clique_ind_of_node))
    if -1 in filtered_clique_ind:
        filtered_clique_ind.remove(-1)  # 剔除掉没有在任何团中的节点(权重一直为-1)
    print(f'After filtered: %d' % len(filtered_clique_ind))

    # 对mac分组 by size of mac
    group = []  # TODO: 组?
    for s in range(3, max_size + 1):
        group.append([])  # 每一个size对应一个组?
    for ind in filtered_clique_ind:
        mac = list(macs[ind])
        group[len(mac) - 3].append(ind)  # 不同的团大小进入不同的group, 例如 group[4] 对应于大小为4的group

    # [batch1, batch2] 存储多个不同大小团的batch，每个 batch 的大小为 [b, num_mac, 3]
    tensor_list_A = []
    tensor_list_B = []
    for i in range(len(group)): # for each group (mac size)  -> 方便 batch 计算
        # 取出group i的第一个 mac
        batch_A = src_pts[list(macs[group[i][0]])][None]
        batch_B = tgt_pts[list(macs[group[i][0]])][None]
        if len(group) == 1:
            continue
        # 将后续 mac 添加进去
        for j in range(1, len(group[i])):
            mac = list(macs[group[i][j]])
            src_corr = src_pts[mac][None]
            tgt_corr = tgt_pts[mac][None]
            batch_A = torch.cat((batch_A, src_corr), 0)  # cat along batch axis
            batch_B = torch.cat((batch_B, tgt_corr), 0)
        tensor_list_A.append(batch_A)  # list of batch
        tensor_list_B.append(batch_B) 

    # 假设 + 检验
    inlier_threshold = 0.1
    max_score = 0
    final_trans = torch.eye(4)
    for i in range(len(tensor_list_A)):  # for each group
        # 计算每个 batch 对应的 trans -> [b, 4, 4]
        trans = putils.rigid_transform_3d(tensor_list_A[i], tensor_list_B[i], None, 0)
        # 计算匹配源关键点等变换结果
        trans = trans.to(src_pts.dtype)
        pred_tgt = putils.transform(src_pts[None], trans)  # [bs,  num_corr, 3]
        # 计算变换后的的 l2 距离 
        L2_dis = torch.norm(pred_tgt - tgt_pts[None], dim=-1)  # [bs, num_corr]

        # 平均误差分数 TODO: 评估某个变换是否好坏 -> 平均分数
        MAE_score = torch.div(torch.sub(inlier_threshold, L2_dis), inlier_threshold)  # [bs, num_corr], [-inf, 1]
        MAE_score = torch.sum(MAE_score * (L2_dis < inlier_threshold), dim=-1)  # 仅考虑 l2_dis 小于一定阈值的来计算MAE
        max_batch_score_ind = MAE_score.argmax(dim=-1)  # 分数最好的 of each batch 
        max_batch_score = MAE_score[max_batch_score_ind]

        # 若大于全局最大，则认为其是最优的?
        if max_batch_score > max_score:
            max_score = max_batch_score
            final_trans = trans[max_batch_score_ind]
    
    return final_trans    


# DONE: 继续整理代码! 与之前毕设的代码合并! 形成一套体系!
def max_clique_create(src_pts, tgt_pts):
    """
    输入匹配对, 输出配准结果

    Args:
        src_pts: (N, 3) torch.Tensor
        tgt_pts: (N, 3) torch.Tensor
    """
    # check size
    assert src_pts.shape == tgt_pts.shape

    # 根据匹配对之间的相容，创建 N个节点的相容图矩阵 
    SCG = graph_constructe(src_pts, tgt_pts, thresh_d=0.1, thresh_score=0.99)  # (N, N) 的对称阵
    Gmat = SCG > 0
    return Gmat, SCG


def extract_fpfh_features(keypts, downsample):
    keypts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=downsample * 2, max_nn=30))
    features = o3d.pipelines.registration.compute_fpfh_feature(keypts, o3d.geometry.KDTreeSearchParamHybrid(
        radius=downsample * 5, max_nn=100))
    features = np.array(features.data).T
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
    return features

def run_test():
    # load data
    GTmat, src_pcd, tgt_pcd = load_data()
    
    # down sample
    src_kpts = src_pcd.voxel_down_sample(0.05)
    tgt_kpts = tgt_pcd.voxel_down_sample(0.05)
    src_desc = extract_fpfh_features(src_kpts, 0.05)
    tgt_desc = extract_fpfh_features(tgt_kpts, 0.05)

    # create correspondences
    corr = putils.create_corr_via_descriptor_sample(src_desc, tgt_desc) 

    # distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
    # source_idx = np.argmin(distance, axis=1)  # for each row save the index of minimun
    # feature matching
    # corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]],
                        #   axis=-1)  # n to 1
    src_pts = np.array(src_kpts.points, dtype=np.float32)[corr[:,0]]
    tgt_pts = np.array(tgt_kpts.points, dtype=np.float32)[corr[:,1]]

    # to cuda for fast 
    src_pts = torch.from_numpy(src_pts).cuda()  # DONE: why to cuda? 后面用 torch 来计算可以用cuda加速!
    tgt_pts = torch.from_numpy(tgt_pts).cuda()
    GTmat = torch.from_numpy(GTmat).cuda()


    Gmat, SCG = max_clique_create(src_pts, tgt_pts)
    final_trans = max_clique(Gmat, SCG, src_pts, tgt_pts)
    
    # 精细化
    # RE TE
    re, te = putils.transformation_error(final_trans, GTmat)
    # 迭代求解
    final_trans1 = putils.post_refinement(initial_trans=final_trans, src_kpts=src_pts, tgt_kpts=tgt_pts, iters=20)
    # 再一次计算
    re1, te1 = putils.transformation_error(final_trans1, GTmat)
    # 若误差更小，则保留精细化后的结果
    if re1 <= re and te1 <= te:
        final_trans = final_trans1
        re, te = re1, te1

    print(f'RE: %.2f, TE: %.2f' % (re, te))
    final_trans = final_trans.cpu().numpy()
    print(final_trans)
    
    # 可视化
    putils.visualization_2phase(src_pcd, tgt_pcd, final_trans, is_pass_first=False)


if __name__=="__main__":
    run_test()

"""
python -m ColaPCRModules.AdvancePCR.maxcliques
"""