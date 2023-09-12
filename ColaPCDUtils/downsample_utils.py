import numpy as np

# COLA
import ColaOpen3D as o4d


def voxel_downsample_with_label(pts, labels, voxel_size):
    """
    pts: Nx3 ndarray
    labels: Nx1  ndarray
    """
    pcd = o4d.geometry.ColaPointCloud(pts)
    tree = o4d.geometry.KDTreeFlann(pcd.data)

    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # get label
    pts_down = np.asarray(pcd_down.arr)
    labels_down = np.zeros(len(pts_down), dtype=np.int32)
    for i, query in enumerate(pts_down):
        query = query.reshape(3, -1)
        idx = tree.search_knn_vector_3d(query=query, knn=1)[1][0]
        labels_down[i] = labels[idx]
    
    return pts_down, labels_down   

