import numpy as np
import open3d as o3d
import os

def load_data(folder=None):
    if folder is None:
        folder = os.path.join(os.path.dirname(__file__), "./Data1")

    GTmat_path = folder + '/GTmat.txt' # ground truth transformation for calculate RE TE
    src_pcd_path = folder + '/source.ply'
    tgt_pcd_path = folder + '/target.ply'
    GTmat = np.loadtxt(GTmat_path, dtype=np.float32)
    src_pcd = o3d.io.read_point_cloud(src_pcd_path)
    tgt_pcd = o3d.io.read_point_cloud(tgt_pcd_path)
    return GTmat, src_pcd, tgt_pcd

if __name__=="__main__":
    print(load_data())

