import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
import numpy as np
import os
import bisect
from dotmap import DotMap
import pykitti

# cola
import ColaOpen3D as o4d
from ColaUtils import pcr_utils, pcdvis_utils

"""
ColaDO: add some attribute to pykiuit.odometry
    - self.poses_vel:  # 在velodyne坐标系下相机的姿态信息
    - self.labels_sem:  # 语义信息的标注
"""
class ColaOdometry(pykitti.odometry):
    def __init__(self, base_path, sequence, **kwargs):
        super().__init__(base_path, sequence, **kwargs)

        # just add pose_vel
        self._load_pose_vel()
        
        # for lidar semantic labels
        self.fb_sem = os.path.join(base_path, "./sequences/{}/labels".format(sequence), "./{:06d}.label") 

    def _load_pose_vel(self):
        # Tr and TrI
        Tr = self.calib.T_cam0_velo
        TrI = np.zeros((4, 4), dtype=np.float32)
        TrI[:3, :3] = Tr[:3, :3].T
        TrI[:3, 3] = -Tr[:3, :3].T @ Tr[:3, 3]
        TrI[3, 3] = 1     

        self.poses_vel = TrI[None] @ self.poses @ Tr[None]
    
    def get_pose_vel(self, idx):
        return self.poses_vel[idx]
    
    def get_sem_vel(self, idx, is_ins=False):
        """
        return: two ndarray (N, ) of each point  
        """
        file_label = self.fb_sem.format(idx)
        labels = np.fromfile(file_label, dtype=np.uint32).reshape(-1) 
        labels_sem = labels & 0xFFFF  # 高16位构成label
        labels_ins = labels >> 16  #  低16位构成实体的id
        labels_sem = labels_sem.astype(np.int32)
        labels_ins = labels_ins.astype(np.int32)
        
        if is_ins:
            return labels_sem, labels_ins
        else:
            return labels_sem


class ColaKITTIUtils(data.Dataset):
    def __init__(self, 
                 dir_kitti, 
                 seq_list,  # list of int, [0, 1, 2]
                 ) -> None:
        super().__init__()
        self.dir_kitti = dir_kitti
        self.seq_list = seq_list
        
        # create pykitti loader -> list
        self._load_kiters_and_accum_len()

    def idx_map(self, idx_all):
        """
        idx_all -> idx_kiter, idx_sub
        NOTE: self.seq_list[idx_kiter] -> idx_seq
        """
        # idx_all < len and >= 0
        assert idx_all < len(self) and idx_all >= 0

        # conver
        idx_kiter = bisect.bisect_left(
            a=self.accum_lens, 
            x=idx_all+1)

        num_pre = self.accum_lens[idx_kiter-1] if idx_kiter > 0 else 0
        idx_sub = (idx_all + 1 - num_pre) - 1
        return idx_kiter, idx_sub

    def __len__(self):
        return self.accum_lens[-1]   

    def _load_kiters_and_accum_len(self):
        self.kiters = []
        lens = []
        for seq in self.seq_list:
            kiter = ColaOdometry(self.dir_kitti, sequence="{:02d}".format(seq))
            self.kiters.append(kiter)
            lens.append(len(kiter))
        
        self.accum_lens = np.cumsum(lens)
    
    def forward(self, idx_all):
        pass
    
    def get_item_pcr(self, idx_all, is_reflect=False):
        """
        return: lidar, pose, sem,  
        """
        idx_kiter, idx_sub = self.idx_map(idx_all)

        lidar = self.kiters[idx_kiter].get_velo(idx_sub)
        if not is_reflect:
            lidar = lidar[:, :3]
        pose_vel = self.kiters[idx_kiter].get_pose_vel(idx_sub)
        label_sem = self.kiters[idx_kiter].get_sem_vel(idx_sub)

        assert len(lidar) == len(label_sem)

        return lidar, pose_vel, label_sem



class ColaKITTIPCR(ColaKITTIUtils):
    """
    用于点云配准的数据读取
    """
    def __init__(self, 
                 dir_kitti, 
                 seq_list, 
                 **kwargs
                 ) -> None:
        """
        num_downsample: int 
        is_perturbation: bool
        """
        super().__init__(dir_kitti, seq_list)

        # for data config
        self.num_downsample = kwargs.get("num_downsample", 16384)
        self.is_perturbation = kwargs.get("is_perturbation", False)

        self.mode = kwargs.get("mode", 1) 

        # for mode2
        self.step_velo = kwargs.get("step_velo", 11)
    
    def __len__(self):
        return super().__len__()

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 1:
            return self.get_mode1(index)
        elif self.mode == 2:
            return self.get_mode2(index)
        else:
            return None

    def get_mode1(self, idx_all):
        # 获取随机变换
        # randome transformation: create T -> 变换
        angles = np.random.uniform(0, 1, (3, 1)) * np.pi # [0, 1]
        R = o4d.geometry.get_rotation_matrix_from_axis_angle(angles).astype(np.float32) # 各个轴旋转的角度 -> 旋转矩阵
        shift = np.random.uniform(low=-10, high=20, size=(1, 3)).astype(np.float32)

        T = np.zeros((4, 4, ), dtype=np.float32)
        T[:3, :3] = R 
        T[:3, 3] = shift
        T[3, 3] = 1

        # get raw data
        pts_raw, _, label_sem = self.get_item_pcr(idx_all, is_reflect=False)
        pts_raw2 = pts_raw @ R.T + shift


        # 对原始点云按照进行随机降采样 -> 得到两个不同降采样结果的点云
        pts_down1, idxs_down1 = pcr_utils.downsample_random(pts_raw, num=self.num_downsample)
        label_down1 = label_sem[idxs_down1]

        pts_down2, idxs_down2 = pcr_utils.downsample_random(pts_raw2, num=self.num_downsample)
        label_down2 = label_sem[idxs_down2]
        pts_down2 = pts_down2 @ R.T + shift  # (N, 3)@(3, 3)+(1, 3)

        # 噪声扰动: 给每一个点坐标添加加性正态分布的噪声扰动
        if self.is_perturbation:
            sigma = 0.1
            pts_down1 += np.random.randn(*pts_down1.shape) * sigma
            pts_down2 += np.random.randn(*pts_down2.shape) * sigma


        rslt = \
        (
            pts_raw, label_sem, 
            pts_raw2, label_sem, 
            pts_down1, label_down1, 
            pts_down2, label_down2,  
            T
        )

        return rslt
    
    def get_mode2(self, idx_all, step_velo=None):
        # para set
        if step_velo is None:
            step_velo = self.step_velo
        
        # 检查两帧是否在同一seq   若不是则选择最后一帧
        # TODO: 回环
        idx_src = idx_all
        idx_dst = idx_all + step_velo
        idx_kiter1, _ = self.idx_map(idx_src)
        idx_kiter2, _ = self.idx_map(idx_dst)
        if idx_kiter1 != idx_kiter2:
            idx_dst = len(self.accum_lens[idx_kiter1]) - 1  # 选择最后一帧

        # get raw data
        pts_raw_src, pose_vel_src, label_raw_src = self.get_item_pcr(idx_src, is_reflect=False)
        pts_raw_dst, pose_vel_dst, label_raw_dst = self.get_item_pcr(idx_dst, is_reflect=False)

        # down sample
        pts_src_down, idx_src = pcr_utils.downsample_random(pts_raw_src, num=self.num_downsample)
        pts_dst_down, idx_dst = pcr_utils.downsample_random(pts_raw_dst, num=self.num_downsample)
        label_down_src = label_raw_src[idx_src]
        label_down_dst = label_raw_dst[idx_dst]

        # raw T
        T = np.linalg.inv(pose_vel_dst) @ pose_vel_src

        # return 
        rslt = \
            (
                pts_raw_src, label_raw_src, 
                pts_raw_dst, label_raw_dst, 
                pts_src_down, label_down_src, 
                pts_dst_down, label_down_dst, 
                T 
            )
        return rslt
    

def test_mode1(kiter:ColaKITTIPCR):
    """
    读取mode1数据 -> 可视化(语义信息、变换参数)
    """
    idx = np.random.permutation(len(kiter))
    for i in idx:
        (
            pts_raw_src, label_raw_src, 
            pts_raw_dst, label_raw_dst, 
            pts_src_down, label_down_src, 
            pts_dst_down, label_down_dst, 
            T 
        ) = kiter.get_mode1(i)


        # vis
        # 原始数据
        pcd_raw_ = o4d.geometry.ColaPointCloud(pts_raw_src)
        pcdvis_utils.set_pcd_with_semantic_label(pcd_raw_.data, label_raw_src)
        pcdvis_utils.vis_geo_static([pcd_raw_.data])

def test_mode2(kiter:ColaKITTIPCR):
    """
    读取mode1数据 -> 可视化(语义信息、变换参数)
    """
    for i in range(len(kiter)):
        (
            pts_raw_src, label_raw_src, 
            pts_raw_dst, label_raw_dst, 
            pts_src, label_src, 
            pts_dst, label_dst, 
            T 
        ) = kiter.get_mode2(np.random.randint(len(kiter)))


        # vis
        pcd_src_ = o4d.geometry.ColaPointCloud(pts_src)
        pcdvis_utils.set_pcd_with_semantic_label(pcd_src_.data, label_src)
        pcdvis_utils.vis_geo_static([pcd_src_.data])


def read_kitti_velody(filepath: str, is_xyz=True, is_four_col=True) -> np.ndarray:
    """
    filepath: path to velodye xxx.bin
    is_four: is the .bin file store for columns
    is_xyz: is return (x, y, z) or return (x, y, z, r)
    return: np.adarray (x, y, z, r)
    """
    data = np.fromfile(filepath, dtype=np.float32, sep="").reshape(-1, 4 if is_four_col else 3)
    return data[:, :3] if is_xyz else data


def test_dataloader(ds):
    dataloaer = torch.utils.data.DataLoader(
        ds, 
        batch_size=1, 
        shuffle=True, 
        pin_memory=False)
    
    for data in dataloaer:
        print(data)
        break
    

if __name__=="__main__":
    kiter = ColaKITTIPCR(
        "/home/cola/datasets/kitti_odometry", 
        # [1, 4],
        [0, 5],
        step_velo=100
    )


    # test function
    # test_mode1(kiter)
    # test_mode2(kiter)
    test_dataloader(kiter)

"""
python -m ColaDatasets.kitti_loader_gp
"""

