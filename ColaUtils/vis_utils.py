import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import ColaOpen3D as o4d

# Cola import
from .o3d_utils import any_to_PointCloud, tensor_to_ndarray


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                       for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_pc(pc_np, title=None, z_cutoff=1000, birds_view=False, color='height', size=0.3, ax=None, cmap=cm.jet, is_equal_axes=True):
    # remove large z points
    valid_index = pc_np[:, 0] < z_cutoff
    pc_np = pc_np[valid_index, :]

    if ax is None:
        fig = plt.figure(figsize=(9, 9), facecolor="white")
        ax = Axes3D(fig)
    if type(color) == str and color == 'height':
        c = pc_np[:, 2]
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
                   s=size, c=c, cmap=cmap, edgecolors='none')
    elif type(color) == str and color == 'reflectance':
        assert False
    elif type(color) == np.ndarray:
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
                   s=size, c=color, cmap=cmap, edgecolors='none')
    else:
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
                   s=size, c=color, edgecolors='none')

    if is_equal_axes:
        axisEqual3D(ax)
    if True == birds_view:
        ax.view_init(elev=0, azim=-90)
    else:
        ax.view_init(elev=-45, azim=-90)
    ax.invert_yaxis()

    ax.axis("off")
    # ax.set_xtickllabels([])
    # 调整视角
    ax.view_init(elev=20, azim=50)

    # 设置显示范围
    ax_min = np.min(pc_np, axis=0) - 20 
    ax_max = np.max(pc_np, axis=1) +20
    # print(ax_min, ax_max)
    ax.set_ylim(ax_min[0], ax_max[0]) 
    ax.set_xlim(ax_min[1], ax_max[1]) 
    ax.set_zlim(ax_min[2], ax_max[2]) 

    return ax


def vis_pcd_plt(pts, labels=None, color=None, config=None, ax=None):
    """
    data: Nx3 Tensor|ndarray or list of both of all
    labels: (N, ) Tensor|ndarray
    color: color cmap, cmap(float)->tupe(4), or list of color cmap, ex: [(0.1, 0.2, 0.3), (0.1, 0.2, 0.4)] for two pcd color
    example:
        utils.vis_pcd_plt([pcd_src_i, pcd_dst_i], color=[(0.3, 0.3, 0.3), (0.6, 0.1, 0.1)])
        utils.vis_pcd_plt([utils.pcd_to_ndarray(pcd_src_i_o3d), pcd_dst_i], color=[(0.3, 0.3, 0.3), (0.6, 0.1, 0.1)])
        utils.vis_pcd_plt([kpts_src_i, kpts_dst_i], color=[(1.0, 0, 0), (0, 1, 0)], config={"size": 20})
        config: parameters of plot_pc
        - semmantic: vis.vis_pcd_plt(rslt[2], labels=rslt[3])
    """
    # to list
    if not isinstance(pts, list):
        pts = [pts]
        color = [color]
    # to ndarray
    pts = [tensor_to_ndarray(item) for item in pts]

    # 2. paint color
    if labels is not None:  # 有标签则根据标签产生颜色
        # 1. trans to list of label(ndarray)
        if not isinstance(labels, list):
            labels = [labels]
        labels = [tensor_to_ndarray(item) for item in labels]
        
        # 2. create list of color
        cmap = cm.get_cmap("Spectral")
        color_list = []
        for label in labels:  # each pcd
            color = np.asarray([ cmap(l * 20) for l in label ])  # each point
            color[:, 3] = 1.0
            color_list.append(color)
        color = color_list
    elif color is None:  # 产生一种随机颜色
        cmap = cm.get_cmap("Spectral")
        color = [cmap(item) for item in np.random.rand(len(pts))]
    else:
        color = [np.asarray(item).reshape(-1, 4) for item in color]
    assert len(color) == len(pts)

    # plot
    if config is None:
        config = {}
    ax = plot_pc(pts[0], color=color[0], ax=ax, **config)
    for i in range(1, len(pts)):
        ax = plot_pc(pts[i], color=color[i], ax=ax, **config)
    return ax

color_dict = {
    0 : [0, 0, 0], 
    1 : [0, 0, 255], 
    10: [245, 150, 100], 
    11: [245, 230, 100], 
    13: [250, 80, 100], 
    15: [150, 60, 30], 
    16: [255, 0, 0], 
    18: [180, 30, 80], 
    20: [255, 0, 0], 
    30: [30, 30, 255], 
    31: [200, 40, 255], 
    32: [90, 30, 150], 
    40: [255, 0, 255], 
    44: [255, 150, 255], 
    48: [75, 0, 75], 
    49: [75, 0, 175], 
    50: [0, 200, 255], 
    51: [50, 120, 255], 
    52: [0, 150, 255], 
    60: [170, 255, 150], 
    70: [0, 175, 0], 
    71: [0, 60, 135], 
    72: [80, 240, 150], 
    80: [150, 240, 255], 
    81: [0, 0, 255], 
    99: [255, 255, 50], 
    252: [245, 150, 100], 
    256: [255, 0, 0], 
    253: [200, 40, 255], 
    254: [30, 30, 255], 
    255: [90, 30, 150], 
    257: [250, 80, 100], 
    258: [180, 30, 80], 
    259: [255, 0, 0]
}


def vis_pcd_label_o3d(pcd, label):
    """ 
    TODO: 根据label来绘制点云
    """
    N = len(label)
    color = np.zeros(shape=(N, 3), dtype=np.float32)
    for i in range(N):
        color[i, :] = color_dict[label[i]]
    color /= 255
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])

class PCDVisualer:
    def __init__(self) -> None:
        # cmap: label->color BGR
        self.cmap = {
            0 : [0, 0, 0], 
            1 : [0, 0, 255], 
            10: [245, 150, 100], 
            11: [245, 230, 100], 
            13: [250, 80, 100], 
            15: [150, 60, 30], 
            16: [255, 0, 0], 
            18: [180, 30, 80], 
            20: [255, 0, 0], 
            30: [30, 30, 255], 
            31: [200, 40, 255], 
            32: [90, 30, 150], 
            40: [255, 0, 255], 
            44: [255, 150, 255], 
            48: [75, 0, 75], 
            49: [75, 0, 175], 
            50: [0, 200, 255], 
            51: [50, 120, 255], 
            52: [0, 150, 255], 
            60: [170, 255, 150], 
            70: [0, 175, 0], 
            71: [0, 60, 135], 
            72: [80, 240, 150], 
            80: [150, 240, 255], 
            81: [0, 0, 255], 
            99: [255, 255, 50], 
            252: [245, 150, 100], 
            256: [255, 0, 0], 
            253: [200, 40, 255], 
            254: [30, 30, 255], 
            255: [90, 30, 150], 
            257: [250, 80, 100], 
            258: [180, 30, 80], 
            259: [255, 0, 0]
        }
        self.name_label_mapping = {
            'unlabeled': 0, 
            'outlier': 1,
            'car': 10,
            'bicycle': 11,
            'bus': 13,
            'motorcycle': 15,
            'on-rails': 16,
            'truck': 18,
            'other-vehicle': 20,
            'person': 30,
            'bicyclist': 31,
            'motorcyclist': 32,
            'road': 40,
            'parking': 44,
            'sidewalk': 48,
            'other-ground': 49,
            'building': 50,
            'fence': 51,
            'other-structure': 52,
            'lane-marking': 60,
            'vegetation': 70,
            'trunk': 71,
            'terrain': 72,
            'pole': 80,
            'traffic-sign': 81,
            'other-object': 99,
            'moving-car': 252,
            'moving-bicyclist': 253,
            'moving-person': 254,
            'moving-motorcyclist': 255,
            'moving-on-rails': 256,
            'moving-bus': 257,
            'moving-truck': 258,
            'moving-other-vehicle': 259
        }

    def vis_pcd(self, pcd_list):
        print(pcd_list)
        pcd_list = any_to_PointCloud(pcd_list)
        o3d.visualization.draw_geometries(pcd_list)

    def set_pcd_with_semantic_label(self, pcd, label, is_vis=False):
        """ 给pcd上色 via label (输入可以是list)
        args:
            pcd: ndarray|PCD (N, 3) or list of it
            label: ndarray|PCD (N, ) or list of it
        """
        if not isinstance(pcd, list):
            pcd = [pcd]
            label = [label]
        pcd = any_to_PointCloud(pcd)  
        # set color with label
        for idx_pcd in range(len(pcd)):
            # data
            _label = label[idx_pcd]
            _pcd = o4d.geometry.ColaPointCloud(pcd[idx_pcd])
            N = _label.shape[0]
            assert N == _pcd.cola_get_size()

            # color
            color = np.zeros(shape=(N, 3), dtype=np.float32)
            for i in range(N):
                color[i, :] = self.cmap[_label[i]]
            _pcd.cola_set_colors(color)
        
        if is_vis:
            self.vis_pcd(pcd)
        if len(pcd) == 1:
            return pcd[0]
        else:
            return pcd
    

def vis_pcd_o3d(pcd, labels=None, colors=None, config=None):
    """
    pcd: Nx3 Tensor|ndarray|PointCloud or list of both of all
    color: color cmap, cmap(float)->tupe(4), or list of color cmap, ex: [(0.1, 0.2, 0.3), (0.1, 0.2, 0.4)] for two pcd color
    """
    # 1. convert pcd to list of PointCloud
    # to pcd list
    if not isinstance(pcd, list):
        pcd = [pcd]
    # to PointCloud
    pcd = [any_to_PointCloud(item) for item in pcd]

    # 2. paint color
    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels]
        colors = []
        for label in labels:  # for each pcd
            cmap = cm.get_cmap("Spectral")
            color = np.zeros(shape=(len(label), 3), dtype=np.float32)
            label_unique = np.unique(label[0])
            for l in label_unique:
                idx_bool = l == label
                color[idx_bool, :] = cmap(l % 20 / 20.0)[:3] # 第idx_point个标签->颜色
            colors.append(color)
    elif colors is None:
        cmap = cm.get_cmap("Spectral")
        colors = [cmap(item)[:3] for item in np.random.rand(len(pcd))]
    assert len(colors) == len(pcd)
    for i in range(len(pcd)):
        if colors[i].size == 3:
            pcd[i].paint_uniform_color(colors[i])
        else:
            pcd[i].colors = o4d.utility.Vector3dVector(colors[i])

    # visualization
    if config is None:
        o3d.visualization.draw_geometries(pcd)
    else:
        o3d.visualization.draw_geometries(pcd, **config)


if __name__ == "__main__":
    def test_plt_vis_pcd():
        cmap = cm.get_cmap("Spectral")  # input: float in (0,1) -> (RGBA)
        points = [np.random.randn(1000, 3) for i in range(2)]
        vis_pcd_plt(points, config={"size": 10},
                    color=[(0.1, 0.2, 0.8), (1, 0, 0)])
        plt.show()
    # test_plt_vis_pcd()

    def test_o3d_vis():
        points = np.random.randn(1000, 3)
        vis_pcd_o3d(points, config={
                    "window_name": "Cola", "width": 800, "height": 600})
    test_o3d_vis()


"""
python -m ColaUtils.vis_utils
"""