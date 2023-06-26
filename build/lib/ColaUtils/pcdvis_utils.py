"""
author: Cola
brief: visualize pointcloud via Open3D with different data type -> tensor|ndarray|open3dGeometry
NOTE: 
    - tensor|ndarray|open3dGeometry -> vis_datatype
    - tensor|ndarray will be same as PointCloud 
    - based all on o4d|o3d
"""
import numpy as np
import open3d as o3d

import ColaOpen3D as o4d
from ColaUtils import o3d_utils 

class VisDynamic:
    """
    动态图绘制
    """
    def __init__(self, geo_list=[], window_name="Open3D", width=800, height=600, left=50, top=50) -> None:
        """ 
        NOTE: 输入必须是几何体(不要输入点)
        geo_list: list of geometry(PCD|LINE|blabla)
        """
        self.geo_list = geo_list
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width, height, left, top)
        for geo in geo_list:
            self.vis.add_geometry(geo)
    
    def update_renderer(self):
        """ 重新渲染每一个几何体
        """
        for pcd in self.geo_list:
            self.vis.update_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def remove_geometry(self, geo_list):
        # TODO: 更新几何体 list
        for item in geo_list:
            self.vis.remove_geometry(item)  

            # 仅保留存下的
            self.geo_list = [ item_ for item_ in self.geo_list if item_ is not item ]
        self.update_renderer()

    def clear_geometries(self):
        self.vis.clear_geometries()
        self.update_renderer()
    
    def add_geometry(self, geo_list):
        for item in geo_list:
            self.vis.add_geometry(item)
            self.geo_list.append(item)
        self.update_renderer()

    def run(self):
        self.vis.run()

    def close(self):
        self.vis.close()

    def destroy_window(self):
        self.vis.destroy_window()
    

def vis_geo_static(pcd_list, window_name="Open3D", width=800, height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False):
    """ visualize geometry
    可视化静态几何体
    pcd_list: list of geometry
    """
    o4d.visualization.draw_geometries(pcd_list, window_name="Open3D", width=width, height=height, left=left, top=top, point_show_normal=point_show_normal, mesh_show_wireframe=mesh_show_wireframe, mesh_show_back_face=mesh_show_back_face)



## set function: 给几何体上色|语义颜色等等
def set_geometry_color(geo, color_arr):
    """
    给几何体设置颜色, 指定一个点 or 全部点
    NOTE: every gemoetry have colors
    color_arr: ndarray
        - (N, 3) for PCD N points
        - (3, ) for uniform color 
    """
    if not isinstance(color_arr, np.ndarray):
        color_arr = np.asarray(color_arr)
    if color_arr.size <= 3:
        geo.paint_uniform_color(color_arr)
    else:
        geo.colors = o4d.utility.Vector3dVector(color_arr)
    return True


kitti_semantic_cmap = {
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
kitti_semantic_name_map = {
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
def set_pcd_with_semantic_label(pcd, label, is_vis=False, cmap=None):
    """ 
    给pcd上色 via label (输入可以是list)
    args:
        pcd: PCD (N, 3) or list of it
        label: ndarray (N, ) or list of it
    return: 
        - if there are multiple pcd -> list of pcd
        - one pcd -> pcd
    """
    if not isinstance(pcd, list):
        pcd = [pcd]
        label = [label]
    if cmap is None:
        cmap = kitti_semantic_cmap
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
            idx_label = _label[i]
            if idx_label not in cmap.keys():
                color[i, :] = 0
            else:
                color[i, :] = cmap[_label[i]]
        # _pcd.cola_set_colors(color) 同下
        set_geometry_color(_pcd.data, color)
    
    if is_vis:
        vis_geo_static(pcd)
    if len(pcd) == 1:
        return pcd[0]
    else:
        return pcd

"""
根据关键点及其匹配关系绘制匹配线, 返回odometry.ColaLines
"""
def get_corres_lines(kpts_src, kpts_dst, corres, is_show=False):
    """
    根据匹配关键点获得线条对象
    kpts: (N|M, 3), ndarray
    corres: (N, 2) int ndarray
    return: ColaLineSets
    """
    num_src = len(kpts_src)
    pts = np.vstack([kpts_src, kpts_dst])
    lines_idxs = corres
    lines_idxs[:, 1] += num_src
    lines = o4d.geometry.ColaLineSet()
    lines.cola_init_lines(pts, lines_idxs)

    if is_show:
        vis_geo_static([lines.data])
    
    return lines



if __name__=="__main__":
    import ColaOpen3D as o4d
    from ColaUtils import o3d_utils 
    import numpy as np
    # 测试绘制线条
    def test2():
        line = o4d.geometry.ColaLineSet()
        pts = np.asarray([
            0, 0, 0, 
            2., 2, 2, 
            -1.0, -2.0, -3.0
        ]).reshape(-1, 3)
        line_indices = np.asarray([
            0, 1, 
            1, 2
        ]).astype(np.int32).reshape(-1, 2)
        line.cola_init_lines(pts, line_indices)
        set_geometry_color(line.data, np.asarray([0.1, 0.2, 0.9]))
        vis_geo_static([line.data])

    # test2()
    # 测试点云的语义标签可视化
    def test1():
        from ColaPCRModules.Assets.ExampleData import load_data
        pcd  = load_data()[1]
        # o4d.geometry.ColaPointCloud
        label = np.zeros(len(pcd.points))
        for idx in range(10):
            pcd = set_pcd_with_semantic_label(pcd, label=label)
            vis_geo_static([pcd])
            break
    test1()

"""
python -m ColaUtils.pcdvis_utils
"""