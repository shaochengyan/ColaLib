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
    

def vis_geo_static(*pcd_list, window_name="Open3D", width=800, height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False):
    """ visualize geometry
    可视化静态几何体
    pcd_list: list of geometry
    """
    o4d.visualization.draw_geometries(pcd_list, window_name=window_name, width=width, height=height, left=left, top=top, point_show_normal=point_show_normal, mesh_show_wireframe=mesh_show_wireframe, mesh_show_back_face=mesh_show_back_face)



## set function: 给几何体上色|语义颜色等等
def set_geometry_color(geo, color_arr):
    """
    给几何体设置颜色, 指定一个点 or 全部点
    NOTE: every gemoetry have colors
    color_arr: ndarray
        - (N, 3) for PCD N points
        - (3, ) for uniform color 
    example:
        - dst_pcd.cola_set_colors(np.asarray([67,200,117]) / 255.0)
        - src_pcd.cola_set_colors(np.asarray([137,117,221]) / 255.0)
    """
    if not isinstance(color_arr, np.ndarray):
        color_arr = np.asarray(color_arr)
    if color_arr.size <= 3:
        geo.paint_uniform_color(color_arr)
    else:
        geo.colors = o4d.utility.Vector3dVector(color_arr)
    return True

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
    from .others.kitti_semantic_cmap import kitti_semantic_cmap, kitti_semantic_name_map
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
        assert N == len(_pcd)

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




def get_lines(pts1, pts2, color=None):
    """
    pts1, pts2: Nx3 with same N
    the i'th line: pts1[i] -- pts2[i]
    """
    pts1 = pts1.reshape(-1, 3)
    pts2 = pts2.reshape(-1, 3)

    lines = o4d.geometry.ColaLineSet()
    pts = np.vstack([pts1, pts2])
    indic = np.column_stack([np.arange(len(pts1)), np.arange(len(pts1)) + len(pts1)])
    lines.cola_init_lines(points=pts, line_indices=indic)

    if color is None:
        colors_line = np.tile(np.asarray([0, 1.0, 0]), reps=(len(pts1), 1))
    else:
        color = np.tile(np.asarray(color), reps=(len(pts), 1))
    lines.cola_init_colors(colors_line)


    return lines


def get_sphere(center, radius, color):
    # 创建一个球体几何体
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    # 平移球体使其位于指定中心位置
    mesh.translate(center)
    # 设置球体颜色
    mesh.paint_uniform_color(color)
    return mesh


def vis_corres(
    pts_src, pts_dst, kpts_src, kpts_dst
):
    """
    可视化源点和目标点即匹配关系
    TODO: MASK 以区分 in/outlier  
        - label_inout is a True/False array with length == len(kpts_src) == len(kpts_dst)
            - red: inlier
            - green: outlier
    """
    # pts_dst 翻转: z反向后加上一定值
    z_delta = 30
    pts_dst_new = pts_dst.copy()
    # pts_dst_new[..., 2] = -pts_dst_new[..., 2]
    pts_dst_new[..., 2] += z_delta

    kpts_dst_new = kpts_dst.copy()
    # kpts_dst_new[..., 2] = -kpts_dst_new[..., 2]
    kpts_dst_new[..., 2] += z_delta

    geo_list = [] 
    for i in range(len(kpts_src)):
        sp_src = get_sphere(center=kpts_src[i], radius=0.5, color=[1.0, 0, 0])
        sp_dst = get_sphere(center=kpts_dst_new[i], radius=0.5, color=[1.0, 0, 0])
        geo_list.append(sp_src)
        geo_list.append(sp_dst)
    
    lines = o4d.geometry.ColaLineSet()
    pts_line = np.vstack([kpts_src, kpts_dst_new])
    line_indic = np.column_stack([np.arange(len(kpts_src)), np.arange(len(kpts_src)) + len(kpts_src) ])
    lines.cola_init_lines(pts_line, line_indic)
    colors_line = np.tile(np.asarray([0, 1.0, 0]), reps=(len(line_indic), 1))
    lines.cola_init_colors(colors_line)

    pcd_src = o4d.geometry.ColaPointCloud(pts_src)
    pcd_src.paint_uniform_color(np.asarray([1, 0.706, 0])[:, None])
    pcd_src.estimate_normals(radius=0.3)

    pcd_dst = o4d.geometry.ColaPointCloud(pts_dst_new)
    pcd_dst.paint_uniform_color(np.asarray([0, 0.651, 0.929])[:, None])
    pcd_dst.estimate_normals(radius=0.3)

    geo_list.extend([pcd_src.data, pcd_dst.data, lines.data])
    vis_geo_static(*geo_list)



def vis_two_pts(pts_src, label_src, pts_dst, label_dst):
    pcd_src = get_pcd_coloring_with_label(pts_src, label_src)
    pcd_dst = get_pcd_coloring_with_label(pts_dst, label_dst)
    vis_geo_static(pcd_src, pcd_dst)
    

def vis_with_label(pts, label):
    """
    vis with semantic label
    """
    pcd = get_pcd_coloring_with_label(pts, label)
    vis_geo_static(pcd)


def get_pcd_coloring_with_label(pts, label):
    pcd = o4d.geometry.ColaPointCloud(pts)
    pcd.cola_set_color_with_label(label)
    return pcd.data




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