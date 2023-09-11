# ColaDatasets
- [readme](./ColaDatasets/readme.md)

# ColaPCRModules
- - [readme](./ColaPCRModules/readme.md)

# ColaUtils

## [time_utils](./ColaUtils/time_utils.py)
- `TimeRecorder` 记录一段时间并打印出来


## [pcr_utils](ColaUtils/pcr_utils.py)
- 配准流程
    - `integrate_trans` (R, t) -> trans
    - `transform` rslt = R @ pts + shift
    - `rigid_transform_3d` 根据匹配对求解刚体变换
    - `post_refinement` 给定初始变换后，在改变换的基础上，计算所有内点 -> 若内点数更多则通过这些内点重新计算一个新的变换 -> 不断迭代
    - `estimate_normal` estimate normal by neighbor
    - `transformation_error` 计算变换误差
    - `create_corr_via_descriptor_sample` 直接计算各个描述符之间的距离然后返回匹配对下标
- 可视化
    - `visualization_2phase` vis origin and after trans
- 降采样
    - `downsample_random` 按照一定比例|数量均匀随机降采样
    - `downsample_voxel` 体素降采样
    - `downsample_fps` 最远点采样
    - `downsample_fps_batch` 批量FPS降采
- 配准结果
  - `calculate_right_correspondence_mask` 给定匹配关系和正确变换 -> 正确匹配对的mas
  - `calculate_right_correspondences_rate` 计算正确匹配率
  - 

## [log_utils](ColaUtils/log_utils.py)
- `Logger` 对于某个日志(name指定文件名), 将其存储到 log_dir 下

## [data_saveload_utils](ColaUtils/data_saveload_utils.py)
- `DatasetSaveLoader` 在指定文件夹下存储某一类文件, 每一类文件下有多个(通过idx来区分),支持加载某个文件或指定idx的全部类别文OR某种类别的全部数据

## [sequence_utils](./ColaUtils/sequence_utils.py)
> - 在序列上处理的函数
- `find_first_ge` 有序序列第一个大于等于某个数的下标


## [wrapper](./ColaUtils/wrapper.py)
> - 一些 wrapper
- `wrapper_ignore_error` 函数报错会被忽略
- `wrapper_data_trans` 原本对f(a)->b: 转为 [a1, a2] -> [b1, b2] 同时也支持tuple

## [code_utils](./ColaUtils/code_utils.py)
> - 用于方便写代码的函数
- `para2dict` 将函数参数以命名函数的形式给出,然后返回dict


## [torchnp_utils](./ColaUtils/torchnp_utils.py)
- 数据转换
  - `ndarray_to_tensor`
  - `tensor_to_ndarray`
- 维度变换
  - `expand_axis`
  - `transpose`

## [statistic_utils](./ColaUtils/statistic_utils.py)
> - 统计相关工具
- `calculate_stats` 一组数据的统计量

## [o3d_utils](./ColaUtils/o3d_utils.py)
- torch ndarray <-> open3d
    - `ndarray_to_pcd` ndarray(N, 3) 转为 pcd 支持多个
    - `ndarray_to_Feature` feature: ndarray (dim, N) N个长度为dim的列特征向量 or list of it
    - `any_to_PointCloud` 任意Tensor|Ndarray 转为 PointClou
    - `pcd_to_ndarray` PCD -> ndarray
    - `feature_to_ndarray` feature -> ndarray

## [pcdvis_utils](./ColaUtils/pcdvis_utils.py)
- `VisDynamic` 动态图绘制
- `vis_geo_static` 可视化静态几何体
- `set_geometry_color` 给几何体设置颜色, 指定一个点 or 全部点
- `set_pcd_with_semantic_label` 给pcd上色 via label (输入可以是list)
- `get_corres_lines` 根据匹配关键点获得线条对象

# ColaPCRModules

## Datasets

### [kitti_dataset](./ColaPCRModules/Datasets/kitti_dataset.py)
- `read_kitti_velody` 读取kitti的点云数据 with bin file path