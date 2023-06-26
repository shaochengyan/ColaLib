# Detector
> - 关键点检测器

## [pcl_kpt_detector](./Detector/pcl_kpt_detector.py)
- `random_select` 随机关键点选取 
- TOOD


# Descriptor

## [classical_descriptor](./Descriptor/classical_descriptor.py)
> 经典描述符 by o3d

- fpfh

## [pcl_descriptor](./Descriptor/pcl_descriptor.py)
> 经典描述符 by pcl 

- `run_fpfh33` 
- TODO

# AdvancePCR
## [maxcliques](./AdvancePCR/maxcliques.py)
> - CVPR 2023 最大团
- `max_clique` 输入匹配对, 输出配准结果

## [SeToReg](./AdvancePCR/SeToReg.py)
> - 自己的RAL论文
- TODO: 创建匹配对的函数

- `max_clique` 输入匹配对, 输出配准结果

## [registrator_o3d](./AdvancePCR/registrator_o3d.py)
> - open3d中的配准ransac配准方案

- `O3dRegistratorRansac` 输入关键点及其描述符->创建匹配对->ransac (根据MM自动创建匹配对)
- `O3dRegistratorBasedCorreRansac` 输入关键点及匹配关系