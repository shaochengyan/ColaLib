# KITTI数据集
## [kitti_loader_gp](./kitti_loader_gp.py)
> - COLA在做毕设的时候写的KITTI数据加载器
> - **NOTE-存在问题-batch只能为1** 想要多个batch，则需要每一次返回的数据尺度相同
- `read_kitti_velody` 读取kitti雷达数据
- `ColaOdometry` 在原本`pykitti.odometry` 的基础上添加了相机姿态等信息
- `ColaKITTIUtils` 指定加载KITTI的某些seq
- `ColaKITTIPCR` 用于PCR的数据加载, 继承自 `ColaKITTIUtils`
