注意
- 某些class使用的默认构造函数, 从而无法继承
    - 例如 PointCloud 的 [Bind函数](https://github.com/isl-org/Open3D/blob/a167efb1852abd291b7fa8f03593d448d29737e4/src/Python/open3d.h#L56), 里面直接导出PointCloud的默认构造
    - [Pybind 对需要多种构造函数的解释定制方法](https://pybind11.readthedocs.io/en/stable/advanced/classes.html#custom-constructors)
    - 解决方案: 不使用继承,而采用成员变量的方式
    - 缺点: 若没有定义 PointCloud.xx 将无法使用 xxx 函数
        - 解决方案: 直接使用成员变量来操作, 例如 pcd.pcd.xxx
    - 好处: 若几何体来自其他地方,则可以直接对内部成员赋值 例如 pcd.data = pcd2
    - 标识方法
        - 这种类前都加上 Cola 作为前缀, 例如 ColaPointCloud
        - 内部数据采用 data 统一标识
        - 对于可以继承的类 则使用原始类名, 例如 KDTreeFlann
    - 建议: 这种方法相当于使用了一个外部管理器来管理 open3d 的对象, 在使用前可以将其取出来
``` python
lines_cola = o4d.geometry.ColaLineSet()
lines = lines_cola.data  # 取出来 引用
lines_cola.cola_init(pts, line_indices)
```


# PointCloud
- [point_cloud_bounding_box](./study/point_cloud_bounding_box.py) 获取点云的包围盒(沿着坐标轴|点云方向) -> 以不同颜色可视化出来 
    - 包围盒也是一个几何体

