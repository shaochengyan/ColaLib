import open3d as o3d


class KDTreeFlann(o3d.geometry.KDTreeFlann):
    def __init__(self, *args, **kwargs) -> None:
        """
        NOTE: 
            - BUG: 建议不要使用直接传入数据,貌似可能会出现bug -> 先创建KDTree, 然后调用 set_geometry
        input: 
            NULL: do nothing
            data: numpy.ndarray[numpy.float64[m, n]]) -> None
            geometry: open3d.cpu.pybind.geometry.Geometry) -> None
            feature: open3d::pipelines::registration::Feature) -> None
        """
        return super().__init__(*args, **kwargs)

    
    def set_feature(self, feature):
        """
        Sets the data for the KDTree from the feature data.
        Parameters
            feature (open3d.pipelines.registration.Feature) Feature data.
        Returns
            bool
        """
        return super().set_feature(feature)


    def set_geometry(self, geometry):
        """
        Sets the data for the KDTree from geometry.
        Parameters
            geometry (open3d.geometry.Geometry)
        Returns
            bool
        """
        return super().set_geometry(geometry)


    def set_matrix_data(self, data):
        """Sets the data for the KDTree from a matrix.
        Parameters
            data (numpy.ndarray[numpy.float64[m, n]]) Matrix data.
        Returns
            bool
        """
        return super().set_matrix_data(data)


    def search_radius_vector_xd(self, query, radius):
        """
        query (numpy.ndarray[numpy.float64[m, 1]]) The input query point.
        radius (float) Search radius.
        Returns
            Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        """
        return super().search_radius_vector_xd(query, radius)


    def search_radius_vector_3d(self, query, radius):
        """ KNN
        query (numpy.ndarray[numpy.float64[3, 1]]) The input query point.
        radius (float) Search radius.
        return: Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        """
        return super().search_radius_vector_3d(query, radius)


    def search_knn_vector_xd(self, query, max_nn):
        """ KNN
        query (numpy.ndarray[numpy.float64[m, 1]]) The input query point.
        max_nn (int) At maximum, max_nn neighbors will be searched.
        return: Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        """
        return super().search_knn_vector_xd(query, max_nn)


    def search_knn_vector_3d(self, query, knn):
        """ KNN
        query (numpy.ndarray[numpy.float64[3, 1]]) The input query point.
        knn (int) knn neighbors will be searched.
        return: Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        """
        return super().search_knn_vector_3d(query, knn)


    def search_hybrid_vector_xd(self, query, radius, max_nn):
        """
        query (numpy.ndarray[numpy.float64[m, 1]]) The input query point.
        radius (float) Search radius.
        max_nn (int) At maximum, max_nn neighbors will be searched.
        return: Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        """
        return super().search_hybrid_vector_xd(query, radius, max_nn)

    
    def search_hybrid_vector_3d(self, query, radius, max_nn):
        """ knn and radius
        query (numpy.ndarray[numpy.float64[3, 1]]) The input query point.
        radius (float) Search radius.
        max_nn (int) At maximum, max_nn neighbors will be searched.
        return: Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        """
        return super().search_hybrid_vector_3d(query, radius, max_nn)


if __name__=="__main__":
    import ColaOpen3D as o4d
    import open3d as o3d
    import numpy as np
    """
    return
        k: int, 多少个点
        idx: IntVector
        dis: DoubleVector
    """

    # For pcd data
    points = np.random.rand(1000, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    tree = o4d.geometry.KDTreeFlann(pcd)
    [k, idx, dis] = tree.search_knn_vector_3d(pcd.points[0], 10)
    print("Total %d points." % k)
    print(points[idx, :])
    print(dis)


    # For [M, N] data, M 是特征的维度|一共有N个特征向量|所以特征向量是列向量
    features = np.random.rand(32, 100)
    tree.set_matrix_data(features)
    [k, idx, dis] = tree.search_knn_vector_xd(features[:, 0], 4)  # 注意使用的是 xd
    print("Total %d points." % k)
    print(features[:, idx])
    print(dis)

    # TODO: For o3d feature

    # For radius
    [k, idx, dis] = tree.search_radius_vector_xd(features[:, 0], 4.1)  # 注意使用的是 xd

    # For knn and radius: 半径10范围内最近的10个点
    [k, idx, dis] = tree.search_hybrid_vector_xd(features[:, 0], 10, 10)







