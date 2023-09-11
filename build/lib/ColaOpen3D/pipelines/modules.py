import open3d as o3d
import numpy as np


"""
http://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.Feature.html
"""

class Feature(o3d.pipelines.registration.Feature):
    def __init__(self, colum_data=None):
        """
        NOTE: colum_data (dim, N)
        """
        super().__init__()
        if colum_data is not None:
            self.data = colum_data
    
    def dimension(self):
        return self.dimension()

    def num(self):
        return self.num()

    def resize(self, dim, n):
        """
        resize to dim x n
        """
        return self.resize(dim, n)


if __name__=="__main__":
    data = np.random.rand(33, 1000)
    fet = Feature()
    fet = Feature(data)
    print(fet)


"""
python -m ColaOpen3D.pipelines.modules
"""




