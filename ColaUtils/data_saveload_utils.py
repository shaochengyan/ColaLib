import os
import numpy as np
from dotmap import DotMap 
import re
import glob

"""
目的-存储数据集文件: 将有序数据(idx作为序号)存储到指定文件夹爱下，通过<idx>_<name>.npy来指定
save idx data, idx represent the data idx in torch_dataset
"""
class DatasetSaveLoader:
    """
    在指定文件夹下存储某一类文件, 每一类文件下有多个(通过idx来区分),支持加载某个文件或指定idx的全部类别文件
    example: 下面有两类(A, B)，每一类下有三个数据
        000001_A.npy
        000001_B.npy
        000002_A.npy
        000002_B.npy
        000003_A.npy
        000003_B.npy
    """
    def __init__(self, dir_save) -> None:
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        self.dir_save = dir_save
        self.filebase = os.path.join(dir_save, "{:06d}_{}")
    
    def save(self, idx, name, data):
        """
        name: save name -> name.npy
        data: ndarray or list of it
        """
        # data convert
        if isinstance(data, (list, tuple)):
            data = np.asarray(data, dtype=object)
        
        # save
        filename = self.filebase.format(idx, name)
        np.save(filename, data, allow_pickle=True)

    def find_data_number(self):
        """
        1. load all filename -> check idx 
        """
        filenames = os.listdir(self.dir_save)
        num = 0
        for fn in filenames:
            idx = int(fn[:6]) + 1
            if num < idx:
                num = idx

        return num

    def load(self, idx, name):
        """
        name: file name of .npy file, e.g usip_kpts.npy
        """

        file = self.filebase.format(idx, name) + ".npy"
        data = np.load(file, allow_pickle=True)
        if data.dtype == np.dtype("O"):
            data = data.tolist()
        return data
    
    def is_exist(self, idx, name):
        file = self.filebase.format(idx, name) + ".npy"
        return os.path.isfile(file)

    def loadall_idx(self, idx):
        """ load all data of idx, return a dotmap
        """
        filepattern = self.filebase.format(idx, "*")
        repattern = r"/\d{6}_(.+)\."
        return self.loadall(filepattern, repattern)
    
    def loadall_cls(self, cls):
        """根据数据类型名来提取全部"""
        filepattern = os.path.join(self.dir_save, "{}_{}.npy".format("*", cls)) 
        repattern = r"/(\d+)_\w+"
        return self.loadall(filepattern, repattern)
    
    def save_with_name(self, name, data):
        if isinstance(data, (list, tuple)):
            data = np.asarray(data, dtype=object)
        
        # save
        filename = os.path.join(self.dir_save, name)
        np.save(filename, data, allow_pickle=True) 
    
    def load_with_name(self, name):
        filename = os.path.join(self.dir_save, name+".npy")
        data = np.load(filename, allow_pickle=True) 
        # if data.dtype is np.ndarray: 
        if data.dtype == np.dtype("O"):
            data = data.tolist()
        return data
    
   
    def loadall(self, filepattern, repattern):
        file_list = glob.glob(filepattern) 

        dataall = dict()
        for file in file_list:
            name = re.findall(repattern, file)[0]
            data = np.load(file, allow_pickle=True) 
            
            # if data.dtype is np.ndarray: 
            if data.dtype == np.dtype("O"):
                data = data.tolist()

            dataall[name] = data
        
        return DotMap(dataall)
 


def test_DatasetSaveLoader():
    dataloader = DatasetSaveLoader(dir_save="./TMP")
    # 存储
    for i in range(10):
        data = [np.random.rand(2, 3)]
        data.append(1)
        data.append([1, 2, 3])
        dataloader.save(i, "A", data)
        dataloader.save(i, "B", data)
    
    # 加载
    rslt = dataloader.load(1, "A") 
    print(rslt)

    rslt = dataloader.loadall_idx(1) 
    print(rslt.A, rslt.B)

    rslt = dataloader.loadall_cls("A")
    pass


if __name__=="__main__":
    test_DatasetSaveLoader()


"""
python -m ColaUtils.data_saveload_utils
"""