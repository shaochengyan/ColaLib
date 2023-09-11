import torch
import numpy as np
from .wrapper import wrapper_data_trans

@wrapper_data_trans
def tensor_to_type(tensor, dtype):
    return tensor.to(dtype)

@wrapper_data_trans
def ndarray_to_type(arr, dtype):
    return arr.astype(dtype)



@wrapper_data_trans
def ndarray_to_tensor(arr, dtype=None):
    """
    NOTE: must specify dtype=bbb if you pass the dtype paramenter
    arr: ndarray or tuple/list of it or tensor
    dtype: default not change dtype, else torch.float|torch.double|torch.int
    ex:
        b = ndarray_to_tensor2([a, a, a], dtype=torch.float)
    """
    tensor = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    return tensor.to(dtype)


@wrapper_data_trans
def tensor_to_ndarray(tensor, dtype=None):
    """
    NOTE: must specify dtype=bbb if you pass the dtype paramenter
        - not: tensor_to_ndarray(tensor, np.float32)
        - yes: tensor_to_ndarray(tensor, dtype=np.float32)
    tensor: torch.Tensor or tuple|list of it or ndarray
    dtype: np.float32, np.float64
    """
    arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
    return arr.astype(dtype)


@wrapper_data_trans
def expand_axis(data, axis=0):
    """ (批量)扩展ndarray的指定轴
    data: ndarray|tensor or list of it
    """
    return np.expand_dims(data, axis) if isinstance(data, np.ndarray) \
        else torch.unsqueeze(data, axis) 


@wrapper_data_trans
def transpose(data):
    """ 对传入的data进行批量转置
    NOTE: numpy|torch 均使用 .T 进行转置
    """
    assert len(data.shape) == 2
    return data.T

if __name__=="__main__":
    a = torch.rand(100, 20)
    print(tensor_to_type([a, a], torch.float16)) 

    b = np.random.rand(10, 20)
    print(ndarray_to_type([b, b], np.float128))

"""
python -m ColaUtils.torchnp_utils
"""
