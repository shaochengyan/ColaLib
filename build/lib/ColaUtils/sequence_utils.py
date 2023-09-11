"""
sequence上的操作
"""
import numpy as np


def find_first_ge(arr, value):
    """
    arr中第一个大于value的值说明该seq能包含多余value个
    asumme that arr is sorted ascending 
    return: the index of first value in arr greater equal than value
    """
    for i in range(len(arr)):
        if value <= arr[i]:
            return i
    return len(arr)

def normalize(data: np.ndarray):
    """
    data: (N, dim) or (dim)
    return: same with dim, norm(reslt) == 1
    """
    return data / (np.linalg.norm(data, axis=-1, keepdims=True) + 1e-9)