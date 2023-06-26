"""
sequence上的操作
"""


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