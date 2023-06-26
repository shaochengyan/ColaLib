import numpy as np

def calculate_stats(data):
    """
    计算一列数据的分布统计数据。

    参数：
    data: 一维数组，需要计算统计数据的数据集。

    返回值：
    一个字典，包括平均值、中位数、标准差、方差、最小值、最大值和一些常见百分位数的键值对。
    """
    stats = {}
    stats['mean'] = np.mean(data)
    stats['median'] = np.median(data)
    stats['std'] = np.std(data)
    stats['var'] = np.var(data)
    stats['min'] = np.min(data)
    stats['max'] = np.max(data)
    stats['25th_percentile'] = np.percentile(data, 25)
    stats['75th_percentile'] = np.percentile(data, 75)
    stats['90th_percentile'] = np.percentile(data, 90)
    stats['95th_percentile'] = np.percentile(data, 95)
    stats['99th_percentile'] = np.percentile(data, 99)
    return stats