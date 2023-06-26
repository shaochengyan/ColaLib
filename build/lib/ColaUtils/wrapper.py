def wrapper_ignore_error(func):
    """函数报错会被忽略"""
    def wrapper_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as ex:
            print(ex)
    return wrapper_func   

def wrapper_data_trans(trans):
    """
    原本对f(a)->b: 转为 [a1, a2] -> [b1, b2] 同时也支持tuple
    """
    def trans_all(obj, **kwargs):
        if isinstance(obj, (tuple, list)):
            return [ trans(item, **kwargs) for item in obj ]
        else:
            return trans(obj, **kwargs)
    return trans_all

@wrapper_ignore_error
def test_wrapper_ignore_error():
    return 1 / 0

if __name__=="__main__":
    test_wrapper_ignore_error()