class ColoringStr:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # color
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"

    @staticmethod
    def coloring(s:str, *c_list): 
        rslt = s
        for c in c_list:
            rslt = c + rslt + ColoringStr.ENDC
        return rslt

if __name__=="__main__":
    print(ColoringStr.coloring("abcd", ColoringStr.YELLOW, ColoringStr.BOLD))
"""

字符序列	颜色/样式
\033[0m	重置所有样式
\033[1m	加粗
\033[2m	暗色
\033[3m	斜体
\033[4m	下划线
\033[5m	闪烁
\033[7m	反色
\033[8m	隐藏
\033[30m	黑色
\033[31m	红色
\033[32m	绿色
\033[33m	黄色
\033[34m	蓝色
\033[35m	紫色
\033[36m	青色
\033[37m	白色
\033[40m	黑色背景
\033[41m	红色背景
\033[42m	绿色背景
\033[43m	黄色背景
\033[44m	蓝色背景
\033[45m	紫色背景
\033[46m	青色背景
\033[47m	白色背景
python -m ColaUtils2.coloring_str_utils
"""