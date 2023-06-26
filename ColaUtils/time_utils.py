import time

class TimeRecorder:
    def __init__(self, timer=None) -> None:
        self.timer = timer if timer is not None else time.perf_counter
        self.t1 = 0
    
    def start(self):
        self.t1 = self.timer()
    
    def end(self, info="Delta (s): "):
        print(info, self.timer() - self.t1)


class TimerStampRecord:
    def __init__(self) -> None:
        self.time_stamp = []
    
    def stamp(self):
        """记录一个时间戳
        """
        self.time_stamp.append(time.time())
    
    def show(self, info="Total time (sec): "):
        print(info, self.get())

    def get(self):
        """整个时间戳结束并打印结果
        """
        assert len(self.time_stamp) % 2 == 0
        N = len(self.time_stamp)
        t = 0
        for i in range(0, N, 2):
            t += self.time_stamp[i + 1] - self.time_stamp[i]
        return t

    def clear(self):
        self.time_stamp = []

"""
Test function
"""
def test_TimeRecorder():
    timer = TimeRecorder()
    timer.start()
    time.sleep(0.5)
    timer.end()


def test_TimerStampRecord():
    timer = TimerStampRecord()
    # 第一段
    timer.stamp()
    time.sleep(1)
    timer.stamp()

    # 其他事情 不被包含
    time.sleep(0.5)

    # 第二段
    timer.stamp()
    time.sleep(1)
    timer.stamp()

    timer.show()


if __name__=="__main__":
    test_TimeRecorder()
    test_TimerStampRecord()


"""
python -m ColaUtils.time_utils
"""