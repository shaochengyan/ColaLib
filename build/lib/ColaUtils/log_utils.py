import logging
import os


class Logger(logging.Logger):
    def __init__(self, name: str, log_dir="./TMP/log") -> None:
        """
        对于某个日志(name指定文件名), 将其存储到 log_dir 下
        通过不同的函数输出不同等级的日志
        """
        super().__init__(name)

        self.dir_log = log_dir
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        self.level = logging.DEBUG
        self.setLevel(self.level)

        # 日志文件
        self.name_log = "log_cola_{}".format(name)
        self.path_log = os.path.join(self.dir_log, self.name_log)
        self.handler = logging.FileHandler(self.path_log) 
        self.handler.setLevel(self.level)

        # 日志输出格式
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)

        self.addHandler(self.handler)


if __name__ == "__main__":
    logger = Logger("实验测试")
    logger.debug("Cola debug.")
    logger.info("Cola info.")

