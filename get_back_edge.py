import pandas as pd
import numpy as np
import time
import logging
import numpy as np
import networkx as nx
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # 日志格式化输出
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"                        # 日期格式
# fp = logging.FileHandler('config.log', encoding='utf-8')
fs = logging.StreamHandler()
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fs])    # 调用handlers=[fp,fs]
class back_writer:
    def __init__(self):
        self.read_path = "dataset/test_graph.txt"
        self.write_path = "dataset/test_graph_bf.txt"
    def get_back_edge(self):
        front_graph = np.loadtxt(self.read_path,delimiter="\t")
        f = open(self.write_path, 'a')  # 打开文件
        for edge in front_graph:
            f.write(str(int(edge[0]))+'\t'+str(int(edge[1]))+'\t'+str(int(edge[2]))+'\n')
            f.write(str(int(edge[1]))+'\t'+str(int(edge[0]))+'\t'+str(int(edge[2]))+'\n')


        f.close()

if __name__ == '__main__':
    bw = back_writer()
    bw.get_back_edge()