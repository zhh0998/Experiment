# @desc   : 完成点分类任务
import pandas as pd
import numpy as np
import logging
import sklearn
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.cluster import KMeans
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # 日志格式化输出
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"                        # 日期格式
fp = logging.FileHandler('config.log', encoding='utf-8')
fs = logging.StreamHandler()
from sklearn.metrics import accuracy_score
from sklearn import metrics as mr

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fs, fp])    # 调用handlers=[fp,fs]
#  输出到命令行中
#  计时器
import time


class node_cluster:
    def __init__(self):
        logging.info("初始化分类器")
        self.vectors_dict = {}  # 每个点对应的字典{点index(str):向量(nparray)} 这个str可不能带小数点
        self.node_dict={}  # 每个点的label
        self.node_vec=[]  # 备选点
        self.node_label = []
    # 操作：首先导入数据集，格式不变；原来的test_set输入为以空格为分隔符的node1 node2，主程序的embedding，格式不变

    def import_model(self, model_name, option=0):
        #  0代表主方法，model_name代表的是主函数输出的节点向量
        if option == 0:
            word_vectors = np.loadtxt(model_name, delimiter=' ')
            for line in word_vectors:
                tmp = {str(int(line[0])): line[1:-1]}
                self.vectors_dict.update(tmp)

        if option == 1:
            model = Word2Vec.load(model_name)
            word_vectors = KeyedVectors.load(model_name)

            # 构造新字典
            for key in word_vectors.wv.vocab.keys():
                tmp = {key: model.wv.__getitem__(key)}
                self.vectors_dict.update(tmp)




    def import_node(self, dsname):
        # 统一操作为str
        node_data = np.loadtxt(dsname, delimiter="\t").tolist()
        for edge in node_data:
            tmp_1 = {str(int(edge[0])): int(edge[1])}
            tmp_2 = {str(int(edge[2])): int(edge[3])}  # 这里和其他的不一样！！！！！这里用的是int，其他地方是str
            self.node_dict.update(tmp_1)
            self.node_dict.update(tmp_2)

        for node in self.vectors_dict:
            self.node_vec.append(self.vectors_dict.get(node))
            self.node_label.append(self.node_dict.get(node))

    def cluster(self):

        estimator = KMeans(n_clusters=2)  # 构造聚类器
        estimator.fit(self.node_vec)  # 聚类
        label_predict = estimator.predict(self.node_vec)  # 获取聚类标签


        print(mr.normalized_mutual_info_score(label_predict,self.node_label))



if __name__ == '__main__':
    nc = node_cluster()
    nc.import_model("vec_final.emb",0)
    nc.import_node("train_graph.txt")
    nc.cluster()