#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @file   : stdnes_by_dict.py
# @author : Zhutian Lin
# @date   : 2019/10/12
# @version: 1.2
# @desc   : 完成字典数据结构的网络优化，提升选边效率
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
# import logging
# logging.basicConfig(filename="config.log", filemode="w", format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
#                     level=logging.INFO)  #输出到文件里

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # 日志格式化输出
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"                        # 日期格式
# fp = logging.FileHandler('config.log', encoding='utf-8')
fs = logging.StreamHandler()
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fs])    # 调用handlers=[fp,fs]
#  输出到命令行中
#  计时器
import time



class Network:

    def __init__(self):  # 构造函数的写法是__init__
        '''
        调通
        初始化Network类，主要是导入文件
        '''
        #  注意原来的表格必须是时间升序的！！！！！！这个升序是在数据预处理中完成
        #  格式：读入的数据集必须是node1\t node2\t time，文件格式是txt

        #  逐行读取数据矩阵
        name = "dataset/train_graph.txt"  # 第一次使用这个作为训练的时间进行测试

        #  初始构建net和network
        self.net = np.loadtxt(name, delimiter='\t')
        self.network = {}

        #  把内容进行强制类型转换
        self.net = self.net.astype(np.int32)

        #  构造字典network
        for i in range(len(self.net)):
            tmp_node = self.network.get(self.net[i][0])
            if tmp_node == None :
                tmp= {self.net[i][0]:[self.net[i][1: 3].tolist()]}  # 注意是一个list里套list
                self.network.update(tmp)
            else:
                tmp_node.append(self.net[i][1: 3].tolist())

        #  构建net为list
        self.net.tolist()

        #  初始化超参数
        self.N = len(self.network.keys())
        self.r = 10  # 每个节点应该行走多少次
        self.l = 80  # 一个行走最长是多少
        self.w = 10  # 一个行走最短是多少
        # self.beta = self.r*self.N*(self.l-self.w+1)  # 先写小一点
        self.beta = self.N * (self.l - self.w + 1)  # 行走时间约束
        self.d = 128  # embedding向量维度

    def ctdns(self):
        '''
        ctdns的主程序
        '''
        # 由于修改了get_walks_set这个函数，后面要调用的时候得直接读文件了，毕竟文件比较靠谱，这么大个数据悬在内存里emmm
        # 这里认为walks是走好的前提下进行这个学习
        # 导入index
        texts = self.loadDataset('dataset/tmpindex.txt')  # 把每个walk都看作是一个句子
        # 使用word2vec训练
        # 确定模型为word2vec

        # model.build_vocab(texts)
        # 默认min_count是5
        model = Word2Vec(texts, sg=1, size=self.d,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
        # 建立vocabulary

        #  保存模型
        model.save('dataset/temporal_model_skipgram')
        #  测试以下模型情况,可以跑的通昂！！！！
        # print(model.__getitem__("27448"))

    def loadDataset(self, infile):
        '''
        把walks读入，并且变成单词的集合格式
        :param infile:输入的文件名
        :return: 返回格式符合要求的dataset
        '''

        #  把index的文件里的格式整理成[]都删掉的情况，录入tmpindex
        # fp = open('dataset/tmpindex.txt', 'w')
        # lines = open(infile).readlines()
        # for s in lines:
        #     fp.write(s.replace('[', '').replace(']', ''))
        # fp.close()
        #  打开tmpindex，每个都是string，这样可以看成是单词了
        f = open('dataset/tmpindex.txt', 'r')
        sourceInLine = f.readlines()
        dataset = []
        for line in sourceInLine:
            temp1 = line.strip('\n')
            temp2 = temp1.split(' ')#fixme:修改为了空格，之前是逗号，上面的代码按道理要打开
            dataset.append(temp2)
        return dataset

    def get_walks_set(self):  # 这里感觉写的乱七八糟的，beta这里传值到时候有了封装好的函数就要修改

        '''
        静态的时序网络的embedding训练
        :param beta: 时序上下文窗口的大小
        :param w: 窗口大小
        :param d: embedding的维度
        无返回值，向量均存在文件里
        '''
        print("运行ctdnes程序！")
        w = self.w
        l = self.l
        beta = self.beta
        l = 80  # 论文里取80
        c = 0
        logging.debug("beta="+str(beta))

        while beta-c > 0:
            logging.info("---------------------C="+str(c)+'-------------------------------')
            logging.info(self.beta)
            logging.info("进程为："+str(c/self.beta))
            # root = self.sample_init_edge()  # 这个是原始的函数，如果跑得动那就弃了
            root = self.get_init_edge()
            logging.info("开始一次行走")
            walk_index = self.temporal_walk(root, w+beta-c-1)

            if len(walk_index) > w:
                #  不要写在内存里，直接写进文件里
                # walks.append(walk)
                # walks_index.append(walk_index)
                #  传出来两个都是list类型，walks里面是Series类型，walks_index里面是int类型
                c = c+(len(walk_index)-w+1)
                # print(walks_index)
                #  写入文件
                f=open('dataset/wait/tmpindex.txt', 'a') # 打开文件
                f.write(str(walk_index))
                f.write('\n')
                f.close()
            # else:
                # logging.info("长度过短，舍去")

    def temporal_walk(self, start_edge, c):  # 取消l变量
        '''
        调通
        选出一个合法的游走路径
        :return: 从给定节点下的合法游走路径
        '''
        #  预先设定值
        l = self.l
        #  这些值之后每次迭代都需要更新
        #  把edge看作是一个一维list
        curr_walk_index = [start_edge[0], start_edge[1]]  # 取出起始点和第一个点，index对应的是0，1
        curr_edge = start_edge  # 当前的走到的边
        for p in range(1, min(l, c)-1):  # 注意开闭

            legal_neighbour_edge = self.get_legal_neighbour_edges(curr_edge)  # 传入curr_edge可以相当于传入了时间和位置，每次更新curr_edge即可
            if len(legal_neighbour_edge) > 0:  # 因为这个邻居节点集合是list
                choose_neighbour = self.sample_next_edge(curr_edge, legal_neighbour_edge)  # 由于时序不减，所以后面不可能取到这个原来取过的边
                # logging.info("从所有合法邻边中按照概率选一个")
                curr_walk_index.append(choose_neighbour[1])  # 把下一个节点写进去
                # print(curr_walk_index)
                curr_edge = choose_neighbour  # 因为这个返回的就是一条完整的边，所以可以写这个
            else:
                # logging.info("结束本轮行走！")
                return curr_walk_index
                # return curr_walk, curr_walk_index

        return curr_walk_index  # 返回可行的walk和walk代表的index

    def get_init_edge(self, option=0):
        '''
        调通
        取出一开始的边，按照边的时间增序所取
        :return: 返回所取的初始边
        '''
        prob_list = []
        # 假设为线性的，option为0
        if option == 0:
            denominator = sum(range(len(self.net) + 1))
            for i in range(len(self.net)):
                prob_list.append((i + 1) / denominator)  # 不能每个循环都调用一个range这样也太慢了
        # 假设为exp的，option为1
        if option == 1:
            tmplist = []
            t_min = min(self.net[:, 2])
            t_max = max(self.net[:, 2])
            for i in range(len(self.net)):
                tmplist.append(np.exp((self.net[i][2] - t_min) / (t_max - t_min)))
            tmplist = np.array(tmplist)
            prob_list = tmplist * (1 / sum(tmplist))

        # index = np.random.choice(range(0, len(self.net)), p=prob_list)
        index = np.random.choice(range(0, len(self.net)))
        return self.net[index, :]

    def get_legal_neighbour_edges(self, curr_edge):
        '''
        调通了
        获取一系列的符合要求的edge，以curr_edge的node2为起点，以curr_edge的t为时间约束
        :return legal_neighbour 就是一个二重list的结构，返回的是合法的邻边
        '''
        next_node = curr_edge[1]  # 1是node2的索引
        neighbour = self.network.get(next_node)  # 是查的出来的，接下来就是取一个合法邻居了
        curr_time = curr_edge[2]
        if neighbour == None:
            return []
        else:
            #  暂时先写成>因为有可能自己跟自己绕着走
            # 这里是把node2和time存进来了，所以1索引是时间,保证不往回走
            legal_neighbour = [shift for shift in neighbour if shift[1] > curr_time and shift[0] != curr_edge[0]]
            return legal_neighbour

    def sample_next_edge(self, curr_edge, legal_neighbour_edge,  option=0):
        '''
        调通线性
        从邻居边中按照概率找到能用的
        :param legal_neighbour_edge: 邻边list
        :return: 按概率取出一个符合要求的边,如果没有就返回为[]这个需要注意了！！
        '''
        mid_node = curr_edge[1]  # 这个curr_edge的node2充当中继点的作用
        prob_list = []
        if len(legal_neighbour_edge) == 0:
            return []

        if option == 0:  # 这里是线性概率表
            denominator = sum(range(len(legal_neighbour_edge) + 1))
            for i in range(len(legal_neighbour_edge)):
                prob_list.append((i + 1) / denominator)  # 不能每个循环都调用一个range这样也太慢了
        if option == 1:
            #  TODO：这里的exp概率表还没有写
            print(0)  # 这里是非线性概率表
        index = np.random.choice(range(0, len(legal_neighbour_edge)), p=prob_list)  # 这里先写死为线性的，之后其他的再做实验
        sample = [mid_node, legal_neighbour_edge[index][0], legal_neighbour_edge[index][1]]  # 合并为符合条件的新边
        #  按概率取出一个符合要求的边,如果没有就返回为[]这个需要注意了！！
        return sample


nw = Network()
start_time = time.time()
# nw.get_walks_set()
end_time1 = time.time()
logging.info("行走结束，时间为：")
logging.info(start_time-end_time1)
nw.ctdns()
end_time2 = time.time()
logging.info("训练结束，时间为：")
logging.info(end_time2-end_time1)

