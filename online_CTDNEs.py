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

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # 日志格式化输出
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"  # 日期格式
fp = logging.FileHandler('config.log', encoding='utf-8')
fs = logging.StreamHandler()
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fp,fs])  # 调用handlers=[fp,fs]
#  输出到命令行中
#  计时器
import time


class online_ctdns:
    def __init__(self):
        self.backward_dict = {}  # 以后节点为索引的传播路径
        self.net = []  # 整个原来的图
        self.walk_path = "walk.txt"
        self.new_edge_set = []
        self.time_new_edge_dict = {}  # 以时间为索引，边为value的dict

        self.N = len(self.backward_dict.keys())
        self.r = 10  # 每个节点应该行走多少次
        self.l = 80  # 一个行走最长是多少
        self.w = 10  # 一个行走最短是多少
        # self.beta = self.r*self.N*(self.l-self.w+1)  # 先写小一点
        self.beta = self.N * (self.l - self.w + 1)  # 行走时间约束
        self.d = 128  # embedding向量维度
        self.granularity = 10000  # 切分粒度
    # def experiment_get_init_model(self):
    #     name = "dataset/train_init_graph_bf.txt"  # 第一次使用这个作为训练的时间进行测试
    #
    #     #  初始构建net和network
    #     self.net = np.loadtxt(name, delimiter='\t')
    #     #  把内容进行强制类型转换
    #     self.net = self.net.astype(np.int64)
    #     #  构建node_dict
    #     for i in range(len(self.net)):
    #         #  update正向的dict
    #         forward_tmp_node = self.forward_dict.get(self.net[i][0])
    #         if forward_tmp_node == None :
    #             forward_tmp_edge= {self.net[i][0]: [self.net[i][1: 3].tolist()]}  # 注意是一个list里套list
    #             self.forward_dict.update(forward_tmp_edge)
    #         else:
    #             forward_tmp_node.append(self.net[i][1: 3].tolist())
    #
    #         backward_tmp_node = self.backward_dict.get(self.net[i][1])
    #         if backward_tmp_node == None :
    #             backward_tmp_edge = {self.net[i][1]: [[self.net[i][0], self.net[i][2]]]}
    #             self.backward_dict.update(backward_tmp_edge)
    #         else:
    #             backward_tmp_node.append([self.net[i][0], self.net[i][2]])
    #     self.net.tolist()

    def import_new_net(self):
        # TODO：这里的括号问题还没解决，上面是改好的
        path = "dataset/whole.txt"
        self.new_edge_set = np.loadtxt(path, delimiter='\t')
        self.new_edge_set = self.new_edge_set.astype(np.int64)
        for i in range(len(self.new_edge_set)):
            #  update正向的dict
            # forward_tmp_node = self.forward_dict.get(self.new_edge_set[i][0])
            #
            # if forward_tmp_node == None:
            #     forward_tmp_edge = {self.new_edge_set[i][0]: [self.new_edge_set[i][1: 3].tolist()]}  # 注意是一个list里套list
            #     self.forward_dict.update(forward_tmp_edge)
            # else:
            #     forward_tmp_node.append(self.new_edge_set[i][1: 3].tolist())
            #
            backward_tmp_node = self.backward_dict.get(self.new_edge_set[i][1])

            if backward_tmp_node == None:
                backward_tmp_edge = {self.new_edge_set[i][1]: [[self.new_edge_set[i][0], self.new_edge_set[i][2]]]}
                self.backward_dict.update(backward_tmp_edge)
            else:
                backward_tmp_node.append([self.new_edge_set[i][0], self.new_edge_set[i][2]])

            time_tmp_node = self.time_new_edge_dict.get(self.new_edge_set[i][2])

            if time_tmp_node == None:
                time_tmp_node = {self.new_edge_set[i][2]: [self.new_edge_set[i][0:2].tolist()]}
                self.time_new_edge_dict.update(time_tmp_node)
            else:
                time_tmp_node.append(self.new_edge_set[i][0:2].tolist())
        self.N = len(self.backward_dict.keys())
        self.beta = self.N * (self.l - self.w + 1)  # 行走时间约束
        self.new_edge_set.tolist()

    def update_by_time(self):
        time_list = [time for time in self.time_new_edge_dict]
        for i in range(len(time_list)//self.granularity):
            if (i+1)*5 >= len(time_list):
                time_slice = time_list[i*self.granularity:len(time_list)]
            else:
                time_slice = time_list[i*self.granularity:(i+1)*self.granularity]
            for time in time_slice:
                self.get_walks_set(time, walk_num=20)
            #  这个是对于每个切片而言的
            try:
                texts = self.loadDataset(self.walk_path)
                model = Word2Vec(texts, sg=1, size=self.d, window=5, min_count=5, negative=3, sample=0.001, hs=1, workers=4)
                # 建立vocabulary

                #  保存模型
                model.save('dataset/'+str(i) + '_model')
            except:
                print("行走太短了，目前无法训练")


    def get_walks_set(self, time, walk_num=30):  # 这里感觉写的乱七八糟的，beta这里传值到时候有了封装好的函数就要修改

        '''
        TODO：改了改了！！！！
        静态的时序网络的embedding训练
        :param walk_index 行走的次数（算是第几次行走，存在文件名里头）
        无返回值，向量均存在文件里
        '''
        logging.info("获取" + str(time) + "时间下的一次更新！")
        w = self.w
        i = 0
        c = 0
        #  做一个加上时间的新边表（下面就开始遍历新边了）
        tmp_list = self.time_new_edge_dict.get(time)
        curr_new_edge_list = []
        for edge in tmp_list:  # append无返回值！！！！
            edge.append(time)
            curr_new_edge_list.append(edge)

        # TODO：选边的排序问题是个问题，因为要逆向游走了
        for each_new_edge in curr_new_edge_list:
            logging.info(each_new_edge)
            for i in range(walk_num):  # 采用的办法是就走五次，能取多少就是多少
                walk_index = self.get_backward_walk(each_new_edge)  # 这里简化为l，对结果影响不大
                if len(walk_index) > w:
                    logging.info("长度足够，可以选取！！")
                    c = c + (len(walk_index) - w + 1)
                    i = i + 1
                    f = open('walk.txt', 'a')  # 打开文件
                    walk_index.reverse()  # 反向行走一定要取反噢
                    f.write(str(walk_index))
                    f.write('\n')
                    f.close()
                # else:
                #   logging.info("反向行走长度过短，舍去")

    def loadDataset(self, infile):
        '''
        把walks读入，并且变成单词的集合格式
        :param infile:输入的文件名
        :return: 返回格式符合要求的dataset
        '''

        #  把index的文件里的格式整理成[]都删掉的情况，录入tmpindex
        fp = open('tmpindex.txt', 'w')
        lines = open(infile).readlines()
        for s in lines:
            fp.write(s.replace('[', '').replace(']', ''))
        fp.close()
        #  打开tmpindex，每个都是string，这样可以看成是单词了
        f = open('tmpindex.txt', 'r')
        sourceInLine = f.readlines()
        dataset = []
        for line in sourceInLine:
            temp1 = line.strip('\n')
            temp2 = temp1.split(', ')
            dataset.append(temp2)
        return dataset

    def get_backward_walk(self, start_edge):
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
        for p in range(1, l - 1):  # 注意开闭

            legal_neighbour_edge = self.get_legal_neighbour_edges(curr_edge)  # 传入curr_edge可以相当于传入了时间和位置，每次更新curr_edge即可
            #  需要按照时间次序排个序
            if legal_neighbour_edge != []:
                edge_candi = pd.DataFrame(legal_neighbour_edge, columns=['node2', 'time'])
                edge_candi = edge_candi.sort_values(by=['time'], ascending=True)
                legal_neighbour_edge = edge_candi.values
            if len(legal_neighbour_edge) > 0:  # 因为这个邻居节点集合是list
                choose_neighbour = self.sample_next_edge(curr_edge, legal_neighbour_edge)  # 由于时序不减，所以后面不可能取到这个原来取过的边
                # logging.info("从所有合法邻边中按照概率选一个")
                curr_walk_index.append(choose_neighbour[1])  # 把下一个节点写进去
                curr_edge = choose_neighbour  # 因为这个返回的就是一条完整的边，所以可以写这个
            else:
                # logging.info("结束本轮反向行走！")
                return curr_walk_index

        return curr_walk_index  # 返回可行的walk和walk代表的index

    def get_legal_neighbour_edges(self, curr_edge):
        '''
        反向游走，时间改为减
        调通了
        获取一系列的符合要求的edge，以curr_edge的node2为起点，以curr_edge的t为时间约束
        :return legal_neighbour 就是一个二重list的结构，返回的是合法的邻边
        '''
        #  注意curr_edge传进来的应当还是完整形态的node1，node2，time
        next_node = curr_edge[1]  # 1是node2的索引
        neighbour = self.backward_dict.get(next_node)  # 是查的出来的，接下来就是取一个合法邻居了
        curr_time = curr_edge[2]
        if neighbour == None:
            return []
        else:
            #  暂时先写成>因为有可能自己跟自己绕着走
            # 这里是把node2和time存进来了，所以1索引是时间,保证不往回走
            #  TODO:这块的neighbour没写好！！！！注意，还是原来的取neighbour的函数有问题，要不然就是dict有问题，要不然就是函数返回值有问题
            legal_neighbour = [shift for shift in neighbour if shift[1] < curr_time]
            return legal_neighbour

    def sample_next_edge(self, curr_edge, legal_neighbour_edge, option=0):
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
        #  新边格式仍然为node1，node2，time
        #  按概率取出一个符合要求的边,如果没有就返回为[]这个需要注意了！！
        return sample


if __name__ == '__main__':
    oc = online_ctdns()
    oc.import_new_net()
    oc.update_by_time()

