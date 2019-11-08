#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @file   : stdnes_by_dict.py
# @author : Zhutian Lin
# @date   : 2019/10/12
# @version: 1.2
# @desc   : 完成字典数据结构的网络优化，提升选边效率
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

'''
本文件是用来输入标准的node1，node2，label，time文件
输出两类文件，一类是带label的文件，一类是不带label的文件
这两类文件下面又分出训练集（包含新边和旧边）与测试集

'''
'''
第一类：前75%为原图，原图总文件叫train_graph
第二类：原图的前75%作为embedding的起始图，叫train_init_graph
第三类：原图后25%作为后续图，叫train_new_graph
第四类：训练图，文件叫test_graph
第五类：完整图叫complete_grah
'''

class spliter:
    def __init__(self):
        self.complete_dataset = np.loadtxt("E:\\MLcode\\New_Code\\Dataset\\foursq\\foursq2014_TKY_edge_format.txt", delimiter=',')  # 完整的dataset
        # self.complete_dataset = np.loadtxt("E:\\MLcode\\New_Code\\Dataset\\Phone\\Phone_edge_format.txt", delimiter='\t')  # 完整的dataset
        self.root_address = "E:\\MLcode\\New_Code\\Dataset\\foursq"
        self.train_node = []
        self.test = []
        self.train = []

        self.train_old_edge = []
        self.train_new_edge = []

        self.train_old_edge_without_label = []
        self.train_new_edge_without_label = []
        self.test_without_label = []
    def get_init_train(self):
        logging.info("get_init_train")
        self.complete_dataset = pd.DataFrame(self.complete_dataset, columns=['node1', 'node2', 'label' ,'time'])
        self.complete_dataset.sort_values(by=['time'], ascending=True)
        split = int(np.ceil(len(self.complete_dataset)/4*3))
        tmp_dataset = self.complete_dataset.values
        self.train = tmp_dataset[0:split]
        self.test = self.complete_dataset.values[split-1:-1]

        self.train = np.array(self.train)
        self.train = self.train.astype(np.int32)
        self.train = self.train.tolist()

        self.test = np.array(self.test)
        self.test = self.test.astype(np.int32)
        self.test = self.test.tolist()
        return

    def split_train(self):
        logging.info("split_train")
        split = int(np.ceil(len(self.train)/4*3))
        self.train_old_edge = self.train[0:split]
        self.train_new_edge = self.train[split-1:-1]

    def save_data(self):
        logging.info("save_data")
        train_without_label = [[edge[0],edge[1],edge[3]] for edge in self.train]
        #  一方面是剥除label，另一方面是转换为edgelist形式
        self.train_old_edge_without_label = [[edge[0], edge[1], edge[3]] for edge in self.train_old_edge]
        self.train_new_edge_without_label = [[edge[0], edge[1], edge[3]] for edge in self.train_new_edge]
        self.test_without_label = [[edge[0], edge[1], edge[3]] for edge in self.test]
        format = "\\edge_format"
        #  首先是存成第三种方法的文件
        self.train_new_edge_without_label = np.array(self.train_new_edge_without_label)
        self.train_new_edge_without_label = self.train_new_edge_without_label.astype(np.int32)
        new_net = nx.DiGraph()  # 建立有向图网络对象
        new_net.add_weighted_edges_from(train_without_label)  # 注意！新图是最后一个时刻的图，是全图！！！！对于这个dynamic方法而言
        nx.write_weighted_edgelist(new_net, self.root_address+format+'\\bn.edgelist', delimiter=' ')
        print("转换完毕，格式为edgelist文件！")

        self.train_old_edge_without_label = np.array(self.train_old_edge_without_label)
        self.train_old_edge_without_label = self.train_old_edge_without_label.astype(np.int64)
        network = nx.DiGraph()  # 建立有向图网络对象
        network.add_weighted_edges_from(self.train_old_edge_without_label)  # 输入网络
        nx.write_weighted_edgelist(network, self.root_address+format+'\\b.edgelist', delimiter=' ')
        print("转换完毕，格式为edgelist文件！")

        #  保存有label和无label文件

        self.write_file(self.train_old_edge, self.root_address+format+"\\has_label\\train_init_graph.txt",True)
        self.write_file(self.train_new_edge, self.root_address+format+"\\has_label\\train_new_graph.txt",True)
        self.write_file(self.test, self.root_address + format+"\\has_label\\test_graph.txt",True)
        self.write_file(self.train, self.root_address+format+"\\has_label\\train_graph.txt",
                        True)

        self.write_file(self.train_old_edge_without_label, self.root_address+format+"\\no_label\\train_init_graph.txt",False)
        self.write_file(self.train_new_edge_without_label, self.root_address+format+"\\no_label\\train_new_graph.txt",False)
        self.write_file(self.test_without_label, self.root_address+format+"\\no_label\\test_graph.txt",False)
        self.write_file(train_without_label, self.root_address+format+"\\no_label\\train_graph.txt",False
                        )
    def write_file(self,content,path,option=True):
        logging.info("写入"+path)
        f = open(path, 'w')  # 打开文件
        if option == True:
            for edge in content:
                f.write(str(edge[0])+'\t'+str(edge[1])+'\t'+str(edge[2])+'\t'+str(edge[3]))
                f.write('\n')

        else:
            for edge in content:
                f.write(str(edge[0])+'\t'+str(edge[1])+'\t'+str(edge[2]))  #  因为这个是已经删除该过label得了
                f.write('\n')
        f.close()



if __name__ == '__main__':
    sp = spliter()
    sp.get_init_train()
    # sp.get_legal_train_test()
    sp.split_train()
    sp.save_data()