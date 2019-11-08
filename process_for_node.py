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

class preprocessing_for_node:
    def __init__(self):
        self.root_address = "E:\\MLcode\\New_Code\\Dataset\\foursq\\"
        self.original_file = "foursq2014_TKY_node_format.txt"

        self.complete_dataset = []
        self.test = []
        self.train = []

        self.train_old_edge = []
        self.train_new_edge = []

    def get_init_train_test(self):
        with open(self.root_address+self.original_file, 'r') as file:
            for line in file:
                line = line.strip('\n')
                line_list = line.split(',')
                self.complete_dataset.append([int(line_list[0]), 0, int(line_list[2]), 1, int(line_list[4])])
        self.complete_dataset = pd.DataFrame(self.complete_dataset, columns=['nid1', 'label_1', 'nid_2', 'label_2', 'time'])
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

    def split_train(self):
        logging.info("split_train")
        split = int(np.ceil(len(self.train)/4*3))
        self.train_old_edge = self.train[0:split]
        self.train_new_edge = self.train[split-1:-1]

    def write_file(self, content, path):
        logging.info("写入" + path)
        f = open(path, 'w')  # 打开文件
        for edge in content:
            split_str = '\t'
            content_str = ""
            for i in range(len(edge)):
                if i == 0:
                    content_str = content_str+str(edge[i])
                else:
                    content_str = content_str+split_str+str(edge[i])
            f.write(content_str)
            f.write('\n')
        f.close()

    def save_data(self):
        logging.info("save_data")
        format = "\\node_format"
        self.write_file(self.train_old_edge, self.root_address+format+"\\train_init_graph.txt")
        self.write_file(self.train_new_edge, self.root_address+format+"\\train_new_graph.txt")
        self.write_file(self.test, self.root_address + format+"\\test_graph.txt")
        self.write_file(self.train, self.root_address+format+"\\train_graph.txt")

if __name__ =='__main__':
    ps = preprocessing_for_node()
    ps.get_init_train_test()
    ps.split_train()
    ps.save_data()