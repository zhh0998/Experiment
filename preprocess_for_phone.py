import numpy as np
import time
import hashlib
class processer:
    def __init__(self):
        #  注意！！！！目前用的还是原来的代码集里的处理
        self.phone_path = "E:\MLcode\DBIS\pre_processing\dirty_dataset\Phone_data"
        self.edge_file = "E:\MLcode\DBIS\pre_processing\dirty_dataset\\Phone_edge_format.txt"
        self.node_file = "E:\MLcode\DBIS\pre_processing\dirty_dataset\\Phone_node_format.txt"
        f1 = open(self.edge_file, 'a')  # 打开文件
        f2 = open(self.node_file, 'a')
        with open(self.phone_path, 'r') as file:
            for line in file:
                line = line.strip('\n')
                line_list = line.split(',')

                #
                f1.write(line_list[1]+"\t"+line_list[3]+"\t"+line_list[5]+"\t"+line_list[0])
                f1.write('\n')

                f2.write(line_list[1]+"\t"+line_list[2]+"\t"+line_list[3]+"\t"+line_list[4]+"\t"+line_list[5]+"\t"+line_list[0])
                f2.write('\n')
                print("写入一行")
        f2.close()
        f1.close()



if __name__=="__main__":
    ps = processer()
