import numpy as np
import time
import hashlib
class processer:
    def __init__(self):
        self.DBLP_path = "E:\MLcode\DBIS\pre_processing\dirty_dataset\DBLPEdges"
        self.edge_file_path = "E:\MLcode\DBIS\pre_processing\dirty_dataset\\DBLP_edge_format.txt"
        self.node_file_path = "E:\MLcode\DBIS\pre_processing\dirty_dataset\\DBLP_node_format.txt"
        f1 = open(self.edge_file_path, 'a')  # 打开文件
        f2 = open(self.node_file_path,'a')
        with open(self.DBLP_path, 'r') as file:
            for line in file:
                #  处理f1
                line = line.strip('\n')
                line_list = line.split(',')
                if 'author' != line_list[1] or 'article' != line_list[-3]:
                    print("例外")
                node_1 = id(line_list[0])
                node_2_str = ""
                index_1 = line_list.index('author')
                index_2 = line_list.index('article')
                if index_1+2 != index_2:
                    for i in range(index_1+1, index_2):
                        node_2_str = node_2_str+line_list[i]
                else:
                    node_2_str = line_list[2]
                node_2 = id(node_2_str)
                time_array = time.strptime(line_list[-1], '%Y-%m-%d')
                time_stamp = int(time.mktime(time_array))

                f1.write(str(node_1)+"\t"+str(node_2)+"\t"+line_list[-2]+"\t"+str(time_stamp))
                f1.write('\n')
                print("写入一行")
                f2.write(str(node_1)+'\t'+'author'+"\t"+str(node_2)+"\t"+'article'+"\t"+line_list[-2]+"\t"+str(time_stamp))
                f2.write('\n')

        f2.close()
        f1.close()



if __name__=="__main__":
    ps = processer()
