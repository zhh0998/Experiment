import time


def pre_process():
    DBLP_path = "DBLPEdges"
    edge_file_path = "DBLP_edge_format.txt"
    node_file_path = "DBLP_node_format.txt"
    node2id = dict()
    nid_cnt = 0

    # 得到节点id
    with open(DBLP_path, 'r') as rf:
        for line in rf:
            toks = line.strip().split(',')
            # 检查非分隔符逗号问题
            if toks.__len__() != 6:
                if (toks[1] != 'author') | (toks[-3] != 'article'):
                    print('error')
                    exit(-1)

            node_1 = toks[0]
            node_2 = ''.join(toks[2:-3])
            node2id[node_1] = str(nid_cnt+1)
            node2id[node_2] = str(nid_cnt+2)
            nid_cnt += 2
    print("得到节点id")

    # 写文件
    ef = open(edge_file_path, 'w')
    nf = open(node_file_path, 'w')
    with open(DBLP_path, 'r') as rf2:
        for line in rf2:
            toks = line.strip().split(',')
            # 检查非分隔符逗号问题
            if toks.__len__() != 6:
                if (toks[1] != 'author') | (toks[-3] != 'article'):
                    print('error')
                    exit(-1)

            node_1_str = toks[0]
            node_1_label = toks[1]
            node_2_str = ''.join(toks[2:-3])  # join函数比+的效率更高
            node_2_label = toks[-3]
            edge_label = toks[-2]
            time_str = toks[-1]

            # 处理时间戳
            time_array = time.strptime(time_str, '%Y-%m-%d')
            time_stamp = str(int(time.mktime(time_array)))

            ef.write(','.join([node2id[node_1_str], node2id[node_2_str], edge_label, time_stamp])+'\n')
            nf.write(','.join([node2id[node_1_str], node_1_label, node2id[node_2_str], node_2_label, time_stamp])+'\n')

    ef.close()
    nf.close()


if __name__ == "__main__":
    pre_process()
