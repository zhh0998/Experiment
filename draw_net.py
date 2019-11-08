import networkx as nx
import numpy as np
import matplotlib.pyplot as plt  # 导入pyplot而不是matplotlib本身
class network_drawer:
    def __init__(self):
        self.G = nx.MultiGraph()

    def import_network(self):
        network = np.loadtxt("Phone_whole_graph.txt", delimiter='\t')
        network.astype(np.int32)
        for edge in network:
            self.G.add_node(edge[0],value=edge[1])
            self.G.add_node(edge[2], value=edge[3])
            self.G.add_edge(edge[0],edge[2],value=edge[4])

    def draw(self):
        print("原始图")
        nx.draw_networkx_edges(self.G,pos=nx.random_layout(self.G),with_labels=False,)
        plt.savefig("phone.png")
        plt.show()
        # print("spring_layout")
        # nx.draw(self.G, pos=nx.spring_layout(self.G))
        # plt.savefig("phone_spring.png")
        # plt.show()
        # print("circular_layout")
        # nx.draw(self.G, pos=nx.circular_layout(self.G))
        # plt.savefig("phone_circular.png")
        # plt.show()
        # print("circular_layout")
        # nx.draw(self.G, pos=nx.shell_layout(self.G))
        # plt.savefig("phone_shell_layout.png")
        # plt.show()
        # print("spectral_layout")
        # nx.draw(self.G, pos=nx.spectral_layout(self.G))
        # plt.savefig("phone_spectral_layout.png")
        # plt.show()
        print("画完辽！")

        # painting.show()
if __name__ == '__main__':
    nd = network_drawer()
    nd.import_network()
    nd.draw()
