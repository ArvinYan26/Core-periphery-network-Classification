import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_blobs, load_digits
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from sklearn.preprocessing import Normalizer
import networkx as nx
import math
import pandas as pd
from networkx.algorithms.distance_measures import center
from collections import Counter
from sklearn.model_selection import train_test_split
#from GetCOVID_19Data import get_data
from GetCOVID_19Data1 import get_data  #原图像傅里叶变换，两类（正常和新冠）
from sklearn import preprocessing
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
#import cpalgorithm as cp
#import igraph as ig
import matplotlib as mpl


class CorePeriphery(object):

    def __init__(self, num_class, in_rate):
        '''
        :param per_class_data_len:  length of per class data
        :param num_classes:         num of classes
        :param k:                   para of kNN
        '''
        self.num_class = num_class
        self.init_rate = in_rate
        self.rcc = []   #存储是哪个网络中每类数据的rich_clup_coefficient
        #self.c_r = c_rate

        self.per_class_data_len = None
        self.train_len = None
        self.train_x = None
        self.data_idxs_list = []
        self.train_y = None

        self.neigh_models = []  #
        self.e_radius = []

        self.G_list = []
        self.mean_dis_list = []
        self.nodes_list = []
        self.edges_list = []
        self.len_list = []  #存储每个组件大小
        self.net_measures = []  # {1:{'averge_degree':[]}}

    def data_processing(self, adj_matrix):
        """
        距离矩阵归一化
        :param adj_matrix:距离矩阵
        :return:
        """
        min_max_scaler = preprocessing.MinMaxScaler().fit(adj_matrix)
        new_adj = min_max_scaler.transform(adj_matrix)
        return new_adj

    def data_reorganization(self, x, y):
        self.train_x = x
        self.train_y = y
        self.train_len = len(x)

        labels = [i for i in Counter(y)]
        labels.sort()
        self.labels = labels
        # print("self.labels:", self.labels)
        self.num_classes = len(labels)
        self.data = []
        self.each_data_len = []

        "1. build network"
        for ith_class in labels:
            # label是按照顺序排的，0， 1， 2， 、, # 所以说从图上通过颜色可以看出来是哪一类
            idxs = np.argwhere(y == ith_class).reshape(-1)
            self.data_idxs_list.append(idxs)
            "adjacency matrix"
            dataset = x[idxs]
            self.data.append(dataset)
            data_len = len(dataset) #每类数据长度
            self.each_data_len.append(data_len)
            adj_matrix = euclidean_distances(dataset, dataset)

            #归一化距离矩阵，此处归一化，那么构建连边和插入新节点分类时都需要归一化距离，否则不同意会出现错误
            #adj_matrix = self.data_processing(adj_matrix)
            #print(adj_matrix)
            # 要先求两两平均距离，后面会改动数据。
            #std = np.std(adj_matrix) #方差
            mean_dis = np.sum(adj_matrix) / (data_len ** 2 - data_len)
            # print("mean_dis:", mean_dis)
            # mean_dis = mean_dis*self.t #变化阈值
            self.mean_dis_list.append(mean_dis)  # 平均差别
        if not self.mean_dis_list == []:
            self.mean_dis_list = sorted(list(set(self.mean_dis_list)))

        #print("self.mean_dis_list:", self.mean_dis_list, self.each_data_len)
        #print(len(self.data))
        #三类
        #self.new_data = np.vstack((self.data[0], self.data[1], self.data[2]))
        each_data = []
        #排序后的新数据集，self.data[0]中有两个元素，0元素是标签为0的所数据，1元素是标签为1的所有数据
        print("len:", len(self.data[0]), len(self.data[1]))
        new_data = np.vstack((self.data[0], self.data[1]))
        #each_data.extend([x, self.mean_dis_list, self.each_data_len])
        each_data.extend([new_data, self.mean_dis_list, self.each_data_len])
        print("self.mean_dis_list：", self.mean_dis_list)

        return each_data

    def fit(self, x_train, y_train):
        """
        Args:
            data: data[i][0]数据， data[i][1]每类数据平均距离， data[i][2]每类数据长度
        Returns: predict_label

        """
        # self.data[0]:数据， 1：平均距离， 2：每类数据长度
        self.new_data = self.data_reorganization(x_train, y_train)

        #print("self.data:", self.data)
        #for ith_class in range(self.num_class):

        print("self.data[0]:", len(self.new_data[0]), self.new_data[2])
        adj_matrix = euclidean_distances(self.new_data[0], self.new_data[0])
        #归一化距离矩阵
        #adj_matrix = self.data_processing(adj_matrix)
        adj_matrix[adj_matrix == 0] = 10000

        print("adj_matrix_or:", len(adj_matrix))
        print("distance:", self.data[1])
        #每一个节点最小距离找到构建连边，防止单节点出现

        for idx, item in enumerate(adj_matrix):
            min_idx = np.argmin(item)
            # 因为是对称矩阵
            adj_matrix[idx, min_idx] = 1
            adj_matrix[min_idx, idx] = 1

        #小于阈值的设置为1即连边
        print("平均距离:", self.new_data[1])
        adj_matrix[adj_matrix < np.min(self.new_data[1]) * self.init_rate] = 1
        #print("adj_matrix_1:", adj_matrix)
        # 将没有连边的部分都设置为0
        adj_matrix[adj_matrix != 1] = 0
        #self.G_list.append(nx.from_numpy_matrix(adj_matrix))

        #print("adj_matrix:", adj_matrix)
        self.G = nx.from_numpy_matrix(adj_matrix)  #将邻接矩阵转化为图

        sub_conponents = sorted(nx.connected_components(self.G), key=len, reverse=True)

        # print('社区数目',len(sub_conponents)
        center_node = center(self.G.subgraph(0))[0]

        # print('---Component----')

        if len(sub_conponents) > 1: #如果组件不是一个而是多个，就执行
            for i in sub_conponents:  # 合并节点就是每个子图中中心节点连接即可
                if i == sub_conponents[0]: #从第二个组件开始找中心节点，防止出现自循环
                    continue
                sub_G = self.G.subgraph(i)
                sub_center_node = center(sub_G)[0]
                edge = (sub_center_node, center_node)
                self.G.add_edges_from([edge])


        #显示邻接矩阵，黑白格
        self.show_core_periphery(self.G)
        #移除自循环边
        #self.G.remove_edges_from(nx.selfloop_edges(self.G))
        print(self.G.nodes())
        k_shell = nx.k_shell(self.G)
        print(k_shell.nodes())  #打印出k_shell节点编号
        print("数据长度：", self.new_data[2])
        self.draw_g(self.new_data[2])

    def show_core_periphery(self, G):
        """
        画出黑白边缘结构图
        :return:
        """
        A = np.array(nx.adjacency_matrix(G).todense())
        # print("A:", A)
        plt.figure(1, (12, 12))
        # plt.title("Adj_Matrix")
        plt.imshow(A)
        plt.imshow(A, "gray")
        plt.show()

    def generate_delta(self, l1, l2, A):

        """
        :param l1: core nodes length (核心节点个数)
        :param l2: periphery nodes length （边缘节点个数）
        :param A: adjacency_matrix
        :return:
        """
        #print("l1, l2", l1, l2)
        delta1 = np.ones(int(l1))
        delta2 = np.zeros(int(l2))

        delta = np.hstack((delta1, delta2))
        #print(delta.shape)
        Delta = delta.reshape(delta.shape[0], 1)*delta
        #print("Delta:", Delta)
        #print(Delta.shape, A.shape)
        Rho = np.sum(Delta*A)/2   #归一化，因为只需要邻接矩阵一半的值

        return Rho


    def draw_adj_matrix(self, adj_matrix, c_n):
        """

        :param adj_matrix:
        :param c_n: each length of data
        :return:
        """
        m = np.zeros_like(adj_matrix) - 2
        size = adj_matrix.shape[0]
        m[:c_n, :c_n] = int(0)
        m[:c_n, c_n:] = int(1)
        m[c_n:, :c_n] = int(1)

        for i in range(size):
            m[i, i] = -1
        fig, ax = plt.subplots(figsize=(12, 12))

        colors = ['white', '#000000', '#6495ED', '#FF6A6A']
        # ax.matshow(m, cmap=plt.cm.Blues)
        cmap = mpl.colors.ListedColormap(colors)
        ax.matshow(m, cmap=cmap)

        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                v = adj_matrix[j, i]
                ax.text(i, j, int(v), va='center', ha='center')

        plt.show()

    def predict(self, x: np.ndarray, y):

        """
        真正的算法中predict函数中参数不包括 测试标签 y
        Args:
            x: test_data
        Returns:

        """
        y_pred = []
        print("test_x_len:", len(x))
        count = 0
        #x = self.data_preprcess(x)
        for idx, item in enumerate(x):  # 遍历测试数据

            l = y[idx]
            #print("label:", l)

            #idx += len(self.G)  # 新节点编号
            #print("new_idx:", idx)
            item = item.reshape(1, -1)

            idx = len(self.G.nodes()) #新节点label
            #print("idx:", idx)
            dis_matrix = euclidean_distances(item, self.new_data[0])
            min_idx = int(np.argmin(dis_matrix[0]))

            #找到所有小于阈值的节点编号
            edge_idxs = list(np.argwhere(dis_matrix[0] < np.min(self.new_data[1]) * self.init_rate))
            #print(self.min_idx, self.edge_idxs)
            # 添加节点， 添加连边
            test_node = (idx, {'value': None, 'class': 'test', 'type': 'test'})
            self.G.add_nodes_from([test_node])

            edges = [(idx, min_idx)] #防止出现单节点

            #将小于阈值的节点与新节点连接。
            for edge_idx in edge_idxs:
                edges.append((idx, int(edge_idx)))

            self.G.add_edges_from(edges)

            new_label = self.k_shell(self.G, idx)
            #print("label, new_label:", (label, new_label))
            #print("new_node_label:", label)
            y_pred.append(new_label[0])
            #将新节点移除
            self.G.remove_node(idx)

        return np.array(y_pred)

    def check(self, x, y):
        y_hat = self.predict(x, y)  #predict函数中不能有y,此处只是为了验证而已
        print("origanl_y:", y)
        print("predict:", y_hat)
        acc = np.sum(y_hat == y) / len(y)

        return acc, y_hat

    def calculate_net_measures(self, G, data_len, idx):

        rcc = nx.rich_club_coefficient(G, normalized=False, Q=100)
        av_rcc = []

        if idx == []:
            #for each_len in self.each_data_len:
            sum_c0 = [rcc.get(i, 0) for i in range(data_len[0])]
            #print("byebye")
            #print("sum_c0:", sum_c0)
            sum_c1 = [rcc.get(i, 0) for i in range(data_len[0], data_len[0] + data_len[1])]
            #print("sum_c1:", sum_c1)
            #print("over")
            ev_rcc0 = [sum(sum_c0) / data_len[0]]
            ev_rcc1 = [sum(sum_c1) / data_len[1]]
            av_rcc.extend([ev_rcc0, ev_rcc1])

            #fina_rcc.append(sum_c0)
            #fina_rcc.append(sum_c1)

        else:

            print("hello_new_node", "idx:", idx, )
            if idx in rcc.keys():
                av_rcc = [rcc[idx]]
            else:
                av_rcc = [0]
            #av_rcc = [rcc[idx]]

        return av_rcc

    def k_shell(self, G, idx):
        """
        :param G: graph
        :return: 0, 1, 2
        """
        measures = []
        k_shell = nx.k_shell(G)
        #print("k_shell:", k_shell)

        if idx in k_shell.nodes():
            measures.append(0)
        else:
            measures.append(1)

        return measures

    def draw_g(self, len):

        plt.figure("Graph", figsize=(12, 12))
        color_map = {0: 'r', 1: 'b'}
        #plt.title("Normal and Covid-19")
        color_list = []
        for idx, thisG in enumerate(len):
            color_list += [color_map[idx]] * (thisG)
        pos = nx.spring_layout(self.G)   #细长
        nx.draw_networkx(self.G, pos, with_labels=False, node_size=80,
                        node_color=color_list, width=0.1, alpha=1)  #
        #plt.legend(loc="lower right", handlelength=4, borderpad=2, labelspacing=2, fontsize=12)
        plt.show()



