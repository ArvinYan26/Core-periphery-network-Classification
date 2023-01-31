import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_blobs, load_digits
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from sklearn.preprocessing import Normalizer
import networkx as nx
import math
import pandas as pd
from networkx.algorithms.distance_measures import center as nxCenter
from collections import Counter
from sklearn.model_selection import train_test_split
#from GetCOVID_19Data import get_data
from GetCOVID_19Data1 import get_data  #原图像傅里叶变换，两类（正常和新冠）
from sklearn import preprocessing
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
#import cpalgorithm as cp
# import igraph as ig
import matplotlib as mpl


class NetworkBaseModel():

    def __init__(self):
        '''

        :param per_class_data_len:  length of per class data
        :param num_classes:         num of classes
        :param k:                   para of kNN
        '''
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

    def get_params(self, in_rate, c_rate):
        """

        :param in_rate:init_threshold_rate
        :param c_rate: classicfication_threshold_rate
        :return:
        """
        #return {'k': self.k, 'num_class': self.class_num}
        self.init_rate = in_rate
        self.c_r = c_rate

    def data_preprcess(self, data):
        min_max_scaler = preprocessing.MinMaxScaler().fit(data)
        new_data = min_max_scaler.transform(data)
        return new_data

    def fit(self, x: np.ndarray, y:np.ndarray):
        """

        Args:
            x: array (n, m) 输入数据
            y: (n)
        Returns: predict_label

        """

        self.train_x = x
        self.train_y = y
        self.train_len = len(x)

        labels = [i for i in Counter(y)]
        labels.sort()
        self.labels = labels
        #print("self.labels:", self.labels)
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
            data_len = len(dataset)
            self.each_data_len.append(data_len)
            adj_matrix = euclidean_distances(dataset, dataset)
            # 要先求两两平均距离，后面会改动数据。
            mean_dis = np.sum(adj_matrix) / (data_len ** 2 - data_len)
            #print("mean_dis:", mean_dis)
            #mean_dis = mean_dis*self.t #变化阈值
            self.mean_dis_list.append(mean_dis)  #平均差别

        if not self.mean_dis_list == []:
            self.mean_dis_list = sorted(list(set(self.mean_dis_list)))
        print("self.mean_dis_list:", self.mean_dis_list)
        #print(len(self.data))
        #三类
        #self.new_data = np.vstack((self.data[0], self.data[1], self.data[2]))
        #正常和新冠
        self.each_data_len.remove(self.each_data_len[1])
        self.new_data = np.vstack((self.data[0], self.data[2]))
        #正常和肺炎
        #self.each_data_len.remove(self.each_data_len[2])
        #self.new_data = np.vstack((self.data[0], self.data[1]))
        #肺炎和新冠d
        #self.each_data_len.remove(self.each_data_len[0])
        #self.new_data = np.vstack((self.data[1], self.data[2]))
        #print(new_data.shape)
        adj_matrix = euclidean_distances(self.new_data, self.new_data)
        adj_matrix[adj_matrix == 0] = 999
        #没一个节点最小距离找到构建连边，防止单节点出现
        for idx, item in enumerate(adj_matrix):
            min_idx = np.argmin(item)
            # 因为是对称矩阵
            adj_matrix[idx, min_idx] = 1
            adj_matrix[min_idx, idx] = 1
        #小于阈值的设置为1即连边
        adj_matrix[adj_matrix < np.median(self.mean_dis_list) * self.init_rate] = 1
        # 将没有连边的部分都设置为0
        adj_matrix[adj_matrix != 1] = 0
        #self.G_list.append(nx.from_numpy_matrix(adj_matrix))
        self.G = nx.from_numpy_matrix(adj_matrix)

        sub_conponents = sorted(nx.connected_components(self.G), key=len, reverse=True)

        # print('社区数目',len(sub_conponents))
        center_node = nxCenter(self.G.subgraph(0))[0]

        # print('---Component----')

        for i in sub_conponents:  # 合并节点就是每个子图中中心节点连接即可

            sub_G = self.G.subgraph(i)

            sub_center_node = nxCenter(sub_G)[0]
            edge = (sub_center_node, center_node)

            self.G.add_edges_from([edge])

        #print(self.G.edges())d
        #print(len(self.G.nodes()), self.G.nodes())
        self.draw_g()
        #self.aver_c = self.calculate_net_measures(self.G, [])
        #print("aver_c:", self.aver_c)
        #k_core = nx.k_core(self.G)
        """
        k_shell = nx.k_shell(self.G)
        print("k_core,k_shell:", k_shell)
        print(k_shell.nodes())
        degree = nx.degree(k_shell)
        print("degree:", degree)
        """
        #将图转化为邻接矩阵
        A = np.array(nx.adjacency_matrix(self.G).todense())
        #print("A:", A)
        plt.figure(1, (12, 12))
        #plt.title("Adj_Matrix")
        plt.imshow(A)
        plt.imshow(A, "gray")
        #画出0，1棋盘格
        #self.draw_adj_matrix(adj_matrix, self.each_data_len[0])
        Rho = self.generate_delta(self.each_data_len[0], self.each_data_len[1], A)
        #print("Rho:", Rho)
        """
        A = np.array(nx.adjacency_matrix(self.G).todense())
        print("A:", A)
        plt.figure(2, (12, 12))
        #plt.title("Adj_Matrix")
        plt.imshow(A)
        plt.show()
        """
        #closess = self.calculate_net_measures(self.G)

        return Rho, self.each_data_len

    def generate_delta(self, l1, l2, A):
        """

        :param l1: core nodes length (核心节点个数)
        :param l2: periphery nodes length （边缘节点个数）
        :param A: adjacency_matrix
        :return:
        """
        delta1 = np.ones(l1)
        delta2 = np.zeros(l2)

        delta = np.hstack((delta1, delta2))
        Delta = delta.reshape(delta.shape[0], 1)*delta
        #print("Delta:", Delta)
        Rho = np.sum(Delta*A)

        return Rho

    def draw_adj_matrix(self, adj_matrix, c_n):
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
            print("label:", l)

            idx += len(self.G)  # 新节点编号
            print("new_idx:", idx)
            item = item.reshape(1, -1)
            count += 1
            new_mesures = []

            for i in self.labels:

                dis_matrix = euclidean_distances(item, self.data[i])
                min_idx = int(np.argmin(dis_matrix[0]))
                edge_idxs = list(np.argwhere(dis_matrix[0] < np.median(self.mean_dis_list) * self.c_r))

                if i == 0:
                    self.min_idx, self.edge_idxs = min_idx, edge_idxs
                if i == 1:
                    self.min_idx = len(self.data[0]) + min_idx
                    self.edge_idxs = [len(self.data[0])+j for j in edge_idxs]
                if i == 2:
                    self.min_idx = (len(self.data[0])+len(self.data[1])) + min_idx
                    self.edge_idxs = [(len(self.data[0])+len(self.data[1])) + j for j in edge_idxs]

                print(self.min_idx, self.edge_idxs)
                # 添加节点， 添加连边
                test_node = (idx, {'value': None, 'class': 'test', 'type': 'test'})
                self.G.add_nodes_from([test_node])

                edges = [(idx, self.min_idx)]
                for edge_idx in self.edge_idxs:
                    edges.append((idx, int(edge_idx)))

                self.G.add_edges_from(edges)

                new_node_m = self.calculate_net_measures(self.G, idx)
                new_mesures.append(new_node_m)
                #将新节点移除
                self.G.remove_node(idx)

            print(self.aver_c, new_mesures)
            diff = paired_euclidean_distances(new_mesures, self.aver_c)

            label = int(np.argmin(diff))
            print("p_label:", label)
            y_pred.append(label)   #返回数组中最小值得索引
            print("="*100)

        return np.array(y_pred)

    #不需要将新节点再插入到新网络中
    def classification(self, label, idx, neighbor):
        for ith_class in range(self.num_classes):
            if ith_class == label:
                self.G_list[ith_class].add_node(idx, value=None, typeNode="test")
                self.G_list[ith_class].add_edges_from(neighbor[ith_class])



    def check(self, x, y):
        y_hat = self.predict(x, y)  #predict函数中不能有y,此处只是为了验证而已
        print("origanl_y:", y)
        print("predict:", y_hat)
        acc = np.sum(y_hat == y) / len(y)
        #con_m = confusion_matrix(y, y_hat, labels=[0, 1, 2])
        con_m = self.draw_confusion_matrix(y, y_hat)
        #print("con_m:")
        #print(con_m)

        return acc, con_m

    def draw_confusion_matrix(self, y_true, y_pred):
        sns.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        #print("confusion_matrix:")  # 打印出来看看
        #print(C2)
        sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('Predict')  # x轴
        ax.set_ylabel('True')  # y轴
        plt.show()

        return C2

    def calculate_net_measures(self, net, idx):


        #degree_assortativity = nx.degree_assortativity_coefficient(G=net)

        #average_clustering_coefficient = nx.average_clustering(G=net)
        #average_degree = np.mean([i[1] for i in nx.degree(net)])  # nx.degree 返回的是每个节点的度, 所以要获取再求平均
        #dimameter = nx.algorithms.distance_measures.diameter(net)
        #effic = nx.global_efficiency(G=net)
        #den = nx.density(G=net)
        cc = nx.closeness_centrality(net)
        #cc  = nx.katz_centrality(net)
        #cc = nx.betweenness_centrality(net)
        fina_cc = []
        if idx == []:
            #for each_len in self.each_data_len:
            sum_c0 = [cc.get(i, 0) for i in range(self.each_data_len[0])]
            sum_c1 = [cc.get(i, 0) for i in range(self.each_data_len[0], self.each_data_len[0] + self.each_data_len[1])]
            #sum_c2 = [cc.get(i, 0) for i in range(self.each_data_len[0] + self.each_data_len[1], len(self.G.nodes()))]
            #ever_cc0 = [sum(sum_c0) / self.each_data_len[0]]
            #fina_cc.append(ever_cc0)
            #ever_cc1 = [sum(sum_c1) / self.each_data_len[1]]
            #fina_cc.append(ever_cc1)
            #ever_cc2 = [sum(sum_c2) / self.each_data_len[2]]
            #fina_cc.append(ever_cc2)
            fina_cc.append(sum_c0)
            fina_cc.append(sum_c1)

        else:
            fina_cc = [cc[idx]]

        return fina_cc


    def draw_g(self):
        color_map = {0: 'r', 1: 'b', 2: 'b', 3: 'g', 4: 'm', 5: 'c', 6: 'black',
                    7: 'grey', 8: 'y', 9: 'magenta'}
        plt.figure("Graph", figsize=(12, 12))
        #Normal, Viral Pneumonia and Covid-19
        plt.title("Normal and Covid-19")
        #G = nx.disjoint_dunion_all(self.G_list)
        color_list = []
        for idx, thisG in enumerate(self.each_data_len):
            color_list += [color_map[idx]] * (thisG)

        pos = nx.spring_layout(self.G)   #细长
        #pos = nx.circular_layout(self.G)  #圆圈
        #pos = nx.shell_layout(self.G)   #圆圈
        #pos = nx.spectral_layout(self.G) #直线
        #pos = nx.random_layout(self.G)  #随机分布
        nx.draw_networkx(self.G, pos, with_labels=False, node_size=80,
                         node_color=color_list, width=0.1, alpha=1)  #

        plt.show()


if __name__ == '__main__':

    start = time.time()

    def data_preprcess(x_train, x_test):
        min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_train = min_max_scaler.transform(x_train)
        x_test = min_max_scaler.transform(x_test)
        return x_train, x_test

    #covid-19数据集
    x, y = get_data()    #covid-19数据集 label：0：normal， 1：P， 2：covid-19

    """
    features = list(df.columns)
    features = features[: len(features) - 1]  # 去掉开头和结尾的两列数据
    x = df[features].values.astype(np.float32)
    y = np.array(df.target)
    """
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train, x_test = data_preprcess(x_train, x_test)  # 数据归一化
    threshold = 1
    model = NetworkBaseModel(threshold)
    model.get_params(0.87, 0.87)
    model.fit(x_train, y_train)
    acc, con_m = model.check(x_test, y_test)
    print("acc:", acc)
    """
    a = []
    con_matrix = []
    #for i in range(10):
    #print("frequency:", i)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train, x_test = data_preprcess(x_train, x_test) #数据归一化
    model = NetworkBaseModel()
    t = np.arange(0.82, 1.5, 0.02)
    #print("t:", t)
    Rho = []
    for i in t:
        model.get_params(i, 0.88)
        rho, l = model.fit(x_train, y_train)
        Rho.append(rho)
    """
    #画节点cloness折线图
    if not len(closess[0]) == len(closess[1]):
        if len(closess[0]) > len(closess[1]):
            closess[0].pop()
            if len(closess[0]) == len(closess[1]):
                break
    """

    x_axis = [i for i in range(len(t))]
    plt.figure()
    plt.title("Core-periphery Measure")
    plt.plot(t, Rho)
    plt.xlabel("Threshold")
    plt.ylabel("Measure Value")
    plt.grid(True, linestyle="--", color="g", linewidth="0.5")
    plt.show()
    #acc, con_m = model.check(x_test, y_test)
    """
    print("acc and con_m:", acc, con_m)

    a.append(acc)
    con_matrix.append(con_m)

    #打印最终结果
    print("final:", a, con_matrix)
    mean_acc = np.mean(a)
    max = np.max(a)
    min = np.min(a)
    var = np.var(a)
    print("%f +- %f", (mean_acc, var))
    print(min, max)
    """
    end = time.time()
    print("time:", end - start)

