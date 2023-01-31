import pandas as pd
from sklearn import preprocessing
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
#import cpalgorithm as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def load_csv(data_path):
    """
    读取csv文件
    """
    data_read = pd.read_csv(data_path)
    print(type(data_read))
    list = data_read.values.tolist()
    data = np.array(list)
    # 按列删除矩阵最后一列.
    x = np.delete(data, data.shape[1] - 1, axis=1)
    y = data[0:len(data), data.shape[1] - 1]
    print(type(y), )
    # transposed = list(map(list, zip(*y)))
    # print(x.shape, transposed.shape)
    # y = y.tolist()
    # print(x.shape, len(y))
    return x, y


def data_preprcess(data):
    min_max_scaler = preprocessing.MinMaxScaler().fit(data)
    data = min_max_scaler.transform(data)
    # x_test = min_max_scaler.transform(data)
    return data


def draw_g(G, len):
    color_map = {0: 'r', 1: 'b', 2: 'b', 3: 'g', 4: 'm', 5: 'c', 6: 'black',
                 7: 'grey', 8: 'y', 9: 'magenta'}
    plt.figure("Graph", figsize=(12, 12))
    # Normal and Covid-19, Normal and Viral Pneumonia, Viral Pneumonia and Covid-19
    plt.title("Normal and Covid-19")
    color_list = []
    for idx, thisG in enumerate(len):
        color_list += [color_map[idx]] * (thisG)
    pos = nx.spring_layout(G)  # 细长
    nx.draw_networkx(G, pos, with_labels=False, node_size=60,
                     node_color=color_list, width=0.1, alpha=1)  #
    plt.show()


def show_core_periphery(G):
    """
    画出黑白边缘结构图
    :return:
    """
    A = np.array(nx.adjacency_matrix(G).todense())
    # print("A:", A)
    plt.figure(figsize=(10, 10))
    # plt.title("Adj_Matrix")
    plt.imshow(A)
    plt.imshow(A, "gray")
    plt.yticks(size=15)  # 设置纵坐标字体信息
    plt.xticks(size=15)
    plt.show()


def draw_line(t, Rho, l, G_list):
    plt.figure(figsize=(10, 10))
    # plt.title("Core-periphery Measure")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(t, Rho)
    max_y = max(Rho)  # 找到最大值
    max_index = Rho.index(max_y)
    max_x = round(t[max_index], 3)  # 找到最大值对应的x坐标
    max_G = G_list[max_index]  # 找到最大值时的图G
    max_G_len = l[max_index]  # 训练数据集长度

    """
    median = np.median(Rho)
    median_index = Rho.index(median)
    median_G = G_list[median_index]
    median_G_len = l[median_index]
    """

    # horizontal, values = t[0:max_x+1], [max_y for i in range(max_index+1)]
    plt.plot([max_x, max_x], [0, round(max_y, 3)], 'r--', label='Highest Value')
    plt.legend(loc='lower right', fontsize=40)  # 标签位置
    print("=" * 50)
    # print([min(t), max_x])
    print("最大值横纵坐标：", [max_x, max_y])
    plt.plot([min(t), max_x], [max_y, max_y], 'r--')
    plt.text(max_x, 0, str(max_x), fontsize='x-large')
    plt.text(min(t), max_y, str(max_y), fontsize='x-large')
    plt.legend(loc='best', handlelength=5, borderpad=2, labelspacing=2, fontsize=15)

    plt.yticks(size=15)  # 设置纵坐标字体信息
    plt.xticks(size=15)
    plt.xlabel("Threshold", fontsize=20)
    plt.ylabel("Measure Value", fontsize=20)
    plt.grid(True, linestyle="--", color="g", linewidth="0.5")
    plt.show()

    # return median_G, median_G_len
    return max_G, max_G_len