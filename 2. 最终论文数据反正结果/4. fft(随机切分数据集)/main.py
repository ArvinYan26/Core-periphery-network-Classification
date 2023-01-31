from CorePeriphery import CorePeriphery
from GetCOVID_19Data1 import get_data  #原图像傅里叶变换，两类（正常和新冠）
from sklearn import preprocessing
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

class DataReorganization(object):
    def __init__(self):

        self.data_idxs_list = []
        self.num_classes = None
        #self.each_data_len = []
        self.mean_dis_list = []


    def data_preprcess(self, x_train, x_test):
        min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_train = min_max_scaler.transform(x_train)
        x_test = min_max_scaler.transform(x_test)
        return x_train, x_test


    def draw_confusion_matrix(self, y_true, y_pred):
        sns.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
        #print("confusion_matrix:")  # 打印出来看看
        #print(C2)
        sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

        #ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('Predict', fontsize=15)  # x轴
        ax.set_ylabel('True', fontsize=15)  # y轴
        plt.show()

if __name__ == '__main__':

    start = time.time()
    dr = DataReorganization()  # 调用模块
    #x_train, y_train, x_test, y_test, p = get_data(percent=0.9) #p:训练集长度
    x, y = get_data() #p:训练集长度

    #print("训练集测试集长度：", len(x_train), len(y_train), len(x_test), len(y_test))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train, x_test = dr.data_preprcess(x_train, x_test)  # 数据归一化
    cp = CorePeriphery(num_class=2, in_rate=1.14)  #调用模块, 类别数
    #G_list, adj = cp.fit(x_train, y_train)
    cp.fit(x_train, y_train)
    acc, y_predict = cp.check(x_test, y_test)
    print("acc:", acc)
    dr.draw_confusion_matrix(y_test, y_predict)


