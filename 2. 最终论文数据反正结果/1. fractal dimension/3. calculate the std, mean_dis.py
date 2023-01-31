import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import entropy
from GetCOVID_19Data1 import get_data



def data_segmentation(x, y, p):
    """
    :param p:切分比例
    :return:
    """
    l = int(len(x)/2)
    data0, data1 = np.array(x[:l]), np.array(x[l:])
    y0, y1 = y[:l], y[l:]

    #计算每类数据集的平均距离的分散度
    m = []
    s = []
    e = []
    dic = {}
    for data in [data0, data1]:
        m_dis, st, en = calculate(data)
        m.append(m_dis)
        s.append(st)
        e.append(en)

    dic["mean_dis"] = m
    dic["std"] = s
    dic["entroy"] = e
    #print("y0,y1", y0, y1)
    p = int(len(data0) * p)  #训练集长度
    x_train = np.vstack((data0[:p], data1[:p]))
    #训练集target
    #print("y0,y1", y0[:p], y1[:p])
    #print("y0,y1", y0[:p].shape, y1[:p].shape)
    y_train = np.array(list(y0[:p]) + list(y1[:p]))
    #print("y_train:", y_train)

    #测试集数据
    x_test = np.vstack((data0[p:], data1[p:]))
    y_test = np.array(y0[p:] + y1[p:])
    return x_train, x_test, y_train, y_test, dic



def data_preprcess(x):
    min_max_scaler = preprocessing.MinMaxScaler().fit(x)
    x_new = min_max_scaler.transform(x)

    return x_new

def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def calculate(dataset):

    data_len = len(dataset)
    adj_matrix = euclidean_distances(dataset, dataset)
    m_dis = np.sum(adj_matrix) / (data_len ** 2 - data_len)
    st = np.std(adj_matrix)  # 计算距离矩阵的标准差
    x = list(adj_matrix.reshape(1, -1)[0])
    en = entropy(x)

    return m_dis, st, en


if __name__ == '__main__':

    #df = pd.read_csv(r"C:\Users\Yan\Desktop\dimension_100_170_10.csv")
    """
    features = list(df.columns)
    features = features[: len(features) - 1]  # 去掉开头和结尾的两列数据
    #print(features)
    x = df[features].values.astype(np.float32)
    #print(type(x))
    y = np.array(df.target)
    #print("y:", y)
    """
    x, y  = get_data()
    x = data_preprcess(x)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train, x_test, y_train, y_test, dic = data_segmentation(x, y, p=0.9)
    #print("data:", x_train.shape, x_test.shape, y_train, y_test)
    #each_data = data_reorganization(x_train, y_train)
    print("each_data:", dic)