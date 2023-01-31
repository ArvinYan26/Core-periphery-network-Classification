import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from GetCOVID_19Data1 import get_data  #调用数据集
from sklearn import preprocessing
import pandas as pd

def data_preprcess(x_train, x_test):
    min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
    x_train = min_max_scaler.transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    return x_train, x_test

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

def compare_algorithm(X_test, y_test, times):
    h = .02  # step size in the mesh

    names = ["Linear SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes"]

    classifiers = [
        SVC(kernel="linear", C=0.025),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB()]
    # iterate over classifiers
    score_dict = dict()
    list_acc = []
    for i in range(times):
        score_list = []
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            # score_dict[name] = score

            score_list.append(score)
        # print("score_list:", score_list)
        list_acc.append(score_list)
    # print("list_acc:", list_acc)

    acc_mean, std = None, None
    if list_acc:
        list_acc = np.array(list_acc)

        acc_mean = np.average(list_acc, axis=0)
        std = list(np.std(list_acc, axis=0))
        std = [round(i, 3) for i in std]
        dict_acc = dict()
    # if acc_mean:
    #     for i in len(acc_mean[0]):
    #         dict_acc[names[i]] = acc_mean[0][i] + "+-" std[0][i]

    return acc_mean, std, list_acc

if __name__ == '__main__':
    # 读取csv文件划分数据集
    data_path = "E:/Datasets/covid-19/COVID-19-c/fractral_dataset/fd_qt.csv"
    x, y = load_csv(data_path)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.1, stratify=y)
    X_train, X_test = data_preprcess(X_train, X_test)
    # list_acc = []
    times = 30
    acc_mean, std, list_acc = compare_algorithm(X_test, y_test, times=times)
    print("times:", str(times) + "次")
    print("np.list_acc:", list_acc)
    print("acc_mean:", acc_mean)
    print("std:", std)