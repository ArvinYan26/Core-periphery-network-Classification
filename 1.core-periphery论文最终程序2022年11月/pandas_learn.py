import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# a = [1, 2, 3]
# b = [4, 5, 6]
# dataframe = pd.DataFrame({"a":a, "b":b})
# dataframe.to_csv("test.csv", index=False, sep=",")


def test_panda():
    a = [['2', '1.2', '4.2', '4', '5', '7', '0'], ['0', '10', '0.3', '4', '5', '7', '1'], ['1', '5', '0', '4', '5', '7', '2']]
    a = np.array(a)
    print(a)
    df = pd.DataFrame(a, columns=['100', '110', '120', '130', '140', '150', 'target'])
    print(df)
    df.to_csv("E:/PycharmProjects/1.Python Fundamental Programme/2. 2021年学习/4.论文扩展程序总结出版/data.csv", sep=",")


def load_csv(path):
    """
    读取csv文件
    """
    data_read = pd.read_csv(path)
    print(type(data_read))
    list = data_read.values.tolist()
    data = np.array(list)
    # 按列删除矩阵最后一列.
    x = np.delete(data, data.shape[1]-1, axis=1)
    y = data[0:len(data), data.shape[1]-1]
    print(type(y), )
    # transposed = list(map(list, zip(*y)))
    # print(x.shape, transposed.shape)
    # y = y.tolist()
    # print(x.shape, len(y))
    return data

path = "E:/Datasets/covid-19/COVID-19-c/fractral_dataset/fd_qt.csv"
# data = load_csv(path)
# print(type(data))
#

data_iris = load_iris()
x = data_iris.data
y = data_iris.target
print(type(data_iris.target), y.shape)
li = []
for i in range(10):
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1, stratify=y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)
    li.append(x_train[0])
print(li)