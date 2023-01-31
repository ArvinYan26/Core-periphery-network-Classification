import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import seaborn as sns


def get_data():
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target

    return iris, iris_data, iris_target

def get_breast_canser():
    data = load_breast_cancer()
    return data

# def get_digits():
#     data = load_digits()
#     return data

def data_preprocess():
    iris_data, iris_target = get_data()
    print(iris_data.shape, iris_target.shape)
    X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    return X_train, X_predict, Y_train, Y_predict


def draw_confusion_matrix(y_true, y_pred):
    sns.set()
    f, ax = plt.subplots()
    C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
    #print("confusion_matrix:")  # 打印出来看看
    #print(C2)
    sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('Predict')  # x轴
    ax.set_ylabel('True')  # y轴
    plt.show()

def get_mean_std(data):
    acc_aver = np.mean(data)
    acc_aver = round(acc_aver, 3)
    std = np.std(data)
    std = round(std, 3)

    print("acc_aver, std:", acc_aver, std)

acc_1 = [0.9,0.933,0.833,0.833,0.833,0.933,0.933,0.9,0.9,0.933]
acc_2 =[0.867,0.9,0.867,0.867,0.9,0.867,0.9,0.867,0.867,0.867]


get_mean_std(acc_1)
get_mean_std(acc_2)

# origanl_y = [1., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
#  1., 1., 0., 0., 0., 1.]
# predict= [1., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
#  1., 1., 0., 0., 0., 1.]
# draw_confusion_matrix(origanl_y, predict)









