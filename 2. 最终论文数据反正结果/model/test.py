import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalization(X):
    """X : ndarray 对象"""
    min_ = X.min(axis=0)        # 得到的是 Series 对象
    max_ = X.max(axis=0)
    row_num = X.shape[0]        # row_num 是X的行数
    ranges = max_ - min_
    molecule = X - np.tile(min_, (row_num, 1))     # np.tile(A, (row, col) ) :把A(看做整体)复制row行，col 列,分子:X-min
    denominator = np.tile(ranges, (row_num, 1))    # 分母 max-min
    X = molecule/denominator
    return X, np.array(min_), np.array(ranges)


def normalization2(X):
    minmax = MinMaxScaler()  # 对象实例化
    X = minmax.fit_transform(X)
    return X


# data = pd.DataFrame({0: [1, 2, 3, 4, 5, 8]})
# data = np.array([4318.0, 4506.0, 4708.0, 4877.0, 5046.0, 5215.0, 5385.0, 5533.0, 5680.0, 5836.0, 6012.0, 6142.0, 6270.0, 6399.0, 6519.0])
# data = np.array([0, 2000, 4000, 6000, 8000, 9453, 10000])
# # print(data)
# # X, x, y = normalization(np.array(data))
# # print(X)
# # print(x)
# # print(y)
#
# X = normalization2(data.reshape(-1, 1)).reshape(1, -1)
# X = list(X[0])
# print("X:", X)

axis_1 = [0, 2000, 4000, 6000, 8000, 9453]
result = [round(x/18225, 2) for x in axis_1]
print(result)