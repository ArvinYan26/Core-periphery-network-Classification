import numpy as np

from utils import *
from CorePeriphery import NetworkBaseModel
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    start = time.time()

    #covid-19数据集
    # x, y = get_data()    #covid-19数据集 label：0：normal， 1：P， 2：covid-19
    # 读取csv文件划分数据集
    data_path = "E:/Datasets/covid-19/COVID-19-c/fractral_dataset/fd_qt.csv"
    x, y = load_csv(data_path)
    # 为验证模型的参数的可行性，每次划分训练集和数据集相同
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
    # stratify=y 保证每一次都是按照与原数据集相同的每个类别比例
    count = 0
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)
        x_train, x_test = data_preprcess(x_train), data_preprcess(x_test) #数据归一化
        model = NetworkBaseModel()
        #其他两种阈值
        #t = np.arange(0.80, 1, 0.01)
        #肺炎和新冠阈值
        #t = np.arange(0.92, 1.12, 0.02)  #阈值范为(正常和新冠，正常和肺炎)
        # t = np.arange(0.9, 1.2, 0.02)  #阈值范为(fft特征阈值范围)
        t = np.arange(0, 4, 0.2)  #阈值范为(fft特征阈值范围)
        # t = np.arange(0.3, 5, 0.1)  #阈值范为(fractral特征阈值范围)
        #print("t:", t)
        Rho = []
        G_list = []
        data_len = []
        data = None
        for i in t:
            model.get_params(i, 0.88)
            rho, l, g, new_data = model.fit(x_train, y_train) #l:每一次得到的切分的每类数据集长度
            Rho.append(rho)
            G_list.append(g)
            data_len.append(l)
            if i == len(t)-1:
                data = new_data

        """
        #画节点cloness折线图
        if not len(closess[0]) == len(closess[1]):
            if len(closess[0]) > len(closess[1]):
                closess[0].pop()
                if len(closess[0]) == len(closess[1]):
                    break
        """

        print("Rho_before:", Rho)
        Rho = np.array(Rho).reshape(-1, 1)
        print("Rho_array:", Rho)
        Rho = data_preprcess(Rho).reshape(1, -1)
        Rho = list(Rho[0])
        print("Rho_after:", Rho)

        max_G, max_len = draw_line(t, Rho, data_len, G_list) #画出变化的measures， max_G此时是图
        A = np.array(nx.adjacency_matrix(max_G).todense()) #将图转化为邻接矩阵
        edges = []   #边缘节点内部连接的边
        #max_len :此时是一个包含很多组相同元素的列表（暂时无法删除重复元素）

        """
        print(data_len[0])  #新数据集的长度
        for i in max_G.nodes:
            for j in max_G.nodes:
                if i >= data_len[0][0] and j >= data_len[0][0]:
                    #edges.extend([(i, j), (j, i)])
                    A[i][j] = A[j][i] = 0
                #if sum(A[i]) == 0:    #将单边缘节点和核心点连接起来。只连接一个边
                    #print(i, A[i])
                    #A[i][0] = A[0][i] = 1
        """
        #max_G.remove_edges_from(edges)
        G = nx.from_numpy_matrix(A)
        draw_g(G, max_len)
        #将图转化为邻接矩阵
        #A = np.array(nx.adjacency_matrix(max_G).todense())
        show_core_periphery(G)
        # acc, con_m = model.check(x_test, y_test)
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
        count += 1
        # if count == 1:
        #     break