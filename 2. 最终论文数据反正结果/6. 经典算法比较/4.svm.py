# 文件功能：svm分类鸢尾花数据集
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from GetCOVID_19Data1 import get_data
import numpy as np
from sklearn.svm import SVC
"""
# 【1】读取数据集  
data = load_iris()

# 【2】划分数据与标签  
x = data.data[:, :2]
y = data.target
train_data, test_data, train_label, test_label = train_test_split \
    (x, y, random_state=1, train_size=0.6, test_size=0.4)
print(train_data.shape)
"""

x_train, x_test, y_train, y_test = get_data(percent=0.8)
# 【3】训练svm分类器  
#classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  #  ovr:一对多策略  
classifier = svm.SVC()  #  ovr:一对多策略  
ir = classifier.fit(x_train, y_train.ravel())  # ravel函数在降维时默认是行序优先  

"""
# 【4】计算分类器的准确率  
print("训练集：", classifier.score(x_train, y_train))
print("测试集：", classifier.score(x_test, y_test))
"""

# 【5】可直接调用accuracy_score计算准确率  
#tra_label = classifier.predict(y_train)  # 训练集的预测标签  
#tes_label = classifier.predict(y_test)  # 测试集的预测标签  
#print("训练集：", accuracy_score(y_train, tra_label))
y_hat1 = ir.predict(x_test)
result = y_hat1 == y_test
print(result)
acc = np.mean(result)
print('准确度: %.2f%%' % (100 * acc))

svm_clf = SVC()
svm_clf.fit(x_train, y_train)
print('SVM accuracy={}'.format(svm_clf.score(x_test, y_test)))
#print("测试集：", accuracy_score(y_test, tes_label))


"""
# 【6】查看决策函数  
print('train_decision_function:\n', classifier.decision_function(y_train))  #  (90,3)  
print('predict_result:\n', classifier.predict(y_train))
"""