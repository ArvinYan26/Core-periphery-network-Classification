from sklearn import datasets  # 导入方法类
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from GetCOVID_19Data1 import get_data

# 【1】载入数据集
#iris = datasets.load_iris()  # 加载 iris 数据集
#iris_feature = iris.data  # 特征数据
#iris_target = iris.target  # 分类数据

# 【2】数据集划分
#feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
#                                                                          random_state=42)
x_train, x_test, y_train, y_test = get_data(percent=0.9)
# 【3】训练模型
dt_model = DecisionTreeClassifier()  # 所有参数均置为默认状态
dt_model.fit(x_train, y_train)  # 使用训练集训练模型
predict_results = dt_model.predict(x_test)  # 使用模型对测试集进行预测

# 【4】结果评估
scores = dt_model.score(x_test, y_test)
print('准确度: %.2f%%' % (100 * scores))