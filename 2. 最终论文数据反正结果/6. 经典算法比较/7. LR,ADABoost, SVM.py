import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from GetCOVID_19Data1 import get_data

warnings.filterwarnings('ignore')

"""
X, y = make_circles(n_samples=300, noise=0.15, factor=0.5, random_state=233)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show(x_train, X_test, y_train, y_test = train_test_split(X, y)
"""
x_train, x_test, y_train, y_test = get_data(percent=0.9)
print('X_train.shape=', x_train.shape)
print('X_test.shape=', x_test.shape)
print(y_test)
print('===========knn==============')
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)
print('knn accuracy={}'.format(knn_clf.score(x_test, y_test)))
print('\n')
print('===========logistic regression==============')
log_clf = LogisticRegression()
log_clf.fit(x_train, y_train)
print('logistic regression accuracy={}'.format(log_clf.score(x_test, y_test)))
print('\n')
print('===========SVM==============')
svm_clf = SVC()
svm_clf.fit(x_train, y_train)
print('SVM accuracy={}'.format(svm_clf.score(x_test, y_test)))
print('\n')
print('===========Decison tree==============')
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)
print('Decison tree accuracy={}'.format(dt_clf.score(x_test, y_test)))
print('\n')

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(),
                             n_estimators=500,
                             learning_rate=0.3)

ada_clf.fit(x_train, y_train)
print('Adaboost accuracy={}'.format(ada_clf.score(x_test, y_test)))
print('\n')